"""
DDP Evaluator for Multi-GPU Evaluation

Based on the original generate_ddp.py implementation.
Uses the new modular validation infrastructure with distributed processing.
"""

import os
import json
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

from validation.base_models import create_model
from validation.evaluator import Evaluator


def setup(rank, world_size):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Set NCCL environment variables to reduce cleanup warnings
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'

    # Initialize process group with timeout
    import datetime
    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)


def cleanup_resources(rank, model=None):
    """Clean up resources before exiting."""
    try:
        # Clean up model resources
        if model is not None:
            try:
                model.cleanup()
            except Exception as e:
                print(f"Rank {rank}: Warning during model cleanup: {e}")

            del model

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(rank)
    except Exception as e:
        print(f"Rank {rank}: Warning during resource cleanup: {e}")


def evaluate_subset(
    rank: int,
    world_size: int,
    dataset,
    models_config: List[Dict],
    output_dir: Path,
    evaluation_method: str,
    parser_fn,
    max_questions: int = None
):
    """
    Evaluate a subset of questions on a single GPU rank using true DDP.

    All GPUs load the same model and process different subsets of questions.
    After finishing all questions for a model, all GPUs synchronize and move to the next model.
    """
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model = None

    try:
        # Each rank writes to its own file
        rank_output_path = output_dir / f"checkpoint_rank_{rank}.json"

        # Load existing results from global checkpoint
        global_finished = set()
        global_checkpoint = output_dir / "checkpoint.json"
        if global_checkpoint.exists():
            with open(global_checkpoint, 'r') as f:
                global_data = json.load(f)
                global_finished = {(r['model_name'], r['question_id']) for r in global_data}

        # Load existing results from rank checkpoint
        local_results = []
        if rank_output_path.exists():
            with open(rank_output_path, 'r') as f:
                local_results = json.load(f)

        local_finished = {(r['model_name'], r['question_id']) for r in local_results}

        if rank == 0:
            print(f"Loaded {len(global_finished)} existing global results")
            print(f"Loaded {len(local_finished)} existing local results")

        # Get all questions to process
        all_questions = list(range(len(dataset)))
        if max_questions is not None:
            all_questions = all_questions[:max_questions]

        # Process each model sequentially (all GPUs work on same model)
        for model_cfg in models_config:
            model_name = model_cfg['name']

            # Find all unfinished questions for this model
            questions_for_model = [
                idx for idx in all_questions
                if (model_name, dataset[idx]['question_id']) not in global_finished
                and (model_name, dataset[idx]['question_id']) not in local_finished
            ]

            if rank == 0:
                print(f"\n{'='*80}")
                print(f"Model: {model_name}")
                print(f"Total questions for this model: {len(questions_for_model)}")
                print(f"{'='*80}")

            if len(questions_for_model) == 0:
                if rank == 0:
                    print(f"Skipping {model_name} - all questions completed")
                # Synchronize before moving to next model
                dist.barrier()
                continue

            # Split questions for this model across all GPUs (true DDP)
            my_questions = questions_for_model[rank::world_size]

            if rank == 0:
                print(f"Each GPU will process ~{len(questions_for_model)//world_size} questions")
                print(f"Rank {rank}: {len(my_questions)} questions")

            # Clean up previous model
            if model is not None:
                cleanup_resources(rank, model)
                model = None

            # All ranks load the same model on their respective GPUs
            device_str = f"cuda:{rank}"
            if rank == 0:
                print(f"Loading {model_name} on all {world_size} GPUs...")
            model = create_model(model_cfg, device=device_str)

            # Synchronize after model loading to ensure all GPUs are ready
            dist.barrier()
            if rank == 0:
                print(f"All GPUs ready, starting evaluation...")

            # Create evaluator for this rank
            evaluator = Evaluator(
                output_dir=output_dir,
                checkpoint_interval=10,  # Save every 10 questions
                evaluation_method=evaluation_method
            )

            # Load local results into evaluator
            evaluator.results = local_results.copy()

            # Get model type for context loading
            model_type = model_cfg.get('type', 'antoni_alpha')

            # Process questions assigned to this rank
            iterator = tqdm(my_questions, desc=f"GPU {rank}: {model_name}", position=rank) if rank == 0 else my_questions

            for q_idx in iterator:
                question_item = dataset[q_idx]

                try:
                    # Load context
                    context = dataset.load_context(question_item, model_type)

                    # Evaluate single question
                    evaluator._evaluate_single_question(
                        model=model,
                        question_item=question_item,
                        context=context,
                        model_name=model_name,
                        eval_method=evaluation_method,
                        parser_fn=parser_fn
                    )

                    # Save checkpoint to rank file periodically
                    if len(evaluator.results) % 10 == 0:
                        with open(rank_output_path, 'w') as f:
                            json.dump(evaluator.results, f, indent=2)

                except Exception as e:
                    print(f"GPU {rank} Error processing question {question_item['question_id']}: {e}")
                    # Add failed result
                    evaluator.add_result(
                        model_name=model_name,
                        question_item=question_item,
                        model_response=f"ERROR: {str(e)}",
                        extracted_answer=None,
                        is_correct=False,
                        evaluation_method=evaluation_method,
                        error=str(e)
                    )

            # Save final checkpoint for this model
            with open(rank_output_path, 'w') as f:
                json.dump(evaluator.results, f, indent=2)

            # Update local results
            local_results = evaluator.results

            # Synchronize all ranks after finishing this model
            if rank == 0:
                print(f"\nGPU {rank}: Finished {model_name}, waiting for other GPUs...")
            dist.barrier()

            if rank == 0:
                print(f"All GPUs finished {model_name}, moving to next model...")

        # Final synchronization
        if rank == 0:
            print(f"\nGPU {rank}: Finished all models, waiting for other GPUs...")
        dist.barrier()

        if rank == 0:
            print(f"All GPUs finished processing")

    finally:
        # Clean up resources
        cleanup_resources(rank, model)

        # Synchronize before destroying process group
        try:
            dist.barrier()
        except:
            pass

        # Destroy process group
        try:
            if rank != 0:
                import time
                time.sleep(0.5)
            dist.destroy_process_group()
        except Exception as e:
            print(f"GPU {rank}: Warning during process group cleanup: {e}")


def merge_results(output_dir: Path, world_size: int):
    """Merge results from all ranks into single checkpoint."""
    final_results = []

    # Load existing global checkpoint
    global_checkpoint = output_dir / "checkpoint.json"
    if global_checkpoint.exists():
        with open(global_checkpoint, 'r') as f:
            final_results = json.load(f)

    print(f"\nInitial global results: {len(final_results)}")

    # Create a set of existing (model, question_id) pairs
    existing = {(r['model_name'], r['question_id']) for r in final_results}

    new_additions = 0

    # Merge rank files
    for rank in range(world_size):
        rank_file = output_dir / f"checkpoint_rank_{rank}.json"
        if rank_file.exists():
            with open(rank_file, 'r') as f:
                rank_data = json.load(f)
                for result in rank_data:
                    key = (result['model_name'], result['question_id'])
                    if key not in existing:
                        final_results.append(result)
                        existing.add(key)
                        new_additions += 1

    # Save merged results
    with open(global_checkpoint, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Merged {new_additions} new results. Total: {len(final_results)}")
    print(f"Merged results saved to {global_checkpoint}")


def run_ddp_evaluation(
    dataset,
    models_config: List[Dict],
    output_dir: Path,
    evaluation_method: str = "generation",
    parser_fn = None,
    max_questions: int = None
):
    """
    Run evaluation using DDP across multiple GPUs.

    Args:
        dataset: Dataset instance (VQADataset)
        models_config: List of model configurations
        output_dir: Output directory for results
        evaluation_method: "generation" or "perplexity"
        parser_fn: Parser function for answer extraction
        max_questions: Maximum number of questions to evaluate (for testing)
    """
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices found. DDP requires GPUs.")
        return

    print(f"\n{'='*80}")
    print(f"DDP EVALUATION: {world_size} GPUs")
    print(f"{'='*80}")
    total_questions = len(dataset) if max_questions is None else min(max_questions, len(dataset))
    print(f"Dataset: {total_questions} questions")
    print(f"Models: {[m['name'] for m in models_config]}")
    print(f"Evaluation method: {evaluation_method}")

    # Spawn processes
    print(f"\nSpawning {world_size} processes...")
    mp.spawn(
        evaluate_subset,
        args=(world_size, dataset, models_config, output_dir, evaluation_method, parser_fn, max_questions),
        nprocs=world_size,
        join=True
    )

    # Merge results
    print("\nMerging results from all ranks...")
    merge_results(output_dir, world_size)

    print("\n✅ DDP evaluation complete!")
