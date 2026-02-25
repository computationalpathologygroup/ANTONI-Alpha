"""
Unified Evaluation Engine

Generic evaluator that works with any dataset and model through abstract interfaces.
Handles the evaluation loop, checkpointing, and result collection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from tqdm import tqdm

from validation.base_models import BaseModel
from validation.datasets.base_dataset import BaseDataset
from validation.organ import compute_organ_score

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Generic evaluator for validation and benchmarking.

    Works with any BaseDataset and BaseModel implementation.
    Handles evaluation loop, checkpointing, progress tracking, and error recovery.
    """

    def __init__(
        self,
        output_dir: Path,
        checkpoint_interval: int = 10,
        evaluation_method: str = "generation"
    ):
        """
        Args:
            output_dir: Directory to save results
            checkpoint_interval: Save checkpoint every N questions
            evaluation_method: "generation" (text parsing) or "perplexity" (log-likelihood ranking)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.evaluation_method = evaluation_method

        self.results = []  # List of result dictionaries
        self.checkpoint_file = self.output_dir / "checkpoint.json"

    def load_checkpoint(self) -> List[Dict]:
        """Load results from checkpoint file if it exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.results = json.load(f)
            logger.info(f"Loaded {len(self.results)} results from checkpoint")
        return self.results

    def save_checkpoint(self):
        """Save current results to checkpoint file."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def is_already_evaluated(self, model_name: str, question_id: Any) -> bool:
        """Check if a question has already been evaluated for a model."""
        for result in self.results:
            if result['model_name'] == model_name and result['question_id'] == question_id:
                return True
        return False

    def add_result(
        self,
        model_name: str,
        question_item: Dict[str, Any],
        model_response: str,
        extracted_answer: Optional[str],
        is_correct: bool,
        **extra_data
    ):
        """
        Add a single evaluation result.

        Args:
            model_name: Name of the model
            question_item: Question dictionary from dataset
            model_response: Raw model response text
            extracted_answer: Extracted/parsed answer
            is_correct: Whether the answer is correct
            **extra_data: Optional extra fields (e.g., log_likelihoods, evaluation_method)
        """
        result = {
            'model_name': model_name,
            'question_id': question_item['question_id'],
            'question': question_item['question'],
            'ground_truth': question_item['ground_truth'],
            'model_response': model_response,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct,
            **extra_data  # Include any extra fields
        }

        # Add context info and metadata
        if 'context_info' in question_item:
            result['context_info'] = question_item['context_info']
        if 'metadata' in question_item:
            result['metadata'] = question_item['metadata']
        if 'choices' in question_item:
            result['choices'] = question_item['choices']

        self.results.append(result)

        # Save checkpoint periodically
        if len(self.results) % self.checkpoint_interval == 0:
            self.save_checkpoint()
            logger.debug(f"Checkpoint saved at {len(self.results)} results")

    def evaluate_model(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        model_name: str,
        max_questions: Optional[int] = None,
        evaluation_method: Optional[str] = None,
        parser_fn: Optional[callable] = None
    ):
        """
        Evaluate a model on a dataset.

        Args:
            model: Model instance (must implement BaseModel interface)
            dataset: Dataset instance (must implement BaseDataset interface)
            model_name: Name of the model for tracking
            max_questions: Maximum number of questions to evaluate (for testing)
            evaluation_method: Override evaluation method (None = use default)
            parser_fn: Optional parser function for extraction (answer_text) -> extracted_answer
        """
        eval_method = evaluation_method or self.evaluation_method
        logger.info(f"Starting evaluation for {model_name} using {eval_method} method")

        model_type = model.model_cfg.get("type", "antoni_alpha")

        # Group questions by context to minimize loading
        questions_by_context = dataset.group_questions_by_context()

        # Filter out already evaluated questions
        filtered_groups = {}
        total_remaining = 0
        for context_key, questions in questions_by_context.items():
            remaining = [
                q for q in questions
                if not self.is_already_evaluated(model_name, q['question_id'])
            ]
            if remaining:
                filtered_groups[context_key] = remaining
                total_remaining += len(remaining)

        logger.info(f"Evaluating {total_remaining} questions across {len(filtered_groups)} contexts")

        # Apply max_questions limit if specified
        if max_questions:
            questions_processed = 0
            limited_groups = {}
            for context_key, questions in filtered_groups.items():
                if questions_processed >= max_questions:
                    break
                take = min(len(questions), max_questions - questions_processed)
                limited_groups[context_key] = questions[:take]
                questions_processed += take
            filtered_groups = limited_groups
            logger.info(f"Limited to {questions_processed} questions (max_questions={max_questions})")

        # Process context by context
        for context_key, questions in tqdm(
            filtered_groups.items(),
            desc=f"Evaluating {model_name}",
            total=len(filtered_groups)
        ):
            try:
                # Load context once for all questions
                context = dataset.load_context(questions[0], model_type)

                # Evaluate all questions for this context
                for question_item in questions:
                    try:
                        self._evaluate_single_question(
                            model=model,
                            question_item=question_item,
                            context=context,
                            model_name=model_name,
                            eval_method=eval_method,
                            parser_fn=parser_fn
                        )

                    except Exception as e:
                        logger.error(f"Error evaluating question {question_item['question_id']}: {e}")
                        # Record failed result
                        self.add_result(
                            model_name=model_name,
                            question_item=question_item,
                            model_response=f"ERROR: {str(e)}",
                            extracted_answer=None,
                            is_correct=False,
                            evaluation_method=eval_method,
                            error=str(e)
                        )

            except Exception as e:
                logger.error(f"Error loading context for {context_key}: {e}")
                # Skip all questions for this context
                for question_item in questions:
                    self.add_result(
                        model_name=model_name,
                        question_item=question_item,
                        model_response=f"ERROR: Failed to load context - {str(e)}",
                        extracted_answer=None,
                        is_correct=False,
                        evaluation_method=eval_method,
                        error=str(e)
                    )

        # Final checkpoint
        self.save_checkpoint()
        logger.info(f"Evaluation complete for {model_name}")

    def _evaluate_single_question(
        self,
        model: BaseModel,
        question_item: Dict[str, Any],
        context: Any,
        model_name: str,
        eval_method: str,
        parser_fn: Optional[callable] = None
    ):
        """
        Evaluate a single question.

        Args:
            model: Model instance
            question_item: Question dictionary from dataset
            context: Loaded context (features/image)
            model_name: Name of the model
            eval_method: "generation" or "perplexity"
            parser_fn: Optional parser function for answer extraction
        """
        question = question_item['question']
        ground_truth = question_item['ground_truth']
        choices = question_item.get('choices', None)

        # Add instruction suffix for generation-based evaluation
        question_type = question_item.get('metadata', {}).get('question_type')
        if eval_method == "generation":
            if question_type == "diagnosis":
                # For diagnosis questions, instruct to use brackets
                question = question + "\n\nPlease provide your final answer in double brackets, like [[diagnosis]]."
            elif question_type == "tumor":
                # For yes/no questions, instruct to start with yes/no
                question = question + "\n\nPlease start your answer with 'Yes' or 'No'."
            elif question_type == "organ":
                # For organ questions, instruct to be concise
                question = question + "\n\nPlease provide a concise answer (1-4 words)."

        if eval_method == "perplexity" and choices:
            # Perplexity-based evaluation (requires multiple-choice format)
            # Convert list of choices to dict format expected by model
            if isinstance(choices, list):
                choices_dict = {choice: choice for choice in choices}
            else:
                choices_dict = choices

            predicted_answer, log_likelihoods = model.evaluate_perplexity(
                context=context,
                question=question,
                choices=choices_dict
            )

            # For perplexity, response is the predicted choice
            response = f"[Perplexity-based prediction: {predicted_answer}] Scores: {log_likelihoods}"
            extracted = predicted_answer
            is_correct = (predicted_answer == ground_truth)

            # Store log-likelihoods in result
            extra_data = {
                'log_likelihoods': log_likelihoods,
                'evaluation_method': 'perplexity'
            }

        else:
            # Generation-based evaluation (text parsing)
            response = model.generate(
                context=context,
                question=question,
                history=None
            )

            # Extract answer using parser if provided
            if parser_fn:
                extracted = parser_fn(response, question_item)
            else:
                # No parser - use raw response as extracted answer
                extracted = response.strip()

            # Check if this is an organ question for hierarchical scoring
            question_type = question_item.get('metadata', {}).get('question_type')

            if question_type == 'organ' and extracted:
                # Use hierarchical organ scoring
                taxonomy_path = Path(__file__).parent / 'organ' / 'taxonomy.yaml'
                organ_score = compute_organ_score(extracted, ground_truth, str(taxonomy_path))
                is_correct = (organ_score == 1.0)  # Exact match for binary accuracy
                extra_data = {
                    'evaluation_method': 'generation',
                    'organ_score': organ_score
                }
            else:
                # Standard exact match for non-organ questions (case-insensitive, punctuation-insensitive)
                if extracted and ground_truth:
                    # Normalize both for comparison (remove punctuation, lowercase)
                    import re
                    def normalize(text):
                        return re.sub(r'[^\w\s]', '', text).strip().lower()
                    is_correct = (normalize(extracted) == normalize(ground_truth))
                else:
                    is_correct = False
                extra_data = {'evaluation_method': 'generation'}

        # Record result
        self.add_result(
            model_name=model_name,
            question_item=question_item,
            model_response=response,
            extracted_answer=extracted,
            is_correct=is_correct,
            **extra_data
        )

    def save_results(self, filename: str = "results.json"):
        """Save detailed results to file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved detailed results to {output_file}")

    def get_results_by_model(self) -> Dict[str, List[Dict]]:
        """Group results by model name."""
        results_by_model = defaultdict(list)
        for r in self.results:
            results_by_model[r['model_name']].append(r)
        return dict(results_by_model)

    def clear_checkpoint(self):
        """Remove checkpoint file (e.g., after successful completion)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint file removed")
