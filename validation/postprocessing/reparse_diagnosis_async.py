#!/usr/bin/env python3
"""
ASYNC VERSION - Much faster LLM judge reparsing using concurrent API calls.

Uses asyncio to process multiple questions in parallel, significantly reducing runtime.
Expected speedup: 5-10x faster than sequential version.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dotenv import load_dotenv
from google import genai
from tqdm.asyncio import tqdm as atqdm

load_dotenv()

# Configuration
INPUT_FILE = Path(__file__).parent / "results_final.json"
OUTPUT_FILE = Path(__file__).parent / "results_llm_judged.json"
CHECKPOINT_FILE = Path(__file__).parent / ".checkpoint_async.json"
ERROR_LOG_FILE = Path(__file__).parent / "reparse_errors_async.log"
CHECKPOINT_INTERVAL = 50  # Save every 50 questions (less frequent since faster)
MAX_RETRIES = 10
MODEL_NAME = "gemini-2.5-flash"
CONCURRENCY_LIMIT = 50  # Number of concurrent API calls


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return re.sub(r'[^\w\s]', '', text).strip().lower()


def initialize_gemini():
    """Initialize Gemini client."""
    try:
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
        return client
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)


def log_error(model_name: str, question_id: str, attempt: int, error: str):
    """Log errors to file (thread-safe with file locking) and console."""
    msg = f"{model_name} | {question_id} | Attempt {attempt} | {error}"

    # Print to console (using tqdm.write to play nice with progress bars)
    atqdm.write(f"⚠️ {msg}")

    with open(ERROR_LOG_FILE, 'a') as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")


async def parse_model_response_async(
    client,
    semaphore: asyncio.Semaphore,
    model_name: str,
    question_id: str,
    model_response: str,
    choices: List[str]
) -> dict:
    """
    Async version of LLM judge parser.
    Uses semaphore to limit concurrent requests.
    """
    choices_formatted = "\n".join([f"- {choice}" for choice in choices])

    prompt = f"""You are analyzing a pathology AI model's response to a multiple-choice diagnosis question.

The model was given these diagnosis options:
{choices_formatted}

The model generated this response:
{model_response}

Your task: Identify which diagnosis option the model selected as its final answer.

Rules:
1. Look for explicit conclusion statements (e.g., "most likely diagnosis", "final answer", "therefore", etc.)
2. The model often discusses all options but concludes with one specific answer
3. If the model clearly selected one of the provided options, return that option exactly as written
4. If the model did not select any of the provided options, return "other"
5. If the model refused to answer or was unclear, return "unknown"

Respond in strict JSON format:
{{
  "selected_option": "exact text of the chosen option, or 'unknown', or 'other'",
  "confidence": "high" or "medium" or "low",
  "reasoning": "1-2 sentence explanation of how you determined this"
}}
"""

    for attempt in range(MAX_RETRIES):
        async with semaphore:  # Limit concurrent requests
            try:
                # Use asyncio to run the synchronous API call in executor
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=MODEL_NAME,
                            contents=prompt,
                            config={"response_mime_type": "application/json"},
                        )
                    ),
                    timeout=60.0
                )

                # Parse JSON response
                try:
                    result = json.loads(response.text)
                except json.JSONDecodeError:
                    clean_text = response.text.replace("```json", "").replace("```", "").strip()
                    result = json.loads(clean_text)

                if 'selected_option' in result:
                    return result
                else:
                    raise ValueError(f"Invalid response structure: {result}")

            except Exception as e:
                error_msg = str(e)
                log_error(model_name, question_id, attempt + 1, error_msg)

                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff
                    wait_time = min(2 ** attempt, 60)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    log_error(model_name, question_id, "FINAL", f"Failed after {MAX_RETRIES} attempts")
                    return {
                        "selected_option": "error",
                        "confidence": "low",
                        "reasoning": f"Error after {MAX_RETRIES} attempts: {error_msg[:100]}"
                    }


def load_checkpoint() -> Set[Tuple[str, str]]:
    """Load checkpoint with processed composite keys."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(tuple(item) for item in data.get('processed', []))
    return set()


def save_checkpoint(processed: Set[Tuple[str, str]]):
    """Save checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'processed': [list(item) for item in processed]}, f)


async def process_question(
    client,
    semaphore: asyncio.Semaphore,
    entry: dict,
    updates: dict,
    processed: Set[Tuple[str, str]]
) -> Tuple[str, str]:
    """
    Process a single question asynchronously.
    Returns the composite key for tracking.
    """
    model_name = entry['model_name']
    question_id = entry['question_id']
    composite_key = (model_name, question_id)

    # Call LLM judge
    llm_result = await parse_model_response_async(
        client,
        semaphore,
        model_name,
        question_id,
        entry['model_response'],
        entry.get('choices', [])
    )

    # Extract answer
    extracted_answer = llm_result['selected_option']

    # Calculate if correct
    is_correct = (
        normalize(extracted_answer) == normalize(entry['ground_truth'])
        if extracted_answer not in ['unknown', 'other', 'error']
        else False
    )

    # Store update
    updates[composite_key] = {
        'extracted_answer': extracted_answer,
        'is_correct': is_correct
    }

    # Mark as processed
    processed.add(composite_key)

    return composite_key


async def main():
    print("=" * 80)
    print("ASYNC LLM JUDGE REPARSING - High Speed Version")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Concurrency: {CONCURRENCY_LIMIT} parallel requests")
    print(f"Max retries: {MAX_RETRIES}")
    print("=" * 80)

    # Initialize client
    client = initialize_gemini()
    print("✓ Gemini client initialized")

    # Load results
    print(f"\nLoading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        all_results = json.load(f)
    print(f"✓ Loaded {len(all_results)} entries")

    # Filter diagnosis questions
    diagnosis_questions = [
        e for e in all_results
        if e.get('metadata', {}).get('question_type') == 'diagnosis'
    ]
    print(f"✓ Found {len(diagnosis_questions)} diagnosis questions")

    # Show distribution
    from collections import Counter
    by_model = Counter(e['model_name'] for e in diagnosis_questions)
    print(f"\nDiagnosis questions by model:")
    for model in sorted(by_model.keys()):
        print(f"  {model}: {by_model[model]}")

    # Load checkpoint
    processed = load_checkpoint()
    if processed:
        print(f"\n✓ Resuming: {len(processed)} already processed")

    # Filter remaining
    remaining = [
        q for q in diagnosis_questions
        if (q['model_name'], q['question_id']) not in processed
    ]

    print(f"\nProcessing {len(remaining)} remaining questions...")
    print(f"⚡ ASYNC MODE with {CONCURRENCY_LIMIT} concurrent requests")
    print(f"🚀 Expected speedup: 5-10x faster than sequential")

    if remaining:
        print("\nPress Enter to start...")
        input()

    # Shared data structures
    updates = {}
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # Create tasks for all remaining questions
    tasks = [
        process_question(client, semaphore, entry, updates, processed)
        for entry in remaining
    ]

    # Process with progress bar
    start_time = time.time()
    results = []

    # Process in batches for checkpointing
    batch_size = CHECKPOINT_INTERVAL
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await atqdm.gather(*batch, desc=f"Batch {i//batch_size + 1}")
        results.extend(batch_results)

        # Save checkpoint after each batch
        save_checkpoint(processed)

        # Save intermediate results
        temp_results = []
        for orig_entry in all_results:
            new_entry = dict(orig_entry)
            key = (orig_entry['model_name'], orig_entry['question_id'])
            if key in updates:
                new_entry['extracted_answer'] = updates[key]['extracted_answer']
                new_entry['is_correct'] = updates[key]['is_correct']
            temp_results.append(new_entry)

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(temp_results, f, indent=2)

        print(f"✓ Checkpoint: {len(processed)}/{len(diagnosis_questions)} processed")

    elapsed = time.time() - start_time
    print(f"\n✓ Completed in {elapsed/60:.1f} minutes ({elapsed/len(remaining):.2f}s per question)")

    # Final save
    print("\nSaving final results...")
    final_results = []
    for entry in all_results:
        new_entry = dict(entry)
        key = (entry['model_name'], entry['question_id'])
        if key in updates:
            new_entry['extracted_answer'] = updates[key]['extracted_answer']
            new_entry['is_correct'] = updates[key]['is_correct']
        final_results.append(new_entry)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"✓ Saved to {OUTPUT_FILE}")

    # Cleanup
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    # Verification
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    final_diagnosis = [e for e in final_results if e.get('metadata', {}).get('question_type') == 'diagnosis']
    final_by_model = Counter(e['model_name'] for e in final_diagnosis)

    print(f"\nDiagnosis questions by model:")
    for model in sorted(final_by_model.keys()):
        print(f"  {model}: {final_by_model[model]}")

    # Accuracy
    print("\n" + "=" * 80)
    print("FINAL ACCURACY")
    print("=" * 80)

    total_correct = sum(1 for e in final_diagnosis if e['is_correct'])
    print(f"\nOverall: {total_correct}/{len(final_diagnosis)} ({100*total_correct/len(final_diagnosis):.2f}%)")

    print(f"\nPer-model:")
    for model in sorted(final_by_model.keys()):
        model_entries = [e for e in final_diagnosis if e['model_name'] == model]
        model_correct = sum(1 for e in model_entries if e['is_correct'])
        print(f"  {model}: {model_correct}/{len(model_entries)} ({100*model_correct/len(model_entries):.2f}%)")

    print("\n" + "=" * 80)
    print("✓ ASYNC REPARSING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
