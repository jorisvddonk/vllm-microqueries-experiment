#!/usr/bin/env python3

import time
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MicroQueryEvaluator:
    """
    Evaluator for micro-queries with prompt caching/KV reuse.

    This class processes multiple micro-queries on the same context efficiently
    by reusing the cached KV cache for the context prefix. The context is processed
    once, and each query is evaluated using the cached context representation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.8,
    ):
        """
        Initialize the evaluator with a vLLM model.

        Args:
            model_name: Name or path of the local open-weight model (default: Qwen/Qwen2.5-3B-Instruct)
                       Recommended local models under 10B:
                       - Qwen/Qwen2.5-3B-Instruct (3B params, excellent quality)
                       - microsoft/Phi-3-mini-4k-instruct (3.8B params)
                       - Qwen/Qwen2.5-0.5B-Instruct (0.5B params, faster)
                       - meta-llama/Llama-3.2-3B-Instruct (3B params)
            max_model_len: Maximum sequence length for the model
            gpu_memory_utilization: Fraction of GPU memory to use (default: 0.8)
        """
        logger.info(f"Initializing model: {model_name}")
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            stop=["\n"],
        )
        self.context_cache: Dict[str, Tuple[List, float]] = {}

    def _create_prompt(self, context: str, question: str) -> str:
        """
        Create a prompt for answering a yes/no question based on context.

        Args:
            context: The text context to analyze
            question: The yes/no question to answer

        Returns:
            Formatted prompt string
        """
        return f"""Context: {context}

Question: {question}
Answer the question with only YES or NO based on the context above.

Answer:"""

    def _parse_answer(self, output: str) -> str:
        """
        Parse the model output to extract YES/NO answer.

        Args:
            output: Raw model output

        Returns:
            'YES', 'NO', or 'UNKNOWN' if cannot be determined
        """
        output_upper = output.strip().upper()
        if "YES" in output_upper:
            return "YES"
        elif "NO" in output_upper:
            return "NO"
        return "UNKNOWN"

    def evaluate_queries(
        self, context: str, questions: List[str], use_cache: bool = True
    ) -> Tuple[List[str], Dict]:
        """
        Evaluate multiple micro-queries on the same context.

        This method implements KV cache reuse by:
        1. First, processing the context as a prefix and caching its KV states
        2. Then, for each query, appending the question and generating the answer
        3. Reusing the cached context KV states for each query

        Args:
            context: The text context to analyze
            questions: List of yes/no questions
            use_cache: Whether to use prefix caching (default: True)

        Returns:
            Tuple of (answers list, benchmark metrics dict)
        """
        if not questions:
            return [], {}

        results = []
        metrics = {
            "context_cache_hit": False,
            "context_processing_time": 0.0,
            "query_times": [],
            "total_time": 0.0,
            "num_queries": len(questions),
            "cache_hits": 0,
        }

        start_total = time.time()

        if use_cache and context in self.context_cache:
            metrics["context_cache_hit"] = True
            metrics["cache_hits"] = len(questions)
            cached_prefix, cached_time = self.context_cache[context]
            metrics["context_processing_time"] = cached_time
            logger.info(f"Context cache HIT - reusing cached KV states")
        else:
            logger.info("Context cache MISS - processing context for caching")
            start_context = time.time()

            # Process context as prefix to build KV cache
            # We create a prompt with context only and process it
            context_prompt = f"Context: {context}\n\nQuestion:"
            prefix_input = [TextPrompt(prompt=context_prompt)]

            # This triggers KV cache generation for the context prefix
            # The enable_prefix_caching=True in LLM initialization enables this
            self.llm.generate(prefix_input, self.sampling_params)

            context_time = time.time() - start_context
            metrics["context_processing_time"] = context_time

            if use_cache:
                # Store reference to indicate context has been cached
                # (vLLM manages the actual KV cache internally)
                self.context_cache[context] = ([], context_time)
                logger.info(f"Context processed and cached in {context_time:.3f}s")

        # Process each query using the cached context KV states
        for i, question in enumerate(questions):
            start_query = time.time()

            # Create prompt with context (will hit prefix cache)
            prompt = self._create_prompt(context, question)
            inputs = [TextPrompt(prompt=prompt)]

            # Generate response - this will reuse the cached context prefix KV states
            outputs = self.llm.generate(inputs, self.sampling_params)

            # Parse the answer
            answer = self._parse_answer(outputs[0].outputs[0].text)
            results.append(answer)

            query_time = time.time() - start_query
            metrics["query_times"].append(query_time)

            logger.debug(
                f"Query {i + 1}/{len(questions)}: {question[:50]}... -> {answer} ({query_time:.3f}s)"
            )

        metrics["total_time"] = time.time() - start_total

        return results, metrics

    def clear_cache(self):
        """Clear the context cache."""
        self.context_cache.clear()
        logger.info("Context cache cleared")


def load_dataset(dataset_path: str) -> Dict[str, str]:
    """
    Load contexts from TSV dataset file.

    Args:
        dataset_path: Path to dataset.tsv file

    Returns:
        Dictionary mapping context ID to text
    """
    contexts = {}
    with open(dataset_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            contexts[row["id"]] = row["text"]
    logger.info(f"Loaded {len(contexts)} contexts from {dataset_path}")
    return contexts


def load_questions(questions_path: str) -> List[Dict]:
    """
    Load questions from TSV questions file.

    Args:
        questions_path: Path to questions.tsv file

    Returns:
        List of question dictionaries
    """
    questions = []
    with open(questions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            questions.append(
                {
                    "id": row["id"],
                    "question_text": row["question_text"],
                    "expected_answer": row["expected_answer"],
                    "context_id": row.get("context_id", row["id"][:6]),
                }
            )
    logger.info(f"Loaded {len(questions)} questions from {questions_path}")
    return questions


def group_questions_by_context(questions: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group questions by their context ID.

    Args:
        questions: List of question dictionaries

    Returns:
        Dictionary mapping context ID to list of questions
    """
    grouped = {}
    for q in questions:
        ctx_id = q["context_id"]
        if ctx_id not in grouped:
            grouped[ctx_id] = []
        grouped[ctx_id].append(q)
    return grouped


def calculate_accuracy(predictions: List[str], expected: List[str]) -> float:
    """
    Calculate accuracy of predictions.

    Args:
        predictions: List of predicted answers
        expected: List of expected answers

    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(expected):
        raise ValueError("Predictions and expected lists must have same length")

    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
    return correct / len(expected) if expected else 0.0


def main():
    """Main function to run the micro-query evaluation benchmark."""
    parser = argparse.ArgumentParser(
        description="Evaluate micro-queries with prompt caching using vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Local open-weight model to use (default: Qwen/Qwen2.5-3B-Instruct). "
        "Models under 10B params recommended: Qwen2.5-3B, Phi-3-mini, Llama-3.2-3B",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset.tsv",
        help="Path to dataset.tsv file (default: ./dataset.tsv)",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="./questions.tsv",
        help="Path to questions.tsv file (default: ./questions.tsv)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable prefix caching for comparison"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization fraction (default: 0.8)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load dataset
    if not Path(args.dataset).exists():
        logger.error(f"Dataset file not found: {args.dataset}")
        return

    if not Path(args.questions).exists():
        logger.error(f"Questions file not found: {args.questions}")
        return

    contexts = load_dataset(args.dataset)
    questions = load_questions(args.questions)

    # Group questions by context
    grouped_questions = group_questions_by_context(questions)

    # Initialize evaluator
    evaluator = MicroQueryEvaluator(
        model_name=args.model, gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Run evaluation
    print("\n" + "=" * 80)
    print("MICRO-QUERY EVALUATION BENCHMARK WITH PROMPT CACHING")
    print("=" * 80 + "\n")

    total_correct = 0
    total_questions = 0
    all_results = []

    for ctx_id in sorted(grouped_questions.keys()):
        if ctx_id not in contexts:
            logger.warning(f"Context {ctx_id} not found in dataset, skipping")
            continue

        context = contexts[ctx_id]
        ctx_questions = grouped_questions[ctx_id]

        print(f"\n{'=' * 80}")
        print(f"Context ID: {ctx_id}")
        print(f"Context length: {len(context)} characters")
        print(f"Number of questions: {len(ctx_questions)}")
        print(f"{'=' * 80}")

        # Extract question texts and expected answers
        question_texts = [q["question_text"] for q in ctx_questions]
        expected_answers = [q["expected_answer"] for q in ctx_questions]
        question_ids = [q["id"] for q in ctx_questions]

        # Evaluate queries
        use_cache = not args.no_cache
        answers, metrics = evaluator.evaluate_queries(
            context, question_texts, use_cache=use_cache
        )

        # Calculate accuracy
        accuracy = calculate_accuracy(answers, expected_answers)
        correct = sum(1 for a, e in zip(answers, expected_answers) if a == e)

        total_correct += correct
        total_questions += len(ctx_questions)

        # Print results
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{len(ctx_questions)} correct)")
        print(f"  Context cache hit: {metrics['context_cache_hit']}")
        print(f"  Context processing time: {metrics['context_processing_time']:.3f}s")
        print(f"  Total evaluation time: {metrics['total_time']:.3f}s")
        if metrics["query_times"]:
            avg_query_time = sum(metrics["query_times"]) / len(metrics["query_times"])
            print(f"  Average query time: {avg_query_time:.3f}s")
            print(f"  Min query time: {min(metrics['query_times']):.3f}s")
            print(f"  Max query time: {max(metrics['query_times']):.3f}s")

        # Print individual results
        if args.verbose:
            print(f"\nDetailed results:")
            for qid, qtext, pred, exp in zip(
                question_ids, question_texts, answers, expected_answers
            ):
                status = "✓" if pred == exp else "✗"
                print(f"  {status} [{qid}] {qtext[:60]}... -> {pred} (expected: {exp})")

        all_results.append(
            {
                "context_id": ctx_id,
                "accuracy": accuracy,
                "correct": correct,
                "total": len(ctx_questions),
                "metrics": metrics,
            }
        )

    # Print overall summary
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total contexts evaluated: {len(all_results)}")
    print(f"Total questions: {total_questions}")
    print(f"Total correct: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.2%}")

    print("\nPer-context results:")
    for result in all_results:
        print(
            f"  {result['context_id']}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})"
        )

    print("\n" + "=" * 80)

    return all_results


if __name__ == "__main__":
    main()
