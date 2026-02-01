#!/usr/bin/env python3

import time
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

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

Answer the question with only YES or NO based on the context above.
Question: {question}
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
            "cache_misses": 0,
        }

        start_total = time.time()

        if use_cache and context in self.context_cache:
            metrics["context_cache_hit"] = True
            cached_prefix, cached_time = self.context_cache[context]
            metrics["context_processing_time"] = cached_time
            logger.info(f"Context cache HIT - reusing cached KV states")
        else:
            logger.info("Context cache MISS - processing context prefix for caching")
            start_context = time.time()

            # Process context as prefix to build KV cache
            # We create a prompt with context only and process it
            # The prefix must match exactly what will be used in queries for caching to work
            context_prompt = f"Context: {context}\n\nAnswer the question with only YES or NO based on the context above.\nQuestion:"
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
                logger.info(
                    f"Context prefix processed and cached in {context_time:.3f}s"
                )

        # Process each query using the cached context KV states
        # First query on each context processes the prefix, subsequent queries reuse it
        for i, question in enumerate(questions):
            start_query = time.time()

            # Create prompt with context (will hit prefix cache for queries 2-N)
            prompt = self._create_prompt(context, question)
            inputs = [TextPrompt(prompt=prompt)]

            # Generate response - this will reuse the cached context prefix KV states
            outputs = self.llm.generate(inputs, self.sampling_params)

            # Parse the answer
            answer = self._parse_answer(outputs[0].outputs[0].text)
            results.append(answer)

            query_time = time.time() - start_query
            metrics["query_times"].append(query_time)

            # Track cache hits/misses at question level
            # First question processes the full prompt, subsequent questions reuse cached prefix
            if use_cache:
                if metrics["context_cache_hit"]:
                    metrics["cache_hits"] += 1
                elif i == 0:
                    metrics["cache_misses"] += 1
                else:
                    metrics["cache_hits"] += 1
            else:
                metrics["cache_misses"] += 1

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


def save_results_to_tsv(
    results: List[Dict],
    output_path: str,
    run_id: str,
    model_name: str,
    gpu_memory_utilization: float,
    use_cache: bool,
    overall_accuracy: float,
    total_correct: int,
    total_questions: int,
):
    """
    Save benchmark results to TSV file.

    The file contains one row per context evaluated, with all timing and accuracy metrics.

    Args:
        results: List of per-context result dictionaries
        output_path: Path to output TSV file
        run_id: Unique identifier for this run (timestamp)
        model_name: Model name used
        gpu_memory_utilization: GPU memory utilization setting
        use_cache: Whether prefix caching was enabled
        overall_accuracy: Overall accuracy across all contexts
        total_correct: Total correct answers
        total_questions: Total number of questions
    """
    file_exists = Path(output_path).exists()

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id",
            "timestamp",
            "model_name",
            "gpu_memory_utilization",
            "use_cache",
            "context_id",
            "accuracy",
            "correct",
            "total",
            "context_processing_time",
            "total_time",
            "avg_query_time",
            "min_query_time",
            "max_query_time",
            "context_cache_hit",
            "cache_hits",
            "cache_misses",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().isoformat()

        for result in results:
            metrics = result["metrics"]
            query_times = metrics.get("query_times", [])
            avg_query_time = sum(query_times) / len(query_times) if query_times else 0.0
            min_query_time = min(query_times) if query_times else 0.0
            max_query_time = max(query_times) if query_times else 0.0

            row = {
                "run_id": run_id,
                "timestamp": timestamp,
                "model_name": model_name,
                "gpu_memory_utilization": gpu_memory_utilization,
                "use_cache": use_cache,
                "context_id": result["context_id"],
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "context_processing_time": metrics["context_processing_time"],
                "total_time": metrics["total_time"],
                "avg_query_time": avg_query_time,
                "min_query_time": min_query_time,
                "max_query_time": max_query_time,
                "context_cache_hit": metrics["context_cache_hit"],
                "cache_hits": metrics["cache_hits"],
                "cache_misses": metrics["cache_misses"],
            }
            writer.writerow(row)

    logger.info(f"Results saved to {output_path}")


def save_summary_to_tsv(
    summary_path: str,
    run_id: str,
    model_name: str,
    gpu_memory_utilization: float,
    use_cache: bool,
    overall_accuracy: float,
    total_correct: int,
    total_questions: int,
    total_contexts: int,
    total_time: float,
    total_cache_hits: int,
    total_cache_misses: int,
):
    """
    Save summary-level benchmark results to TSV file.

    The file contains one row per run with overall statistics.

    Args:
        summary_path: Path to summary TSV file
        run_id: Unique identifier for this run (timestamp)
        model_name: Model name used
        gpu_memory_utilization: GPU memory utilization setting
        use_cache: Whether prefix caching was enabled
        overall_accuracy: Overall accuracy across all contexts
        total_correct: Total correct answers
        total_questions: Total number of questions
        total_contexts: Number of contexts evaluated
        total_time: Total time for all evaluations
        total_cache_hits: Total cache hits across all contexts
        total_cache_misses: Total cache misses across all contexts
    """
    file_exists = Path(summary_path).exists()

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id",
            "timestamp",
            "model_name",
            "gpu_memory_utilization",
            "use_cache",
            "total_contexts",
            "total_questions",
            "total_correct",
            "overall_accuracy",
            "total_time",
            "total_cache_hits",
            "total_cache_misses",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().isoformat()

        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "gpu_memory_utilization": gpu_memory_utilization,
            "use_cache": use_cache,
            "total_contexts": total_contexts,
            "total_questions": total_questions,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "total_time": total_time,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
        }
        writer.writerow(row)

    logger.info(f"Summary saved to {summary_path}")


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
    parser.add_argument(
        "--output",
        type=str,
        default="./results.tsv",
        help="Path to output results TSV file (default: ./results.tsv)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="./summary.tsv",
        help="Path to output summary TSV file (default: ./summary.tsv)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving results to TSV files"
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

    # Generate unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")

    total_correct = 0
    total_questions = 0
    all_results = []
    start_benchmark = time.time()
    use_cache = not args.no_cache

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
        if use_cache:
            print(f"  Cache hits: {metrics['cache_hits']}")
            print(f"  Cache misses: {metrics['cache_misses']}")
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
    total_benchmark_time = time.time() - start_benchmark

    total_cache_hits = sum(r["metrics"]["cache_hits"] for r in all_results)
    total_cache_misses = sum(r["metrics"]["cache_misses"] for r in all_results)

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total contexts evaluated: {len(all_results)}")
    print(f"Total questions: {total_questions}")
    print(f"Total correct: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print(f"Total benchmark time: {total_benchmark_time:.3f}s")
    if not args.no_cache:
        print(f"Total cache hits: {total_cache_hits}")
        print(f"Total cache misses: {total_cache_misses}")

    print("\nPer-context results:")
    for result in all_results:
        metrics = result["metrics"]
        cache_info = ""
        if use_cache:
            cache_info = (
                f" (hits: {metrics['cache_hits']}, misses: {metrics['cache_misses']})"
            )
        print(
            f"  {result['context_id']}: {result['accuracy']:.2%} ({result['correct']}/{result['total']}){cache_info}"
        )

    print("\n" + "=" * 80)

    # Save results to TSV files
    if not args.no_save:
        save_results_to_tsv(
            results=all_results,
            output_path=args.output,
            run_id=run_id,
            model_name=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_cache=use_cache,
            overall_accuracy=overall_accuracy,
            total_correct=total_correct,
            total_questions=total_questions,
        )
        save_summary_to_tsv(
            summary_path=args.summary,
            run_id=run_id,
            model_name=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            use_cache=use_cache,
            overall_accuracy=overall_accuracy,
            total_correct=total_correct,
            total_questions=total_questions,
            total_contexts=len(all_results),
            total_time=total_benchmark_time,
            total_cache_hits=total_cache_hits,
            total_cache_misses=total_cache_misses,
        )
        print(f"\nResults saved:")
        print(f"  Detailed: {args.output}")
        print(f"  Summary: {args.summary}")

    return all_results


if __name__ == "__main__":
    main()
