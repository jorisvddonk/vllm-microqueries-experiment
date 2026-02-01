#!/usr/bin/env python3

import csv
from pathlib import Path


def validate_dataset(dataset_path: str = "./dataset.tsv") -> bool:
    """Validate dataset TSV file."""
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset file not found: {dataset_path}")
        return False

    with open(dataset_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames

        if headers != ["id", "text"]:
            print(f"ERROR: Invalid headers: {headers}")
            return False

        rows = list(reader)
        print(f"Dataset validation passed: {len(rows)} contexts found")

        for row in rows[:3]:
            print(f"  {row['id']}: {row['text'][:50]}...")

    return True


def validate_questions(
    questions_path: str = "./questions.tsv", dataset_path: str = "./dataset.tsv"
) -> bool:
    """Validate questions TSV file."""
    if not Path(questions_path).exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        return False

    with open(questions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames

        valid_headers_old = ["id", "question_text", "expected_answer"]
        valid_headers_new = ["id", "context_id", "question_text", "expected_answer"]

        if headers != valid_headers_old and headers != valid_headers_new:
            print(f"ERROR: Invalid headers: {headers}")
            return False

        rows = list(reader)
        print(f"Questions validation passed: {len(rows)} questions found")

        answer_counts = {}
        for row in rows:
            ans = row["expected_answer"]
            answer_counts[ans] = answer_counts.get(ans, 0) + 1

        print(f"  Answer distribution: {answer_counts}")

        for row in rows[:3]:
            print(
                f"  {row['id']}: {row['question_text'][:50]}... -> {row['expected_answer']}"
            )

    return True


def main():
    """Run validation for both dataset files."""
    print("Validating dataset files...")
    print("-" * 50)

    dataset_valid = validate_dataset()
    print()
    questions_valid = validate_questions()
    print()

    if dataset_valid and questions_valid:
        print("✓ All files are valid!")
        return 0
    else:
        print("✗ Validation failed!")
        return 1


if __name__ == "__main__":
    exit(main())
