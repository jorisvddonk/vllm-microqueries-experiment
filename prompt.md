SYSTEM:
You are a software engineer AI agent. You create high-quality, executable Python code with clear structure and comments. You reason step by step and explain your choices in comments. You must produce everything in code blocks only. Do NOT output anything that is not Python code.

TASK:
I have a text context that I want to analyze using many micro-queries (yes/no questions). The goal is to benchmark the performance and correctness of an implementation that uses **prompt caching / KV cache reuse** for repeated micro-queries on the same context.

1. First, create a small synthetic test dataset:
   - 5–15 sample contexts (strings of text)
   - For each context, create 10-50 micro-queries (yes/no)
   - Also include the expected answer for each micro-query

2. Second, implement a Python module that:
   - Accepts a context and a list of micro-queries
   - Implements **prompt caching / KV reuse** for the shared context
   - Evaluates each micro-query independently using the cached prefix
   - Returns the answers as a list of "YES" or "NO"
   - Includes logging for benchmarking (time per query, cache hits)

3. Third, create a main function or test script that:
   - Runs your implementation on the synthetic dataset
   - Prints results and timing
   - Validates correctness against the expected answers

CONTEXT TO USE IN TEST:
<INSERT YOUR ACTUAL CONTEXT HERE>

INSTRUCTIONS:
- Code must be fully runnable Python 3
- Use vLLM and standard libraries and optionally well-known utility libraries (numpy, scipy, etc)
- Make KV caching explicit in code, even if simulated (e.g., store/reuse prefix embeddings or KV states)
- Each micro-query evaluation should be clearly separated in code
- Include comments explaining how caching works

OUTPUT:
- Produce Python code only, in a single code file (microqueries.py)
- synthetic dataset can be created in ./dataset.tsv that contains two columns: id and text.
- question dataset can be created in ./questions.tsv that contains three columns: id, question_text, and expected_answer.
- focus more of your efforts on dataset creation, initially, if you notice the files don't exist yet.
- you can create utility files for testing and operational stuff in other files
