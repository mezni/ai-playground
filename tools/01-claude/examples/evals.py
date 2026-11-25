import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class Evaluation(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    reasoning: str
    score: int


def run_prompt(llm: Any, test_case: dict[str, Any]) -> str:
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Answer strictly using this policy:
{test_case["context"]}

{test_case["query"]}
"""
    return llm.generate(prompt)


def grade_by_model(llm: Any, test_case: dict[str, Any], output: str) -> dict[str, Any]:
    task = test_case.get("task") or test_case["query"]
    eval_prompt = f"""
You are an expert code reviewer. Evaluate this AI-generated solution.

Task: {task}
Solution: {output}

Provide your evaluation as a structured JSON object with:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your assessment
- "score": A number between 1-10
"""
    llm.messages.clear()
    evaluation = llm.generate(eval_prompt, response_model=Evaluation)
    return evaluation.model_dump()


def run_test_case(llm: Any, test_case: dict[str, Any]) -> dict[str, Any]:
    """Calls run_prompt, then grades the result"""
    llm.messages.clear()
    output = run_prompt(llm, test_case)
    lowered = output.lower()
    total = len(test_case["must_include"]) or 1
    matched = sum(term.lower() in lowered for term in test_case["must_include"])
    score = round(10 * matched / total)

    evaluation = grade_by_model(llm, test_case, output)
    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "evaluation": evaluation,
    }


def run_eval(
    dataset: list[dict[str, Any]], llm: Any | None = None
) -> list[dict[str, Any]]:
    """Loads the dataset and calls run_test_case with each case"""
    if llm is None:
        from src.llm.claude_client import ClaudeProvider

        llm = ClaudeProvider()

    results = []
    for test_case in dataset:
        result = run_test_case(llm, test_case)
        results.append(result)

    return results


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    with Path(__file__).with_name("dataset.json").open() as f:
        dataset = json.load(f)

    results = run_eval(dataset)

    with Path(__file__).with_name("output.json").open("w") as f:
        json.dump(results, f, indent=2)

    for result in results:
        test_case = result["test_case"]
        status = "PASS" if result["score"] == 10 else "FAIL"
        print(f"{test_case['id']}: {status} ({result['score']}/10)")
        if status == "FAIL":
            print(f"  expected: {test_case['must_include']}")
            print(f"  got: {result['output']}")

    average_score = (
        sum(result["evaluation"]["score"] for result in results) / len(results)
        if results
        else 0
    )
    print(f"average_score: {average_score:.2f}")


if __name__ == "__main__":
    main()
