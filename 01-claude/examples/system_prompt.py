import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.llm.claude_client import ClaudeProvider

    system_prompt = """
You are an ai architect.
"""

    llm = ClaudeProvider()
    response = llm.generate("Give 5 topic", system=system_prompt)
    print(response)


if __name__ == "__main__":
    main()
