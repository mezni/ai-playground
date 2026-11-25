import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.llm.claude_client import ClaudeProvider

    llm = ClaudeProvider()
    response = llm.generate("Say hello in one word")
    print(response)


if __name__ == "__main__":
    main()
