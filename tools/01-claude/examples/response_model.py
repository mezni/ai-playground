import sys
from pathlib import Path

from pydantic import BaseModel


class ProductReview(BaseModel):
    product_name: str
    sentiment: str
    rating: int
    key_pros: list[str]


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.llm.claude_client import ClaudeProvider

    user_review = (
        "I bought the UltraSound X200 headphones. The noise"
        " cancellation is amazing and battery life lasts forever, though"
        " they are a bit heavy. Solid 4 out of 5!"
    )

    llm = ClaudeProvider()
    response = llm.generate(user_review, response_model=ProductReview)
    print(response)


if __name__ == "__main__":
    main()
