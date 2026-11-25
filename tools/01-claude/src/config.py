import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        self.anthropic_model = os.environ.get(
            "ANTROPIC_MODEL", "claude-haiku-4-5-20251001"
        )


settings = Settings()
