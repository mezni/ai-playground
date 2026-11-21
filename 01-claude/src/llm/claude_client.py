import anthropic
from src.config import settings


class ClaudeProvider:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.messages = []

    def add_user_message(self, text):
        user_message = {"role": "user", "content": text}
        self.messages.append(user_message)

    def add_assistant_message(self, text):
        assistant_message = {"role": "assistant", "content": text}
        self.messages.append(assistant_message)

    def generate(self, message: str) -> str:
        self.add_user_message(message)
        response = self._create_response()
        text = response.content[0].text
        self.add_assistant_message(text)
        return text

    def _create_response(self):
        try:
            return self.client.messages.create(
                model=settings.anthropic_model,
                system="You are a helpful assistant.",
                max_tokens=1024,
                messages=self.messages,
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError("Provider authentication failed") from exc

        except anthropic.APIConnectionError as exc:
            raise RuntimeError("Provider connection failed") from exc

        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Provider returned HTTP {exc.status_code}") from exc
