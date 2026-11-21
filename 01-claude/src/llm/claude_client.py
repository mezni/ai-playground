import anthropic

from src.config import settings


class ClaudeProvider:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate(self, message: str) -> str:
        response = self._create_response(message)
        return response.content[0].text

    def _create_response(self, message: str):
        try:
            return self.client.messages.create(
                model=settings.anthropic_model,
                system="You are a helpful assistant.",
                max_tokens=1024,
                messages=[{"role": "user", "content": message}],
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError("Provider authentication failed") from exc

        except anthropic.APIConnectionError as exc:
            raise RuntimeError("Provider connection failed") from exc

        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Provider returned HTTP {exc.status_code}") from exc
