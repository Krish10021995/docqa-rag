from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from docqa.config import Settings


class StubChatModel(BaseChatModel):
    """Returns a canned completion. Used in tests and for the no-key quickstart."""

    response: str = '{"answer": "This is a stub answer.", "citations": [1, 2]}'

    @property
    def _llm_type(self) -> str:
        return "docqa-stub"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        prompt = messages[-1].content
        if isinstance(prompt, list):
            prompt = " ".join(str(block) for block in prompt)
        text = self.response
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def create_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "stub":
        return StubChatModel()

    if provider == "openai_compat":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key or "sk-none",
            base_url=settings.openai_api_base or None,
            temperature=settings.llm_temperature,
            max_retries=2,
            timeout=60,
        )

    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")
