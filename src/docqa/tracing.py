from __future__ import annotations

from collections.abc import Sequence

from langchain_core.callbacks import BaseCallbackHandler

from docqa.config import Settings


def get_tracing_handlers(settings: Settings) -> Sequence[BaseCallbackHandler]:
    """Return a Langfuse tracing handler when credentials are configured.

    Requires the [tracing] extra (`uv sync --extra tracing`). Without
    credentials no handlers are attached, keeping the runtime side-effect free.
    """
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []

    try:
        from langfuse.callback import CallbackHandler

        return [
            CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        ]
    except ImportError:  # pragma: no cover - guarded for UX
        raise RuntimeError("Langfuse tracing needs the [tracing] extra") from None


def tracing_enabled(settings: Settings) -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)
