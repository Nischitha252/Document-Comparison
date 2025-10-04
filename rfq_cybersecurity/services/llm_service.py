"""
Azure OpenAI service helpers.

This module encapsulates the creation of embedding and chat model
clients for Azure OpenAI.  Centralising these factory functions
ensures that credentials and configuration are loaded consistently
throughout the application and facilitates unit testing by allowing
mocking of the returned objects.
"""

from __future__ import annotations

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from ..config import load_config

# Import the package logger
from ..logging_config import get_logger

logger = get_logger(__name__)


def get_embeddings() -> AzureOpenAIEmbeddings:
    """Return a configured embeddings client.

    The embeddings deployment name and other credentials are read
    from environment variables loaded by :func:`load_config`.
    """
    # Retrieve configuration values from environment variables
    cfg = load_config()
    try:
        # Instantiate and return an embeddings client
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=cfg["openai_endpoint"],
            api_key=cfg["openai_key"],
            azure_deployment=cfg["embedding_model"],
            openai_api_version=cfg["openai_version"],
        )
        logger.debug("Created embeddings client with deployment '%s'", cfg["embedding_model"])
        return embeddings
    except Exception as exc:
        # Log and re-raise to inform callers
        logger.error("Failed to create embeddings client: %s", exc)
        raise


def get_chat_model() -> AzureChatOpenAI:
    """Return a configured Azure Chat model client.

    The chat model deployment name and credentials are read from
    environment variables loaded by :func:`load_config`.
    """
    # Retrieve configuration values from environment variables
    cfg = load_config()
    try:
        # Instantiate and return a chat model client
        llm = AzureChatOpenAI(
            azure_deployment=cfg["llm_model"],
            openai_api_key=cfg["openai_key"],
            openai_api_version=cfg["openai_version"],
            azure_endpoint=cfg["openai_endpoint"],
        )
        logger.debug("Created chat model client with deployment '%s'", cfg["llm_model"])
        return llm
    except Exception as exc:
        # Log and re-raise so the caller can handle the failure
        logger.error("Failed to create chat model client: %s", exc)
        raise