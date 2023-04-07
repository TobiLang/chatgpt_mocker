"""Process data using ChatGPT."""
import logging
from typing import Optional

import openai
from openai.error import APIConnectionError, AuthenticationError

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class ChatGPTHandler:
    """Process data using ChatGPT."""

    MODEL = "gpt-3.5-turbo"

    def __init__(self, openai_key: str = "") -> None:
        """
        Initialize the class.
        """
        openai.api_key = openai_key

    def query_api(self, query: str) -> Optional[str]:
        """
        Query the ChatGPT API.

        Args:
            query: Query to send to the API

        Returns:
            Response message from the API
        """
        logging.info("Querying API...")

        # No need to query the API if there is no query content
        if not query:
            return None

        message = [{"role": "user", "content": query}]

        result = None
        try:
            completion = openai.ChatCompletion.create(
                model=self.MODEL, messages=message
            )  # type: ignore[no-untyped-call]
            result = completion.choices[0].message.content
        except AuthenticationError as ex:
            logger.error("Authentication error: %s", ex)
        except APIConnectionError as ex:
            logger.error("APIConnection error: %s", ex)

        return result
