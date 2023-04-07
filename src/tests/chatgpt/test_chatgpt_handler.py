"""Tests processing using ChatGPT."""

import pytest
from openai.error import AuthenticationError
from pytest_mock import MockerFixture

from src.chatgpt.chatgpt_handler import ChatGPTHandler
from tests.conftest import MockedCompletion


class TestChatGPTHandler:
    """Test class for chatgpt_handler.py."""

    @pytest.fixture
    def gpt_handler(self, mocker: MockerFixture) -> ChatGPTHandler:
        """
        Fixture for testing, return a mocked ChatGPTHandler instance.

        Args:
            mocker: MockerFixture

        Returns:
            ChatGPTHandler instance
        """
        # Mock the openai.ChatCompletion.create method
        mocker.patch("openai.ChatCompletion.create", side_effect=MockedCompletion())
        return ChatGPTHandler(openai_key="SECRET_API_KEY")

    def test_query_api(self, gpt_handler: ChatGPTHandler) -> None:
        """
        Test for querying the OpenAI API.

        Args:
            gpt_processor: Mocked ChatGPTHandler instance

        Returns:
            None
        """
        query = "Riddle me this ..."

        result = gpt_handler.query_api(query)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_api_empty(self, gpt_handler: ChatGPTHandler) -> None:
        """
        Test for querying the OpenAI API with empty query.

        Args:
            gpt_processor: Mocked ChatGPTHandler instance

        Returns:
            None
        """
        result = gpt_handler.query_api("")

        assert result is None

    def test_query_api_authentication_error(
        self,
        mocker: MockerFixture,
        gpt_handler: ChatGPTHandler,
    ) -> None:
        """
        Test for querying the OpenAI API, triggering an AuthenticationError.

        Returns:
            None
        """
        side_effect = AuthenticationError("mocked error")
        mocker.patch("openai.ChatCompletion.create", side_effect=side_effect)

        query = "Riddle me this ..."

        result = gpt_handler.query_api(query)

        assert result is None
