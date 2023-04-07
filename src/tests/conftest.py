"""Configuration for ChatGPT tests."""
from typing import Any, Dict, Optional


# pylint: disable=too-few-public-methods
class NestedDictToObject:
    """
    Converts a nested dictionary into an object.
    """

    def __init__(self, dictionary: Dict[str, Any]) -> None:
        """
        Initializes an instance of NestedDictToObject.

        Args:
            dictionary: A nested dictionary to convert into an object.
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, NestedDictToObject(value))
            else:
                setattr(self, key, value)


# pylint: disable=too-few-public-methods
class MockedCompletion:
    """
    Mock OpenAI API completion.
    """

    def __init__(self, message: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the mocked API parameters.

        Args:
            message: Dictionary containing the message to be mocked.
                     Will be converted to an object.
        """
        if message is None:
            message = {"message": {"content": "This is a mocked message."}}

        # Convert to object
        message_obj = NestedDictToObject(message)

        self.choices = [message_obj]

    def __next__(self) -> "MockedCompletion":
        """
        Return the class instance itself.

        Unittest mock expects an iterable object, so we need to implement
        this method. Otherwise, we would get an error like this:
            TypeError: 'MockedCompletion' object is not iterable

        Returns:
            The class instance itself.
        """
        return self
