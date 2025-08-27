import tiktoken

from src.config import openai_api_key
from openai import OpenAI
from typing_extensions import Self

class GPT4oAI:
    
    def __init__(self: Self) -> None:
        """
        Class constructor which initializes the OpenAI client and the GPT4o model.

        Parameters
        -----------
        self: the class Object (Self).
        model_name: the model path from HuggingFace (str).
        manual_seed: the model seed (int).

        Returns
        --------
        None.
        """

        self.api_key = openai_api_key
        self.model = 'gpt-4o-mini'
        self.client = OpenAI(api_key = self.api_key)
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')


    def count_tokens(self: Self, text: str):
        """
        Utility function to count tokens using the GPT4o tokenizer.

        Parameters
        -----------
        self: the class Object (Self).
        text: the text to count tokens from (str).

        Returns
        --------
        <object>: the number of tokens (int).
        """
        return len(self.tokenizer.encode(text))


    def infer(self: Self, user_prompt: str) -> str:            
        """
        Class method that provides a chat completion inference for OpenAI models.

        Parameters
        -----------
        self: the class Object (Self).
        user_prompt: the user input prompt (str).

        Returns
        --------
        <object>: the decoded output (str).
        """
        response = self.client.chat.completions.create(
            model = self.model,
            temperature = 1.0,
            max_tokens = 100,
            messages = [
                {
                    'role': 'user',
                    'content': user_prompt,
                }
            ]
        )
        return response.choices[0].message.content
