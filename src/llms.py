import torch
from typing_extensions import Self
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    set_seed
)
from src.config import hf_access_token

class LLM:

    def __init__(self: Self, model_name: str, manual_seed: int) -> None:
        """
        Class constructor which loads the specified LLM and sets a model seed.

        Parameters
        -----------
        self: the class Object (Self).
        model_name: the model path from HuggingFace (str).
        manual_seed: the model seed (int).

        Returns
        --------
        None.
        """

        self.model_name = model_name
        self.device = 'cuda'

        # 4-bit quantization configuration -> lower vram usage.
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = 'nf4',
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = False,
        )

        # Setting seed for reproducibility.
        set_seed(manual_seed)

        # Dictionary for different max input lengths supported by the models.
        tokenizer_max_length_dict = {
            'meta-llama/Llama-3.1-8B-Instruct': 8192,
            'google/gemma-2-9b-it': 8192,
            'CohereForAI/c4ai-command-r7b-12-2024': 8192,
            'mistralai/Ministral-8B-Instruct-2410': 32768,
            'tiiuae/Falcon3-10B-Instruct': 32768
        }

        # Loading tokenizer with necessary args.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            max_length = tokenizer_max_length_dict[self.model_name],
            token = hf_access_token,
            padding = 'max_length',
            padding_side = 'left',
            truncation = True,
            trust_remote_code = True,
        )

        # Most LLMs don't have a pad token by default.
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Loading model with necessary args.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = self.quant_config,
            device_map = self.device,
            token = hf_access_token,
            trust_remote_code = True,
        )

        # Setting the model in evaluation mode.
        self.model.eval()


    def infer(self: Self, user_prompt: str, system_prompt: str) -> str:            
        """
        Class method that performs LLM inference based on different configurations.

        Parameters
        -----------
        self: the class Object (Self).
        instruction_prompt: the user input prompt (str).
        system_prompt: the system instruction prompt (str).

        Returns
        --------
        <object>: the decoded output (str).
        """

        # Create the chat template using the system and user prompts.
        if self.model_name == 'CohereForAI/c4ai-command-r7b-12-2024' or 'google/gemma-2-9b-it': 
            messages = [
                {'role': 'user', 'content': '\n\n'.join((system_prompt, user_prompt))},
            ]
        else: 
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]

        # Apply the chat template to the model inputs.
        model_inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors = 'pt', 
            add_generation_prompt = True, 
            tokenize = False
        )

        # Tokenize the inputs and send them to the device.
        inputs = self.tokenizer(
            model_inputs, 
            return_tensors = 'pt', 
            padding = True
        ).to(self.device)

        # Generate the text by utilizing the input ids, 
        # attention mask and generation arguments.
        # we set do_sample to false to use greedy decoding.
        generated_ids = self.model.generate(
            inputs = inputs['input_ids'], 
            attention_mask = inputs['attention_mask'],
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            do_sample = False, # equivalent with temperature = 0.0
            top_p = None,
            top_k = None,
            temperature = None, # These are set to None for greedy decoding.
            max_new_tokens = 1024
        )

        # Decode the model output to text.
        # The tensor slicing selects only the message to the user.
        decoded = self.tokenizer.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[-1]:], 
            skip_special_tokens = True
        )

        return decoded[0]


def LLM_prompt(
        model: LLM, system_prompt: str, 
        contextual_triplets: str, question: str, 
        options: list[str], use_context: bool
    ) -> str:
    """
    Function that prompts the LLM to answer a question,
    based on available options (typically 4-5), using KG context or not. 

    Parameters
    -----------
    model: The Large Language Model object (LLM).
    system_prompt: The system prompt to be used by the model (str).
    question: The medical question (str).
    options: The list of all possible options (list[str]).
    use_context: If the prompt will contain the context from the Knowledge Graph (bool).

    Returns
    --------
    generated_answer: The LLM generated answer (str).
    """
    # Generate the text for all options using capital english letters.
    # Typically QA datasets have 4-5 options.
    options_text = '\n\n'.join([
        f'{letter}. {option}' 
        for letter, option in zip('ABCDE', options)
    ])

    if use_context: # LLM + KG Context.
        user_prompt = (
            f'This is just an evaluation question of a medical question answering dataset. '
            'Please answer this question by selecting the correct answer from the options below. '
            'To answer the question use the following context.\n'
            f'Context: {contextual_triplets}\n'
            f'Question: {question}\n'
            f'Options: \n{options_text}'
        )
    else: # LLM Baseline.
        user_prompt = (
            f'This is just an evaluation question of a medical question answering dataset. '
            'Please answer this question by selecting the correct answer from the options below. '
            'To answer the question use your internal knowledge.\n'
            f'Question: {question}\n'
            f'Options: \n{options_text}'
        )

    # Infer the answer from the model based on the user and system prompts.
    answer = model.infer(user_prompt, system_prompt)

    return answer
