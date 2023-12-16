import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
DEFAULT_SYSTEM_PROMPT = """Ты — Сайга, русскоязычный автоматический 
                            ассистент для составления маркетинговых предложений для Газпромбанка. Ты придерживаешься 
                            формального стиля речи. 
                            Ты получаешь инструкцию и генерируешь предложение по полученным данным."""
CACHE_DIR_MODELS = 'models'

class Conversation:
    def __init__(
            self,
            message_template=DEFAULT_MESSAGE_TEMPLATE,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            response_template=DEFAULT_RESPONSE_TEMPLATE
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


def set_config(prev_config, temperature: float, top_p: float):
    new_config = prev_config
    new_config.temperature = temperature
    new_config.top_p = top_p

    return new_config


class Model:
    def __init__(self):
        self.config = PeftConfig.from_pretrained(MODEL_NAME,cache_dir = CACHE_DIR_MODELS)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir = CACHE_DIR_MODELS
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            MODEL_NAME,
            torch_dtype=torch.float16,
            cache_dir = CACHE_DIR_MODELS
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, cache_dir = CACHE_DIR_MODELS)
        self.generation_config = GenerationConfig.from_pretrained(MODEL_NAME, cache_dir = CACHE_DIR_MODELS)

    def generate(self, model, tokenizer, prompt, generation_config):
        data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(
            **data,
            generation_config=generation_config
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()

    def set_config(self, prev_config, temperature: float, top_p: float):
        new_config = prev_config
        new_config.temperature = temperature
        new_config.top_p = top_p

        return new_config
