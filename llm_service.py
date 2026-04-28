
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)



class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        ...


class TransformersLLM(BaseLLM):

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: Optional[str] = None,
    ):
        from transformers import pipeline as hf_pipeline
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Загрузка модели %s на устройство %s ...", model_name, device)
        self.pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        logger.info("Модель загружена.")

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return result[0]["generated_text"][-1]["content"]


class DeepseekApiLLM(BaseLLM):

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — помощник, который отвечает на вопросы, используя "
                        "только предоставленный контекст. Если информации "
                        "недостаточно, честно скажи об этом."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content

class StubLLM(BaseLLM):

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        marker = "Текст:"
        idx = prompt.find(marker)
        if idx != -1:
            context = prompt[idx + len(marker) :].strip()
            return f"[StubLLM] Найденный контекст:\n{context}"
        return f"[StubLLM] Промпт получен ({len(prompt)} символов)"
