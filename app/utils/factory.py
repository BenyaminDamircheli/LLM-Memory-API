import importlib
from typing import Optional

from app.configs.embedding.base import BaseEmbeddingConfig
from app.embedding.base import BaseEmbeddingModel


def load_class(class_type:str):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class EmbeddingFactory:
    provider_to_class = {
        "openai": "app.embedding.openai_embedding.OpenAIEmbeddingModel",
    }

    @classmethod
    def create(cls, provider_name:str, config, vector_config: Optional[dict]):
        class_type = cls.provider_to_class[provider_name]
        if class_type:
            embedding_class = load_class(class_type)
            base_config = BaseEmbeddingConfig(**config)
            return embedding_class(base_config)
        else:
            raise ValueError(f"Invalid provider: {provider_name}")

