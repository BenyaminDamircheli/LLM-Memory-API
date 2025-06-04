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
    def create(cls, provider_name: str, config, vector_config: Optional[dict]):
        class_type = cls.provider_to_class.get(provider_name)
        if not class_type:
            raise ValueError(f"Invalid provider: {provider_name}")

        embedding_class = load_class(class_type)
        base_config = BaseEmbeddingConfig(**config)

        # Pass the expected keyword arguments to the embedding model instead of
        # the configuration object itself.
        return embedding_class(
            api_key=base_config.api_key,
            model=base_config.model,
        )

