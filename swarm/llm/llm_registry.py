from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif 'gpt' in model_name.lower() or 'deepseek' in model_name.lower(): # any version of GPTChat like "gpt-4-1106-preview"
            print('### get model from gpt api.')
            model = cls.registry.get('GPTChat', model_name)
        elif 'PRM' in model_name:
            model = cls.registry.get('PRM', model_name)
        else:
            print('### get model from ollama.')
            model = cls.registry.get('ollama', model_name)

        return model
