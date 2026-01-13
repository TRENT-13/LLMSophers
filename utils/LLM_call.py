import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load once at module level
load_dotenv()

def _load_llm_OPEN(model_name: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment. Very bad naughty naughty!")
    
    return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)


def _load_llm_deepseek(model_name: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in environment")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=temperature,
    )