import os

from dotenv import load_dotenv

load_dotenv()


class LLMConfigError(RuntimeError):
    pass


def create_llm(provider, model):

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise LLMConfigError(
                "OPENAI_API_KEY is not set. Add it to .env, or switch the "
                "LLM Provider in the sidebar."
            )

        return ChatOpenAI(
            model=model,
            temperature=0
        )

    if provider == "DeepSeek":
        from langchain_deepseek import ChatDeepSeek

        if not os.getenv("DEEPSEEK_API_KEY"):
            raise LLMConfigError(...)

        return ChatDeepSeek(
            model=model,
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )

    raise LLMConfigError(
        f"Unsupported LLM provider: {provider}. "
        "Supported: OpenAI, DeepSeek."
    )