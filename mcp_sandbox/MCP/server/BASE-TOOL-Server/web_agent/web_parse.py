import json
import os
import sys
import asyncio
import tiktoken
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir))
from get_html import fetch_web_content  
from utils.llm_caller import llm_call 

current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir, '..', '..', '..', '..', 'configs', 'web_agent.json'), 'r') as f:
    tools_config = json.load(f)

USE_PROMPT = tools_config['user_prompt']
USE_LLM = tools_config['USE_MODEL']

def split_chunks(text: str, model: str = ""):
    """
    Ultra-stable chunking:
    - No network
    - No HF dependency
    - Never crashes
    """

    try:
        import tiktoken

        # 永远使用同一个 tokenizer（稳定核心）
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        # 根据模型简单调整 chunk 大小（可选）
        model_lower = (model or "").lower()
        if "qwen" in model_lower:
            chunk_size = 6000
        elif "deepseek" in model_lower:
            chunk_size = 12000
        else:
            chunk_size = 8000

        tokens = enc.encode(text)

        return [
            enc.decode(tokens[i:i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]

    except Exception:
        # 终极 fallback（100% 不可能失败）
        chunk_size = 20000
        return [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

async def read_html(text, user_prompt, model=None):
    chunks = split_chunks(text, model)
    chunks = chunks[:1]  
    template = USE_PROMPT["search_conclusion"]
    final_prompt = template.format(user=user_prompt, info=chunks[0])
    answer = await llm_call(final_prompt, model)
    return _get_contents(answer)


async def parse_htmlpage(url: str, user_prompt: str = "", llm: str = None):
    try:
        is_fetch, text = await fetch_web_content(url)
        if not is_fetch:
            return {"content":"failed to fetch web content", "urls":[], "score":-1}

        model_to_use = llm if llm else USE_LLM
        try:
            print(f"use {model_to_use} to parse")
            result = await read_html(text, user_prompt, model=model_to_use)
            return result
        except Exception as e:
            fallback_model = tools_config['BASE_MODEL']
            print(f"origin llm call failed, use {fallback_model} to parse: {e}")
            result = await read_html(text, user_prompt, model=fallback_model)
            return result

    except Exception as e:
        print(f"parse failed: {str(e)}")
        return {"content":"failed to parse web content", "urls":[], "score":-1}


def _get_contents(response: str):
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        json_str = json_str.replace("\\(", "\\\\(").replace("\\)", "\\\\)")
        try:
            data = json.loads(json_str)
        except Exception as e:
            print(f"\033[91m response parse failed: {str(e)}\033[0m")
            think_end = response.rfind('</think>')
            if think_end != -1:
                return response[think_end + len('</think>'):].strip()
            else:
                return response
        return data
    except Exception as e:
        print(f"parse failed: {str(e)}")
        return {"content":"failed to parse web content", "urls":[], "score":-1}

async def main():
    query = "what is the content of the page"
    url = "https://proceedings.neurips.cc/paper_files/paper/2022"
    results = await parse_htmlpage(url, query, llm="DeepSeek/DeepSeek-V3-0324")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
