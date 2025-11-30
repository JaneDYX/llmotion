# ================================================================
#   dbl_llm.py  —  Unified Local + Cloud LLM Wrapper
#   Supports: Qwen GGUF, Llama GGUF, Ollama, GPT-4, GPT-3.5
# ================================================================

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI, OpenAIChat

import subprocess
import requests
import json
import os
import re
import PIL.Image
import replicate
import google.generativeai as palm


# ================================================================
# If needed, set OpenAI API key (not used for Qwen/Llama)
# ================================================================
os.environ.setdefault("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxx")


# ================================================================
# OLLAMA DISCOVERY
# ================================================================
_OLLAMA_MODELS_CACHE = None

def get_ollama_models():
    """Return a cached list of Ollama models (if any)."""
    global _OLLAMA_MODELS_CACHE

    if _OLLAMA_MODELS_CACHE is not None:
        return _OLLAMA_MODELS_CACHE

    try:
        out = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        models = [
            line.split()[0]
            for line in out.stdout.splitlines()[1:]
            if line.strip()
        ]
        _OLLAMA_MODELS_CACHE = models
        return models
    except Exception:
        return []


# ================================================================
#                             DBL
# ================================================================
class DBL:

    # ------------------------------------------------------------
    # ALL LOCAL GGUF MODELS SUPPORTED (Qwen + Llama)
    # ------------------------------------------------------------
    LOCAL_LLAMA_MODELS = {

        # Legacy models
        "Llama3.1-8B": "/media/krg/Data/llama_checkpoints/checkpoints/Llama3.1-8B-Instruct/model.gguf",
        "Llama3.1-8B-gguf": "/media/krg/Data/llama_checkpoints/checkpoints/Llama3.1-8B-Instruct-gguf/model.gguf",
        "Llama3.2-1B": "/media/krg/Data/llama_checkpoints/checkpoints/Llama3.2-1B-Instruct-int4-spinquant-eo8/model.gguf",
        "Llama3.2-1B-Instruct": "/media/krg/Data/llama_checkpoints/checkpoints/Llama3.2-1B-Instruct/model.gguf",
        "Llama3.2-3B-int4-qlora": "/media/krg/Data/llama_checkpoints/checkpoints/Llama3.2-3B-Instruct-int4-qlora-eo8/model.gguf",
        "Llama3.2-3B-int4-spinquant": "/media/krg/Data/llama_checkpoints/checkpoints/Llama3.2-3B-Instruct-int4-spinquant-eo8/model.gguf",

        # New Llama 3.2 1B models
        "llama3.2-1b": "/media/krg/Data/Talk2Drive/models/llama32-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "llama3.2-1b-q5": "/media/krg/Data/Talk2Drive/models/llama32-1b/Llama-3.2-1B-Instruct-Q5_K_M.gguf",
        "llama3.2-1b-iq3": "/media/krg/Data/Talk2Drive/models/llama32-1b/Llama-3.2-1B-Instruct-IQ3_M.gguf",
        "llama3.2-1b-q4": "/media/krg/Data/Talk2Drive/models/llama32-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "llama3.2-1b-iq4": "/media/krg/Data/Talk2Drive/models/llama32-1b/Llama-3.2-1B-Instruct-IQ4_XS.gguf",

        # Llama 3.2 3B models
        "llama3.2-3b-iq3": "/media/krg/Data/Talk2Drive/models/llama32-3b/Llama-3.2-3B-Instruct-IQ3_M.gguf",
        "llama3.2-3b-q3": "/media/krg/Data/Talk2Drive/models/llama32-3b/Llama-3.2-3B-Instruct-Q3_K_L.gguf",
        "llama3.2-3b-q4": "/media/krg/Data/Talk2Drive/models/llama32-3b/Llama-3.2-3B-Instruct-Q4_0_4_4.gguf",
        "llama3.2-3b-q5": "/media/krg/Data/Talk2Drive/models/llama32-3b/Llama-3.2-3B-Instruct-Q5_K_S.gguf",
        "llama3.2-3b-q8": "/media/krg/Data/Talk2Drive/models/llama32-3b/Llama-3.2-3B-Instruct-Q8_0.gguf",

        # Test models
        "llama3-test": "/media/krg/Data/Talk2Drive/models/h2o-danube3-500m-base-Q5_K_M.gguf",
        "llama3-test2": "/media/krg/Data/Talk2Drive/models/h2o-danube3-500m-base-Q4_K_M.gguf",

        # Qwen models
        "qwen": "/media/krg/Data/Talk2Drive/models/qwen/qwen2.5-0.5b-instruct-q2_k.gguf",
        "qwen-2": "/media/krg/Data/Talk2Drive/models/qwen/qwen2.5-0.5b-instruct-fp16.gguf",
        "qwen-3": "/media/krg/Data/Talk2Drive/models/qwen/qwen2.5-0.5b-instruct-q3_k_m.gguf",
        "qwen-4": "/media/krg/Data/Talk2Drive/models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "qwen-5": "/media/krg/Data/Talk2Drive/models/qwen/qwen2.5-0.5b-instruct-q5_k_m.gguf",
        "qwen-6": "/media/krg/Data/Talk2Drive/models/qwen/qwen2.5-0.5b-instruct-q6_k.gguf",
        "qwen-8": "/media/krg/Data/Talk2Drive/models/qwen/qwen2.5-0.5b-instruct-q8_0.gguf",
    }


    # ================================================================
    def __init__(
        self,
        llm_model_name="gpt-4",
        template_name="basic_demo",
        memory_enable=False,
        memory_path="memory",
        ollama_mode="local",
    ):

        self.llm_modal_name = llm_model_name
        self.template_name = template_name
        self.memory_enable = memory_enable
        self.memory_path = memory_path
        self.ollama_mode = ollama_mode

        name_lower = llm_model_name.lower()

        # ---------------- Flags for Qwen / Llama GGUF -------------------
        self.is_qwen = name_lower.startswith(("qwen", "qwen2"))
        self.is_local_llama = llm_model_name in self.LOCAL_LLAMA_MODELS

        if self.is_local_llama:
            self.model_path = self.LOCAL_LLAMA_MODELS[llm_model_name]

        # ---------------- Ollama detection ------------------------------
        available_ollama = get_ollama_models()
        self.is_ollama = llm_model_name in available_ollama
        if self.is_ollama:
            self.llm = llm_model_name

        # ---------------- Cloud LLMs (GPT-4 etc.) -----------------------
        if not self.is_qwen and not self.is_local_llama:

            if llm_model_name == "gpt-4":
                self.llm = ChatOpenAI(model_name="gpt-4", max_tokens=512)

            elif llm_model_name == "gpt-3.5-turbo":
                self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", max_tokens=512)

            else:
                raise ValueError(f"Unsupported remote model: {llm_model_name}")

        # ---------------- Load template --------------------------------
        with open(f"templates/{template_name}.txt", "r") as f:
            self.template = f.read()

        if memory_enable:
            with open(memory_path, "r") as f:
                self.template += f.read()


    # ================================================================
    # RUN
    # ================================================================
    def run(self, query: str):

        # ------------------------------------------------------------
        # Qwen: forward into dbl_llama (unified llama.cpp runner)
        # ------------------------------------------------------------
        if self.is_qwen:
            from dbl_llm.dbl_llama import DBL_LLAMA
            runner = DBL_LLAMA(
                llm_model_name=self.llm_modal_name,
                template_name=self.template_name,
                memory_enable=self.memory_enable,
                memory_path=self.memory_path,
            )
            return runner.run(query)

        # ------------------------------------------------------------
        # Local llama.cpp GGUF
        # ------------------------------------------------------------
        if self.is_local_llama:
            prompt = f"{self.template}\n{query}"

            cmd = [
                "/media/krg/Data/llama_checkpoints/checkpoints/llama.cpp/build/bin/test-chat",
                "-i",
                "-m", self.model_path,
                "-p", prompt,
                "--n_predict", "256",
            ]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return out.stdout
            except subprocess.CalledProcessError as e:
                return f"[ERROR] llama.cpp failed:\n{e.stderr}"

        # ------------------------------------------------------------
        # Ollama
        # ------------------------------------------------------------
        if self.is_ollama:
            host = "127.0.0.1" if self.ollama_mode == "local" else "128.195.204.246"
            url = f"http://{host}:11434/api/generate"

            prompt = f"{self.template}\n{query}"
            resp = requests.post(url, json={"model": self.llm, "prompt": prompt}, stream=True)

            full = ""
            for line in resp.iter_lines():
                if line:
                    j = json.loads(line.decode("utf-8"))
                    if "response" in j:
                        full += j["response"]

            return full

        # ------------------------------------------------------------
        # Cloud GPT fallback
        # ------------------------------------------------------------
        final_prompt = f"{self.template}\n{query}"

        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(final_prompt)
        elif hasattr(self.llm, "call_as_llm"):
            return self.llm.call_as_llm(final_prompt)
        elif callable(self.llm):
            return self.llm(final_prompt)

        return str(self.llm)




