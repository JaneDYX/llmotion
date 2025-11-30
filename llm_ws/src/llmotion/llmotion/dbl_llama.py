# dbl_llama.py (CLEAN + PATCHED VERSION)
import os
import re
from llama_cpp import Llama

LOCAL_LLAMA_MODELS = {

    # Qwen 0.5B models
    "qwen": "/media/krg/Data/llm_ws/src/llmotion/llmotion/models/qwen/qwen2.5-0.5b-instruct-q2_k.gguf",
    "qwen-2": "/media/krg/Data/llm_ws/src/llmotion/models/qwen/qwen2.5-0.5b-instruct-fp16.gguf",
    "qwen-3": "/media/krg/Data/llm_ws/src/llmotion/models/qwen/qwen2.5-0.5b-instruct-q3_k_m.gguf",
    "qwen-4": "/media/krg/Data/llm_ws/src/llmotion/models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen-5": "/media/krg/Data/llm_ws/src/llmotion/models/qwen/qwen2.5-0.5b-instruct-q5_k_m.gguf",
    "qwen-6": "/media/krg/Data/llm_ws/src/llmotion/models/qwen/qwen2.5-0.5b-instruct-q6_k.gguf",
    "qwen-8": "/media/krg/Data/llm_ws/src/llmotion/models/qwen/qwen2.5-0.5b-instruct-q8_0.gguf",

    # Llama 3.2 1B
    "llama": "/media/krg/Data/llm_ws/src/llmotion/models/llama32-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "llama3.2-1b-q5": "/media/krg/Data/llm_ws/src/llmotion/models/llama32-1b/Llama-3.2-1B-Instruct-Q5_K_M.gguf",
    "llama3.2-1b-iq3": "/media/krg/Data/llm_ws/src/llmotion/models/llama32-1b/Llama-3.2-1B-Instruct-IQ3_M.gguf",
    "llama3.2-1b-q4": "/media/krg/Data/llm_ws/src/llmotion/models/llama32-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "llama3.2-1b-iq4": "/media/krg/Data/llm_ws/src/llmotion/models/llama32-1b/Llama-3.2-1B-Instruct-IQ4_XS.gguf",
}


class DBL_LLAMA:
    def __init__(self, llm_model_name="qwen", template_name="new"):

        model_path = LOCAL_LLAMA_MODELS[llm_model_name]

        # load GGUF LLM on GPU
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False
        )

        template_file = f"/media/krg/Data/llm_ws/src/llmotion/llmotion/templates/{template_name}.txt"
        with open(template_file, "r") as f:
            self.template_text = f.read()

        self.template_text = re.sub(r"[|`]+", "", self.template_text)

    def run(self, llm_input):

        prompt = (
            f"{self.template_text.strip()}\n\n"
            f"User: {llm_input.strip()}\n"
            f"\n"
            f"Output ONLY ONE WORD from this list:\n"
            f"forward, left, right, back, stop\n"
            f"\n"
            f"Answer:"
        )

        output = self.llm(
            prompt,
            max_tokens=16,
            temperature=0.1,
            stop=["\n"]
        )

        # -------------------------------
        # raw LLM output (full text)
        # -------------------------------
        try:
            raw = output["choices"][0]["text"]
        except:
            raw = ""

        raw_clean = raw.strip().lower()

        # -------------------------------
        # parse into action
        # -------------------------------
        valid = ["forward", "left", "right", "back", "stop"]

        parsed = None

        # exact match
        if raw_clean in valid:
            parsed = raw_clean

        # substring match
        if parsed is None:
            for w in valid:
                if w in raw_clean:
                    parsed = w
                    break

        # fallback
        if parsed is None:
            parsed = "stop"

        # we now return tuple
        return raw_clean, parsed
