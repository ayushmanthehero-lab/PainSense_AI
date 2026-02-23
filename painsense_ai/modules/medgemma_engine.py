"""
modules/medgemma_engine.py
───────────────────────────
Loads the local MedGemma model (Gemma3ForConditionalGeneration) with
4-bit quantization and exposes a clean generate() interface.

Model architecture : Gemma3ForConditionalGeneration (multimodal)
Quantization       : 4-bit NF4 via bitsandbytes  (fits in GTX 1650 4 GB VRAM)
"""

from __future__ import annotations

import gc
import os
import sys
import traceback
import torch
from typing import Optional
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_DIR, MEDGEMMA_LOAD_IN_4BIT, MEDGEMMA_MAX_NEW_TOKENS,
    MEDGEMMA_TEMPERATURE, DEVICE,
)

# Lazy imports – only resolved when engine is first instantiated
_AutoProcessor = None
_AutoModelForImageTextToText = None
_BitsAndBytesConfig = None


def _lazy_imports():
    global _AutoProcessor, _AutoModelForImageTextToText, _BitsAndBytesConfig
    if _AutoProcessor is None:
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        _AutoProcessor               = AutoProcessor
        _AutoModelForImageTextToText = AutoModelForImageTextToText
        _BitsAndBytesConfig          = BitsAndBytesConfig
        # ── Compatibility patch ────────────────────────────────────────────
        # accelerate >= 1.x passes `_is_hf_initialized` kwarg to
        # bitsandbytes.Params4bit.__new__() which 0.49.x doesn't accept.
        # Patch Params4bit to absorb the unknown kwarg silently.
        try:
            from bitsandbytes.nn import Params4bit
            import inspect
            if '_is_hf_initialized' not in str(inspect.signature(Params4bit.__new__)):
                _orig_new = Params4bit.__new__
                def _patched_new(cls, *args, _is_hf_initialized=False, **kwargs):
                    return _orig_new(cls, *args, **kwargs)
                Params4bit.__new__ = _patched_new
        except Exception:
            pass


class MedGemmaEngine:
    """
    Singleton wrapper around the local MedGemma model.

    Parameters
    ----------
    model_dir  : Path to the HuggingFace model folder (default from config)
    load_in_4bit: Apply 4-bit NF4 quantization

    Usage
    -----
    engine = MedGemmaEngine()
    response = engine.generate(prompt="Your clinical question here")
    """

    _instance: Optional["MedGemmaEngine"] = None

    def __new__(cls, *args, **kwargs):
        """Enforce singleton – load model only once per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(
        self,
        model_dir:    str  = MODEL_DIR,
        load_in_4bit: bool = MEDGEMMA_LOAD_IN_4BIT,
    ):
        if self._loaded:
            return   # already initialised

        _lazy_imports()

        print(f"[MedGemmaEngine] Loading model from: {model_dir}")
        print(f"[MedGemmaEngine] 4-bit quant: {load_in_4bit}  |  device: {DEVICE}")

        # ── Processor (tokeniser + image processor) ─────────────────────
        self.processor = _AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        # ── Model: try 4-bit GPU → fall back to fp16 CPU ──────────────
        self.model = self._load_model(model_dir, load_in_4bit)
        self.model.eval()

        self._loaded = True
        print("[MedGemmaEngine] Model loaded successfully.")

    @staticmethod
    def _load_model(model_dir: str, load_in_4bit: bool):
        """
        Attempt to load with 4-bit bitsandbytes on CUDA.
        Falls back to fp16 on CPU if quantization fails (version mismatch, OOM, etc.).
        """
        # ── 4-bit NF4 on GPU (all layers forced to GPU 0) ────────────
        # device_map={"": 0} puts every layer on cuda:0, avoiding meta-tensor
        # issues that accelerate triggers when any layer is on CPU/disk.
        if load_in_4bit and DEVICE == "cuda":
            quant_config = _BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = _AutoModelForImageTextToText.from_pretrained(
                model_dir,
                quantization_config=quant_config,
                device_map={"":  0},   # force everything to cuda:0
                dtype=torch.float16,
                local_files_only=True,
                attn_implementation="eager",  # disable flash attention (crashes on GTX 1650)
            )
            print("[MedGemmaEngine] Loaded in 4-bit NF4 on cuda:0 (eager attn).")
            return model

        # ── CPU fallback (fp32) ────────────────────────────────────────
        model = _AutoModelForImageTextToText.from_pretrained(
            model_dir,
            device_map="cpu",
            dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        print("[MedGemmaEngine] Loaded in fp32 on CPU.")
        return model

    def generate(
        self,
        prompt:         str,
        image:          Optional[Image.Image] = None,
        max_new_tokens: int   = MEDGEMMA_MAX_NEW_TOKENS,
        temperature:    float = MEDGEMMA_TEMPERATURE,
        system_prompt:  Optional[str] = None,
    ) -> str:
        """
        Run inference.

        Parameters
        ----------
        prompt        : Text prompt (may include structured clinical data)
        image         : Optional PIL Image for multimodal input
        max_new_tokens: Maximum tokens to generate
        temperature   : Sampling temperature (ignored — greedy decoding used)
        system_prompt : Optional system role message to guide model behaviour

        Returns
        -------
        str : Generated response text
        """
        # ── Build conversation in chat format ──────────────────────────
        # Gemma 3 supports system role in the jinja template, but to avoid
        # any list-vs-string content format inconsistencies we prepend the
        # system instruction directly into the user turn instead.
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        messages: list
        if image is not None:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  full_prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}],
            }]

        # ── Tokenise ────────────────────────────────────────────────────
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # NOTE: Do NOT set pad_token_id or min_new_tokens.
        # Forcing min_new_tokens causes the model to emit pad/unk tokens when it
        # naturally wants to stop → skip_special_tokens=True then strips them all
        # → empty response.  Let the model stop naturally.
        eos_id = self.processor.tokenizer.eos_token_id

        # ── Generate (greedy — avoids CUDA assert on 4-bit quantized model) ──
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_id,
                )
        except Exception:
            print("[MedGemmaEngine] generate() exception:")
            traceback.print_exc()
            raise

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        raw_new   = output_ids[0][input_len:]

        # Decode twice: once clean, once raw for debugging
        response  = self.processor.decode(raw_new, skip_special_tokens=True)

        # If completely empty, try decoding without skipping special tokens
        # then strip known Gemma control strings manually
        if not response.strip():
            raw_decode = self.processor.decode(raw_new, skip_special_tokens=False)
            # Strip <end_of_turn>, <start_of_turn>, <eos>, <pad>, <bos>
            for tok in ["<end_of_turn>", "<start_of_turn>model", "<start_of_turn>",
                        "<eos>", "<pad>", "<bos>"]:
                raw_decode = raw_decode.replace(tok, "")
            response = raw_decode

        # Debug logging
        preview = response.strip()[:120].replace("\n", "\\n")
        print(f"[MedGemmaEngine] Generated {len(raw_new)} tokens → '{preview}…'")

        # Cleanup VRAM
        del inputs, output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()

    def unload(self):
        """Free GPU memory (call before shutdown if needed)."""
        if hasattr(self, "model"):
            del self.model
            del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        MedGemmaEngine._instance = None
        self._loaded = False
        print("[MedGemmaEngine] Model unloaded.")
