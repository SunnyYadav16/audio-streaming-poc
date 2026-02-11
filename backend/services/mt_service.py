"""
Machine Translation service using CTranslate2 + NLLB-200.

This module provides a singleton MTService that:
 1. Downloads facebook/nllb-200-distilled-1.3B from Hugging Face (first run).
 2. Converts it to an optimised CTranslate2 int8 model (one-time).
 3. Loads the CTranslate2 Translator for fast CPU/GPU inference.
 4. Exposes a simple `translate(text, source_lang, target_lang)` method.

Language codes follow the short ISO 639-1 codes used by the rest of the
pipeline ("en", "es", "pt").  Internally we map to the NLLB Flores-200
BCP-47 codes (e.g. "eng_Latn", "spa_Latn", "por_Latn").
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import ctranslate2
import transformers

# ---------------------------------------------------------------------------
#  Language code mapping: short (pipeline) <-> NLLB Flores-200
# ---------------------------------------------------------------------------

LANG_CODE_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
}

NLLB_TO_LANG_CODE: dict[str, str] = {v: k for k, v in LANG_CODE_TO_NLLB.items()}

SUPPORTED_LANGUAGES = set(LANG_CODE_TO_NLLB.keys())

# ---------------------------------------------------------------------------
#  Model paths
# ---------------------------------------------------------------------------

HF_MODEL_NAME = os.environ.get(
    "NLLB_MODEL", "facebook/nllb-200-distilled-1.3B"
)
CT2_MODEL_DIR = Path(__file__).parent.parent / "models" / "nllb-200-distilled-1.3B-ct2"


# ---------------------------------------------------------------------------
#  MTService (singleton)
# ---------------------------------------------------------------------------

class MTService:
    """
    Machine Translation service powered by CTranslate2 + NLLB-200.

    The CTranslate2 Translator and the HF tokenizer are loaded once and
    reused across all requests.  The model is automatically converted from
    Hugging Face format on the very first run (requires network access).
    """

    _instance: Optional["MTService"] = None
    _translator: Optional[ctranslate2.Translator] = None
    _tokenizer: Optional[transformers.AutoTokenizer] = None

    def __new__(cls) -> "MTService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if MTService._translator is None:
            self._load_model()

    # ------------------------------------------------------------------ #
    #  Model loading / conversion                                         #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        ct2_dir = str(CT2_MODEL_DIR)

        # One-time conversion from Hugging Face -> CTranslate2 (int8)
        if not CT2_MODEL_DIR.exists():
            print(f"[MT] Converting {HF_MODEL_NAME} to CTranslate2 format …")
            print("[MT] This downloads the model (~2.5 GB) and converts it. One-time operation.")
            CT2_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ct2-transformers-converter",
                    "--model", HF_MODEL_NAME,
                    "--output_dir", ct2_dir,
                    "--quantization", "int8",
                    "--force",
                ],
                check=True,
            )
            print(f"[MT] Conversion complete → {ct2_dir}")

        # Load CTranslate2 translator
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "int8"  # matches the quantization above

        print(f"[MT] Loading CTranslate2 translator ({device}, {compute_type}) …")
        MTService._translator = ctranslate2.Translator(
            ct2_dir,
            device=device,
            compute_type=compute_type,
        )

        # Load tokenizer (uses the original HF checkpoint's tokenizer files)
        print(f"[MT] Loading tokenizer from {HF_MODEL_NAME} …")
        MTService._tokenizer = transformers.AutoTokenizer.from_pretrained(
            HF_MODEL_NAME,
        )
        print("[MT] Translation model ready ✓")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def translator(self) -> ctranslate2.Translator:
        assert MTService._translator is not None, "MT translator not loaded"
        return MTService._translator

    @property
    def tokenizer(self) -> transformers.AutoTokenizer:
        assert MTService._tokenizer is not None, "MT tokenizer not loaded"
        return MTService._tokenizer

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Translate *text* from *source_lang* to *target_lang*.

        Args:
            text: The string to translate.
            source_lang: Short language code ("en", "es", "pt").
            target_lang: Short language code ("en", "es", "pt").

        Returns:
            The translated string.  If the languages are the same or
            unsupported, the original text is returned unchanged.
        """
        if not text or not text.strip():
            return ""
        if source_lang == target_lang:
            return text

        src_nllb = LANG_CODE_TO_NLLB.get(source_lang)
        tgt_nllb = LANG_CODE_TO_NLLB.get(target_lang)
        if not src_nllb or not tgt_nllb:
            print(
                f"[MT] Unsupported language pair: {source_lang} → {target_lang}"
            )
            return text

        tokenizer = self.tokenizer
        translator = self.translator

        # Set source language so special tokens are correct
        tokenizer.src_lang = src_nllb

        # Tokenize → list of string tokens (including BOS/EOS)
        source_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(text)
        )

        # Target prefix = just the target language token
        target_prefix = [tgt_nllb]

        results = translator.translate_batch(
            [source_tokens],
            target_prefix=[target_prefix],
            beam_size=4,
            max_decoding_length=256,
        )

        # First hypothesis, skip the leading language token
        target_tokens = results[0].hypotheses[0][1:]

        # Filter out special tokens (</s>, <unk>, etc.) before decoding
        special = {"</s>", "<s>", "<unk>", "<pad>"} | set(LANG_CODE_TO_NLLB.values())
        target_tokens = [t for t in target_tokens if t not in special]

        translated = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(target_tokens)
        )

        return translated.strip()


# ---------------------------------------------------------------------------
#  Module-level accessor (mirrors get_asr_service / get_vad_service)
# ---------------------------------------------------------------------------

_mt_service: Optional[MTService] = None


def get_mt_service() -> MTService:
    """Get or create the global MT service singleton."""
    global _mt_service
    if _mt_service is None:
        _mt_service = MTService()
    return _mt_service
