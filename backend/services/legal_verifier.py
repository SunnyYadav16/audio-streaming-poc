"""
Legal Context Verification Service — LLaMA 4 via Vertex AI MaaS

This module provides a singleton LegalVerifierService that:
  1. Calls LLaMA 4 (Scout or Maverick) on Vertex AI Model Garden via the
     OpenAI-compatible endpoint.
  2. Asks the model to verify that an NLLB machine translation preserves
     legal meaning when moving between languages (en / es / pt).
  3. Returns a VerificationResult containing:
       - verified_translation  : LLM-refined translation
       - accuracy_score        : float [0.0–1.0], 1.0 = no change needed
       - accuracy_note         : one-sentence explanation of the score
       - raw_translation       : original NLLB output (passthrough)
       - used_fallback         : True if the LLM call failed

Configuration (backend/.env):
  VERTEX_PROJECT       = your-gcp-project-id          (required)
  VERTEX_LOCATION      = us-central1                   (default)
  LEGAL_LLM_MODEL      = llama-4-scout-17b-16e-instruct-maas  (default)
  LEGAL_VERIFY_TIMEOUT = 5.0                           (seconds, default)

If VERTEX_PROJECT is not set, get_legal_verifier() returns None and the
legal-verification step is silently skipped throughout the pipeline.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

LANG_LABELS = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
}

# ---------------------------------------------------------------------------
#  Result type
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of a single legal-context verification call."""
    verified_translation: str   # LLM-refined translation
    accuracy_score: float       # 0.0–1.0  (1.0 = identical to raw)
    accuracy_note: str          # human-readable explanation of the score
    raw_translation: str        # original NLLB output
    used_fallback: bool         # True if the LLM call failed / timed out


# ---------------------------------------------------------------------------
#  LegalVerifierService (singleton)
# ---------------------------------------------------------------------------

class LegalVerifierService:
    """
    Verifies machine-translated legal speech using LLaMA 4 on Vertex AI.

    Vertex AI exposes an OpenAI-compatible REST endpoint, so we use the
    standard `openai` Python SDK pointed at the Vertex base URL.  Auth is
    handled via Google Application Default Credentials.
    """

    _instance: Optional["LegalVerifierService"] = None
    _client = None  # openai.OpenAI instance

    def __new__(cls) -> "LegalVerifierService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if LegalVerifierService._client is None:
            self._load_client()

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #

    def _load_client(self) -> None:
        """Initialise the OpenAI client pointed at Vertex AI."""
        try:
            import google.auth
            import google.auth.transport.requests
            import openai

            project = os.environ["VERTEX_PROJECT"]
            location = os.environ.get("VERTEX_LOCATION", "us-central1")

            # Fetch a short-lived access token from ADC
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)

            base_url = (
                f"https://{location}-aiplatform.googleapis.com/v1beta1/"
                f"projects/{project}/locations/{location}/endpoints/openapi"
            )

            LegalVerifierService._client = openai.OpenAI(
                base_url=base_url,
                api_key=credentials.token,
            )

            model = os.environ.get(
                "LEGAL_LLM_MODEL",
                "meta/llama-4-scout-17b-16e-instruct-maas",
            )
            LegalVerifierService._model = model
            print(
                f"[LegalVerifier] Client initialised "
                f"(project={project}, location={location}, model={model})"
            )

        except KeyError:
            raise RuntimeError(
                "[LegalVerifier] VERTEX_PROJECT env var is required. "
                "Set it in backend/.env"
            )
        except Exception as e:
            raise RuntimeError(
                f"[LegalVerifier] Failed to initialise Vertex AI client: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def verify(
        self,
        original_text: str,
        raw_translation: str,
        source_lang: str,
        target_lang: str,
    ) -> VerificationResult:
        """
        Verify that *raw_translation* preserves the legal meaning of
        *original_text* when rendered in *target_lang*.

        This is a synchronous, blocking call — callers should run it in
        ``asyncio.to_thread`` to avoid blocking the event loop.

        Args:
            original_text:   The utterance in *source_lang* as transcribed
                             by Faster-Whisper.
            raw_translation: The NLLB-200 machine translation in *target_lang*.
            source_lang:     Short ISO code: "en", "es", or "pt".
            target_lang:     Short ISO code: "en", "es", or "pt".

        Returns:
            VerificationResult — never raises; falls back gracefully on error.
        """
        timeout = float(os.environ.get("LEGAL_VERIFY_TIMEOUT", "5.0"))
        src_label = LANG_LABELS.get(source_lang, source_lang)
        tgt_label = LANG_LABELS.get(target_lang, target_lang)

        system_prompt = (
            "You are a certified legal interpreter assistant specialising in "
            "courtroom proceedings. Your task is to verify that a machine "
            "translation preserves the precise legal meaning of the original "
            "utterance. You must respond with ONLY a single valid JSON object "
            "— no markdown, no commentary, no extra text.\n\n"
            "JSON schema:\n"
            '{"verified_translation": "<string>", "accuracy_score": <float 0.0–1.0>, '
            '"accuracy_note": "<one sentence>"}\n\n'
            "Guidelines:\n"
            "- accuracy_score of 1.0 means the machine translation is legally "
            "precise and no changes are needed.\n"
            "- accuracy_score between 0.8–0.99 means minor rewording for legal "
            "clarity.\n"
            "- accuracy_score below 0.8 means substantial rephrasing was "
            "required to preserve legal meaning.\n"
            "- Never invent facts. Never add or remove legal claims.\n"
            "- If the original is informal speech, preserve its informal register "
            "in the verified translation."
        )

        user_message = (
            f"Original ({src_label}):\n{original_text}\n\n"
            f"Machine translation ({tgt_label}):\n{raw_translation}\n\n"
            "Please verify the translation and return the JSON result."
        )

        try:
            client = LegalVerifierService._client
            model = LegalVerifierService._model

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,   # low temperature for consistency
                max_tokens=512,
                timeout=timeout,
            )

            raw_json = response.choices[0].message.content or ""
            result = self._parse_response(raw_json, raw_translation)
            return result

        except Exception as e:
            print(f"[LegalVerifier] API call failed ({type(e).__name__}): {e}")
            return self._fallback(raw_translation)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _parse_response(
        self, raw_json: str, raw_translation: str
    ) -> VerificationResult:
        """Parse the LLM JSON response, returning a fallback on any error."""
        try:
            # Strip any accidental markdown code fences
            clean = re.sub(r"```(?:json)?|```", "", raw_json).strip()
            data = json.loads(clean)

            verified = str(data.get("verified_translation", raw_translation)).strip()
            score = float(data.get("accuracy_score", 1.0))
            # Clamp to [0.0, 1.0]
            score = max(0.0, min(1.0, score))
            note = str(data.get("accuracy_note", "")).strip()

            if not verified:
                verified = raw_translation
            if not note:
                note = "No additional notes from verifier."

            return VerificationResult(
                verified_translation=verified,
                accuracy_score=round(score, 3),
                accuracy_note=note,
                raw_translation=raw_translation,
                used_fallback=False,
            )

        except Exception as e:
            print(f"[LegalVerifier] JSON parse failed: {e} — raw: {raw_json!r}")
            return self._fallback(raw_translation)

    @staticmethod
    def _fallback(raw_translation: str) -> VerificationResult:
        """Return the raw NLLB translation unchanged when the LLM call fails."""
        return VerificationResult(
            verified_translation=raw_translation,
            accuracy_score=1.0,
            accuracy_note="Verification unavailable — showing machine translation.",
            raw_translation=raw_translation,
            used_fallback=True,
        )


# ---------------------------------------------------------------------------
#  Module-level accessor
# ---------------------------------------------------------------------------

_legal_verifier: Optional[LegalVerifierService] = None
_init_attempted: bool = False   # True once we've tried at least once


def get_legal_verifier() -> Optional[LegalVerifierService]:
    """
    Get (or create) the global LegalVerifierService instance.

    Returns None if VERTEX_PROJECT is not set in the environment, which
    means the feature is disabled and the caller should skip verification.

    Unlike a strict singleton, this retries initialisation on every server
    startup (hot-reload creates a fresh module import, resetting the globals).
    """
    global _legal_verifier, _init_attempted

    if os.environ.get("VERTEX_PROJECT") is None:
        return None

    # Already successfully initialised — return the cached instance
    if _legal_verifier is not None:
        return _legal_verifier

    # Only try once per process lifetime to avoid spamming logs
    if _init_attempted:
        return None

    _init_attempted = True
    try:
        _legal_verifier = LegalVerifierService()
    except Exception as e:
        print(f"[LegalVerifier] Disabled: {e}")
        return None

    return _legal_verifier
