import json

try:
    import ollama
except Exception as exc:  # pragma: no cover - depends on environment
    ollama = None
    _OLLAMA_IMPORT_ERROR = exc
else:
    _OLLAMA_IMPORT_ERROR = None


class LLMClient:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def generate(self, prompt, system=None, temperature=0.2):
        return self._generate(
            prompt,
            system=system,
            temperature=temperature,
        )

    def generate_json(self, prompt, system=None, temperature=0.0, schema=None):
        """Generate JSON using Ollama's native JSON/JSON-schema mode."""
        return self._generate(
            prompt,
            system=system,
            temperature=temperature,
            response_format=schema or "json",
        )

    def _generate(
        self,
        prompt,
        system=None,
        temperature=0.2,
        response_format=None,
    ):
        if ollama is None:
            return self._fallback_response(prompt, system)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            request = dict(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
            )
            if response_format is not None:
                request["format"] = response_format
            response = ollama.chat(**request)
            return response["message"]["content"]
        except Exception:
            return self._fallback_response(prompt, system)

    def _fallback_response(self, prompt, system=None):
        """
        Fallback responses when LLM is not available.
        Raise an error instead of returning bogus data that corrupts testing.
        """
        if ollama is None:
            raise RuntimeError(
                f"ollama module is not available. Cannot generate response. "
                f"Original import error: {_OLLAMA_IMPORT_ERROR}"
            )
        raise RuntimeError(
            "LLM client failed to generate a response. Check your LLM service is running."
        )
