import json

try:
    import ollama
except Exception as exc:  # pragma: no cover - depends on environment
    ollama = None
    _OLLAMA_IMPORT_ERROR = exc
else:
    _OLLAMA_IMPORT_ERROR = None


DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_JUDGE_MODEL = "llama3.1:8b"


class LLMClient:
    def __init__(self, model=DEFAULT_MODEL):
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


def build_clients(model=None, judge_model=None, llm_judge=False):
    """Return (generator_client, judge_client) for one run.

    The judge is a separate client so it can run a model from a different
    family than the generator; a judge sharing the generator's family is a
    model grading its own output. The judge client is None when the judge
    is off, and naming a judge model is enough to turn it on.
    """
    generator = LLMClient(model or DEFAULT_MODEL)
    if not (llm_judge or judge_model):
        return generator, None
    return generator, LLMClient(judge_model or DEFAULT_JUDGE_MODEL)


class CallCountingClient:
    """Count model calls made through any client, real or a test double.

    The counter sits at the client boundary rather than inside LLMClient so
    that it also counts calls made by test doubles, and so that the retries
    inside the judge and the test generator are counted without either
    module knowing about it.

    Attribute access forwards to the wrapped client, which matters for more
    than convenience: the judge and the test generator both branch on
    whether `generate_json` exists. Forwarding through __getattr__ keeps a
    missing method missing instead of pushing a `generate`-only client down
    the JSON path.
    """

    def __init__(self, inner):
        # Bind first, so no attribute lookup reaches __getattr__ before
        # _inner exists and recurses forever.
        self._inner = inner
        self.call_count = 0

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if name in ("generate", "generate_json") and callable(attribute):
            return self._counted(attribute)
        return attribute

    def _counted(self, method):
        def call(*args, **kwargs):
            # Count before delegating, so a call that raises still counts as
            # compute spent.
            self.call_count += 1
            return method(*args, **kwargs)

        return call
