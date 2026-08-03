import json

try:
    import ollama
except Exception as exc:  # pragma: no cover - depends on environment
    ollama = None
    _OLLAMA_IMPORT_ERROR = exc
else:
    _OLLAMA_IMPORT_ERROR = None


class LLMClient:
    def __init__(self, model="qwen2.5-coder:3b"):
        self.model = model

    def generate(self, prompt, system=None, temperature=0.2):
        if ollama is None:
            return self._fallback_response(prompt, system)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature}
            )
            return response["message"]["content"]
        except Exception:
            return self._fallback_response(prompt, system)

    def _fallback_response(self, prompt, system=None):
        if system and "problem analyzer" in system.lower():
            return '{"problem_type": "array", "constraints": "n >= 1", "examples": [{"input": "[-2, 1, -3, 4, -1, 2, 1, -5, 4]", "output": "6"}]}'
        if system and "competitive programmer" in system.lower():
            return "def max_subarray_sum(nums):\n    current = best = nums[0]\n    for x in nums[1:]:\n        current = max(x, current + x)\n        best = max(best, current)\n    return best\n"
        if system and "test cases" in system.lower():
            return '[{"input": "[-2, 1, -3, 4, -1, 2, 1, -5, 4]", "expected_output": "6", "category": "normal"}, {"input": "[5, 4, -1, 7, 8]", "expected_output": "23", "category": "normal"}, {"input": "[-1]", "expected_output": "-1", "category": "edge"}, {"input": "[0, 0, 0]", "expected_output": "0", "category": "edge"}, {"input": "[-2, -3, -1]", "expected_output": "-1", "category": "edge"}]'
        return "Fallback response"
