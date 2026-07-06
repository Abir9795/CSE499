import ollama
import json

class LLMClient:
    def __init__(self, model="qwen2.5-coder:7b"):
        self.model = model

    def generate(self, prompt, system=None, temperature=0.2):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature}
        )
        return response["message"]["content"]

if __name__ == "__main__":
    client = LLMClient()
    result = client.generate("Write a Python function that checks if a number is prime.")
    print(result)