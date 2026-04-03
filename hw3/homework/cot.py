from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You solve unit conversion problems. "
                    "Be concise. "
                    "Show a short calculation and finish with the final result inside "
                    "<answer></answer>."
                ),
            },
            {
                "role": "user",
                "content": "How many centimeters are there in 2 meters?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 meter = 100 centimeters. "
                    "So 2 × 100 = 200. "
                    "<answer>200</answer>"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
