from .base_llm import BaseLLM



class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a careful unit conversion assistant. "
                    "Solve the problem briefly. "
                    "Show only a short reasoning if needed. "
                    "Always end with the final numeric answer inside <answer></answer>. "
                    "Do not omit the tags."
                ),
            },
            {
                "role": "user",
                "content": "How many grams are there in 6 kg?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 kg = 1000 grams. "
                    "So 6 × 1000 = 6000. "
                    "<answer>6000</answer>"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{question}\n"
                    "Respond with the final numeric answer inside <answer></answer>."
                ),
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
