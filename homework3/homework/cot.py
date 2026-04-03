from .base_llm import BaseLLM

class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant for unit conversions. Convert units with clear, step-by-step reasoning. Use the **correct conversion factor** and do the **exact math**. Always show full **decimal precision** in your answer. Wrap only the final result in <answer> tags like this: <answer>42.123456</answer>,showing full decimal precision (e.g. 208.709827875)."
                )
            },
            # Example 1: Digital
            {
                "role": "user",
                "content": "How many MB is 2 GB?"
            },
            {
                "role": "assistant",
                "content": "1 GB = 1000 MB. 2 * 1000 = <answer>2000.0</answer>"
            },
            # Example 2: Weight
            {
                "role": "user",
                "content": "How many grams are in 3 kilograms?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 grams. 3 * 1000 = <answer>3000.0</answer>"
            },
            # Example 3: Time
            {
                "role": "user",
                "content": "How many weeks are there in 4 years?"
            },
            {
                "role": "assistant",
                "content": "1 year = 52.1775 weeks. 4 * 52.1775 = <answer>208.709827875</answer>"
            },
            # Actual question
            {
                "role": "user",
                "content": question
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )



def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    print("\n=== Debugging Sample Outputs ===")

    for i, ex in enumerate(testset):
        if i >= 5:
            break
        question, expected = ex
        prompt = model.format_prompt(question)
        output = model.generate(prompt)
        #print("Prompt:\n", prompt)
        #print("\n Question:",i)
        #print("Model Output:\n", output)  

        print(f"{i} Q: {question}")
        print(f"Model Output:\n{output}")
        print(f"Expected: {expected}")
        print(f"Parsed: {model.parse_answer(output)}")
        print("-" * 50)
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
