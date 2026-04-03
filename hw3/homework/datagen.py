import json
from pathlib import Path

from .cot import CoTModel
from .data import Dataset, is_answer_valid


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    model = CoTModel()
    trainset = Dataset("train")

    questions = [item[0] for item in trainset]
    correct_answers = [item[1] for item in trainset]
    prompts = [model.format_prompt(q) for q in questions]

    generations = model.batched_generate(
        prompts,
        num_return_sequences=oversample,
        temperature=temperature,
    )

    output_data = []
    success = 0

    for question, correct_answer, cand_list in zip(questions, correct_answers, generations):
        chosen = None
        for g in cand_list:
            parsed = model.parse_answer(g)
            if parsed == parsed and is_answer_valid(parsed, correct_answer):
                chosen = g.strip()
                break

        if chosen is not None:
            output_data.append([question, correct_answer, chosen])
            success += 1

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"saved {len(output_data)} examples to {output_json}")
    print(f"success rate: {success / len(trainset):.3f}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
