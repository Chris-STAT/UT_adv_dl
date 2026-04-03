import json
from pathlib import Path
from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    model = CoTModel()
    dataset = Dataset("train")

    results = []

    for question, true_answer in tqdm(dataset, desc="Generating CoT answers"):
        prompt = model.format_prompt(question)

        try:
            generations = model.batched_generate(
                [prompt],
                num_return_sequences=oversample,
                temperature=temperature
            )[0]  # list[str] for a single prompt
        except Exception as e:
            print(f"Skipping question due to error: {e}")
            continue

        found = False
        for reasoning in generations:
            try:
                predicted = model.parse_answer(reasoning)
                if abs(predicted - float(true_answer)) / max(abs(float(true_answer)), 1e-8) < 1e-3:
                    results.append([question, float(true_answer), reasoning])
                    found = True
                    break
            except Exception:
                continue

        if not found:
            continue  # skip if no correct answer found in sampled generations

    print(f"Generated {len(results)} valid reasoning samples out of {len(dataset)}")

    # Save to output_json
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
