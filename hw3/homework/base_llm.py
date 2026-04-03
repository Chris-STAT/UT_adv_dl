from typing import overload
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        return f"{question} Answer with <answer>NUMBER</answer>."

    

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        Be robust to slightly malformed closing tags like </answer
        and to extra trailing text.
        """
        try:
           # First try the normal well-formed case
           m = re.search(r"<answer>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*</answer>", answer)
           if m:
              return float(m.group(1))

           # Fallback: tolerate missing '>' in closing tag
           m = re.search(r"<answer>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*</answer?", answer)
           if m:
              return float(m.group(1))

           # Fallback: if model started the answer tag but never closed it
           m = re.search(r"<answer>\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", answer)
           if m:
              return float(m.group(1))

           return float("nan")
        except (ValueError, TypeError):
           return float("nan")

    def generate(self, prompt: str) -> str:
        """
        Simple single-prompt wrapper around batched generation.
        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        ...

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        from tqdm import tqdm

        if len(prompts) == 0:
            return [] if num_return_sequences is None else []

        # Prevent OOM
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            chunks = []
            for idx in tqdm(
                range(0, len(prompts), micro_batch_size),
                desc=f"LLM Running on Micro Batches {micro_batch_size}",
            ):
                chunk = self.batched_generate(
                    prompts[idx : idx + micro_batch_size],
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                )
                chunks.extend(chunk)
            return chunks

        # Generation setup
        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        nrs = 1 if num_return_sequences is None else num_return_sequences
        do_sample = temperature > 0

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=nrs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Because we left-pad, all prompt lengths are aligned to the same padded width
        prompt_len = inputs["input_ids"].shape[1]
        generated_only = outputs[:, prompt_len:]

        decoded = self.tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True,
        )

        if num_return_sequences is None:
            return decoded

        # reshape flat list -> one list per prompt
        reshaped = [
            decoded[i * num_return_sequences : (i + 1) * num_return_sequences]
            for i in range(len(prompts))
        ]
        return reshaped

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
