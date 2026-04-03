from pathlib import Path
import json

import torch
from transformers import Trainer, TrainingArguments
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    model_path = Path(__file__).parent / "rft_model"

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, str(model_path)).to(llm.device)
    llm.model.eval()
    return llm


def train_model(
    output_dir: str,
    data_path: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 64,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    max_length: int = 384,
    **kwargs,
):
    llm = BaseLLM()
    tokenizer = llm.tokenizer

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm.model = get_peft_model(llm.model, lora_config)

    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    if data_path is None:
        data_path = str(Path(__file__).parent.parent / "data" / "rft.json")

    with open(data_path, "r") as f:
        rft_data = json.load(f)

    def tokenize_example(question: str, reasoning: str):
        prompt = llm.format_prompt(question)
        full_text = f"{prompt} {reasoning}{tokenizer.eos_token}"

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        full = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        prompt_len = len(tokenizer(prompt)["input_ids"])

        labels = [-100] * prompt_len + full["input_ids"][prompt_len:]

        if len(labels) < len(full["input_ids"]):
            labels += [-100] * (len(full["input_ids"]) - len(labels))
        else:
            labels = labels[: len(full["input_ids"])]

        for i in range(len(labels)):
            if full["attention_mask"][i] == 0:
                labels[i] = -100

        full["labels"] = labels
        return full

    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            question, _, reasoning = self.data[idx]
            return tokenize_example(question, reasoning.strip())

    train_dataset = TokenizedDataset(rft_data)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    final_dir = Path(__file__).parent / "rft_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(final_dir)

    test_model(str(final_dir))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})