from .base_llm import BaseLLM
from .sft import test_model
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType


def load() -> BaseLLM:
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, str(model_path)).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    llm = BaseLLM()
    tokenizer = llm.tokenizer

    # Keep your original LoRA settings
    r = 16
    lora_alpha = 64
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llm.model = get_peft_model(llm.model, lora_config)
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()

    # Keep your original dataset loading style, but fix the likely path
    import json
    train_path = Path(__file__).parent.parent / "data" / "rft.json"
    with open(train_path, "r") as f:
        rft_data = json.load(f)

    # Each entry: [question, correct_answer, reasoning]
    def format_example(entry):
        question, _, reasoning = entry
        prompt = llm.format_prompt(question)
        return prompt, reasoning

    class Dataset:
        def __init__(self):
            self.data = rft_data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return len(self.data)

    class TokenizedDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, dataset, format_example):
            self.tokenizer = tokenizer
            self.dataset = dataset
            self.format_example = format_example

        def __getitem__(self, idx):
            prompt, reasoning = self.format_example(self.dataset[idx])
            txt = f"{prompt} {reasoning}{self.tokenizer.eos_token}"

            tokens = self.tokenizer(
                txt,
                truncation=True,
                padding="max_length",
                max_length=384,
                return_tensors="pt",
            )
            tokens = {k: v.squeeze(0) for k, v in tokens.items()}

            prompt_ids = self.tokenizer(prompt)["input_ids"]
            prompt_len = len(prompt_ids)

            labels = tokens["input_ids"].clone()
            labels[:prompt_len] = -100
            labels[tokens["attention_mask"] == 0] = -100
            tokens["labels"] = labels

            return tokens

        def __len__(self):
            return len(self.dataset)

    trainset = Dataset()
    tokenized_dataset = TokenizedDataset(tokenizer, trainset, format_example)

    # Keep your original training hyperparameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=9,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        learning_rate=3e-3,
        save_total_limit=1,
        save_strategy="epoch",
        eval_strategy="no",
        logging_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Keep your original save
    save_path = Path(output_dir)
    trainer.save_model(save_path)

    # Additional save to the exact folder the grader expects
    final_dir = Path(__file__).parent / "rft_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(final_dir)

    test_model(str(final_dir))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})