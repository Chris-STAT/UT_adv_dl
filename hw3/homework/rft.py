from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    data_path: str = "data/rft.json",
    learning_rate: float = 1e-4,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 32,
    lora_rank: int = 16,
    lora_alpha: int = 64,
    max_length: int = 192,
    **kwargs,
):
    """
    Train an RFT model on question / reasoning / answer triples saved in data/rft.json.

    Expected JSON format:
    [
      ["question text", 6000.0, "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"],
      ...
    ]
    """
    import json
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()

    # LoRA setup
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, peft_config)

    # Helps with gradient checkpointing on GPU
    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    # Load generated RFT data
    data_path = Path(data_path)
    with data_path.open() as f:
        rft_data = json.load(f)

    def tokenize(tokenizer, question: str, answer: str):
        """
        Tokenize prompt + target and supervise only the target tokens.
        """
        full_text = f"{question} {answer}{tokenizer.eos_token}"

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        full = tokenizer(full_text, padding="max_length", truncation=True, max_length=max_length)

        input_ids = full["input_ids"]
        question_len = len(tokenizer(question)["input_ids"])

        labels = [-100] * question_len + input_ids[question_len:]

        for i in range(len(labels)):
            if full["attention_mask"][i] == 0:
                labels[i] = -100

        full["labels"] = labels
        return full

    def format_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
        """
        Keep training-time prompt aligned with BaseLLM.format_prompt.
        The target is the full reasoning trace ending in <answer>...</answer>.
        """
        return {
            "question": llm.format_prompt(question),
            "answer": reasoning.strip(),
        }

    class TokenizedRFTDataset:
        def __init__(self, tokenizer, data):
            self.tokenizer = tokenizer
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            q, a, reasoning = self.data[idx]
            formatted = format_example(q, a, reasoning)
            return tokenize(self.tokenizer, **formatted)

    train_dataset = TokenizedRFTDataset(llm.tokenizer, rft_data)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    final_dir = Path(__file__).parent / "rft_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    llm.model.save_pretrained(final_dir)

    test_model(str(final_dir))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})