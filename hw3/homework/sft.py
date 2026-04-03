from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=384)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full
    
# def tokenize(tokenizer, question: str, answer: str):
#     messages = [
#         {"role": "system", "content": (
#             "You are an expert unit conversion assistant. Always convert units with correct math, and clearly show the conversion factor. "
#             "At the end, show the final result wrapped like this: <answer>208.709827875</answer>. Only use one <answer> tag and do not repeat the question."
#         )},
#         {"role": "user", "content": question},
#         {"role": "assistant", "content": answer}
#     ]

#     tokenizer.padding_side = "right"
#     tokenizer.pad_token = tokenizer.eos_token

#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=384,
#     )

#     attention_mask = (input_ids != tokenizer.pad_token_id).long()

#     # compute where assistant response starts
#     user_prompt = tokenizer.apply_chat_template(
#         messages[:-1],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=384,
#     )

#     prompt_len = user_prompt.shape[-1]

#     labels = input_ids.clone()
#     labels[:, :prompt_len] = -100
#     labels[attention_mask == 0] = -100

#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }




def format_example(prompt: str, answer: str) -> dict[str, str]:
    rounded_answer = round(float(answer), 4)
    answer_str = f"<answer>{rounded_answer}</answer>"
    return {"question": prompt, "answer": answer_str}
    

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)
        
def train_model(
    output_dir: str,
    **kwargs,
):
    from pathlib import Path
    import torch
    from transformers import TrainingArguments, Trainer

    # Import LoRA utilities
    from peft import get_peft_model, LoraConfig, TaskType

    # Load base model and tokenizer
    llm = BaseLLM()
    tokenizer = llm.tokenizer

    # Setup LoRA config (adjust r so adapter <20MB, alpha ~4-5*r)
    r = 8  # You might need to tune this, e.g., 8, 16, etc.
    lora_alpha = 80
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Add LoRA adapter
    llm.model = get_peft_model(llm.model, lora_config)
    if torch.cuda.is_available():
        llm.model.enable_input_require_grads()

    # Prepare dataset
    trainset = Dataset("train")
    tokenized_dataset = TokenizedDataset(tokenizer, trainset, format_example)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        learning_rate=1e-3, 
        save_total_limit=1,
        save_strategy="epoch",
        eval_strategy="no",
        logging_strategy="epoch",
        remove_unused_columns=False,
    )

    # Trainer setup
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter to homework/sft_model
    save_path = Path(__file__).parent / "sft_model"
    trainer.save_model(save_path)
    test_model(str(save_path))


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
