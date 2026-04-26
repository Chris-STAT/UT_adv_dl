from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# LOAD
# =========================
def load(model_name: str = "clip_model"):
    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, str(model_path)).to(device)

    clip.model.load_pretrained(model_path)
    clip.model.eval()

    return clip


# =========================
# DATA COLLATOR
# =========================
def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], 0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([pad_tensor(f["labels"], -100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


# =========================
# DATASET
# =========================
class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.processor = processor

        self.image_processor = tv.transforms.Compose([
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)

        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt")

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": text_inputs["input_ids"].squeeze(0),
        }


# =========================
# CLIP MODEL
# =========================
class CLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, proj_dim=256):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        vision_dim = vision_encoder.config.hidden_size
        text_dim = text_encoder.config.hidden_size

        #self.image_proj = nn.Linear(vision_dim, proj_dim)
        #self.text_proj = nn.Linear(text_dim, proj_dim)
        self.image_proj = nn.Linear(vision_dim, proj_dim).to(vision_encoder.dtype)
        self.text_proj = nn.Linear(text_dim, proj_dim).to(text_encoder.dtype)

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

    def encode_image(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values)
        feat = out.last_hidden_state[:, 0]
        feat = self.image_proj(feat)
        return F.normalize(feat, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        feat = out.last_hidden_state[:, 0]
        feat = self.text_proj(feat)
        return F.normalize(feat, dim=-1)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        pixel_values = pixel_values.to(self.vision_encoder.dtype)
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T

        return image_features, text_features, logits

    def save_pretrained(self, save_directory):
        torch.save(self.state_dict(), Path(save_directory) / "clip.pt")

    def load_pretrained(self, load_directory):
        path = Path(load_directory) / "clip.pt"
        if path.exists():
            self.load_state_dict(torch.load(path))


# =========================
# LOSS
# =========================
def compute_clip_loss(outputs, labels, num_items_in_batch=None):
    _, _, logits = outputs

    batch_size = logits.shape[0]
    target = torch.arange(batch_size, device=logits.device)

    loss_i = F.cross_entropy(logits, target)
    loss_t = F.cross_entropy(logits.T, target)

    return (loss_i + loss_t) / 2


# =========================
# TRAIN
# =========================
def train(
    output_dir="clip_model",
    num_train_epochs=1,
    per_device_train_batch_size=256,
    learning_rate=5e-4,
):
    vlm = BaseVLM()

    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model

    model = CLIP(vision_encoder, text_encoder).to(device)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)

    train_dataset = CaptionDataset("train")
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()

    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    return model


# =========================
# TEST
# =========================
def test(ckpt_path, val_dataset="valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)
    clip = load(ckpt_path)

    correct = 0
    total = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")

        pixel = tv.transforms.ToTensor()(image).unsqueeze(0).to(device)

        text_inputs = processor(
            text=[c + processor.tokenizer.eos_token for c in pair["candidates"]],
            return_tensors="pt",
            padding=True,
        )

        input_ids = text_inputs["input_ids"].to(device)
        attn = text_inputs["attention_mask"].to(device)

        img_feat, txt_feat, _ = clip(pixel, input_ids, attn)

        pred = (img_feat @ txt_feat.T).argmax()

        if pred == pair["correct_index"]:
            correct += 1
        total += 1

    print("Accuracy:", correct / total)


# =========================
# MAIN
# =========================
def main():
    from fire import Fire
    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()