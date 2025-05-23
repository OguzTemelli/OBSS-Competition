import os
import torch
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.optim import AdamW
from tqdm import tqdm
from data_loader import CaptionDataset
from model import save_model
import argparse

def train(train_csv, image_dir, output_dir, batch_size=8, epochs=3, learning_rate=5e-5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)

    dataset = CaptionDataset(train_csv, image_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'image_id'}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")
        save_model(model, processor, os.path.join(output_dir, f"checkpoint-epoch-{epoch}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    train(**vars(args))