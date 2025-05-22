import os
import torch
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
from data_loader import get_dataloader
from model import load_model, save_model
import numpy as np
from torch.optim import AdamW


def train(
    train_csv: str,
    image_dir: str,
    output_dir: str,
    batch_size: int = 8,
    epochs: int = 5,
    learning_rate: float = 5e-5,
    device: torch.device = None
):
    """
    Fine-tune BLIP model on the training data.
    
    Args:
        train_csv (str): Path to training CSV file
        image_dir (str): Path to training images directory
        output_dir (str): Directory to save checkpoints
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (torch.device): Device to train on
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load model and processor
    model, processor = load_model(device=device)
    model.train()

    # Create dataloader
    train_dataloader = get_dataloader(
        csv_file=train_csv,
        image_dir=image_dir,
        processor=processor,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        # Calculate average loss
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        save_model(model, processor, checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP model for image captioning")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to training images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    ) 