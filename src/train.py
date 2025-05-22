import argparse
import torch
from torch.optim import AdamW
from tqdm import tqdm
import os

from model import load_model, save_model
from data_loader import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Train BLIP model for image captioning')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum caption length')
    return parser.parse_args()

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and processor
    model, processor = load_model(device=device)
    
    # Create dataloader
    train_dataloader = get_dataloader(
        args.train_csv,
        args.image_dir,
        processor,
        batch_size=args.batch_size
    )
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
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
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Print epoch statistics
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch + 1}')
        save_model(model, processor, checkpoint_dir)

if __name__ == '__main__':
    args = parse_args()
    train(args) 