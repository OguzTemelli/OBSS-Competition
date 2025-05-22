import os
import torch
from tqdm import tqdm
from model import load_model
from data_loader import CaptionDataset
from transformers import BlipProcessor
import pandas as pd


def generate_captions(
    model_name_or_dir: str,
    csv_file: str,
    image_dir: str,
    output_file: str,
    batch_size: int = 8,
    device: torch.device = None
):
    """
    Load a fine-tuned BLIP model and generate captions for images listed in csv_file.
    Expects csv_file to have a column 'image_id'.
    Saves a CSV with columns ['image_id', 'caption'] to output_file.
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # Load model and processor
    model, processor = load_model(model_name_or_dir, device=device)
    model.eval()

    # Read test CSV
    df = pd.read_csv(csv_file)

    captions = []

    for idx in tqdm(range(0, len(df), batch_size), desc="Inference"):
        batch_df = df.iloc[idx : idx + batch_size]
        images = []
        for img_id in batch_df['image_id'].tolist():
            path = os.path.join(image_dir, f"{img_id}.jpg")  # Add .jpg extension
            images.append(path)

        # Process images
        inputs = processor(images=images, return_tensors="pt").to(device)

        # Generate captions
        outputs = model.generate(
            **inputs,
            max_length=processor.tokenizer.model_max_length,
            num_beams=5,
            length_penalty=1.0
        )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        captions.extend(decoded)

    # Save results
    df['caption'] = captions
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inference script for image captioning')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to fine-tuned model or model name')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV with image_id column')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of test images')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    args = parser.parse_args()

    generate_captions(
        model_name_or_dir=args.model_dir,
        csv_file=args.test_csv,
        image_dir=args.image_dir,
        output_file=args.output_csv,
        batch_size=args.batch_size
    ) 