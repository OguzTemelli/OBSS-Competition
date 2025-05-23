import os
import torch
import pandas as pd
from tqdm import tqdm
from model import load_model
from transformers import BlipProcessor

def generate_captions(model_name_or_dir, csv_file, image_dir, output_file, batch_size=8, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    model, processor = load_model(model_name_or_dir, device=device)
    model.eval()
    df = pd.read_csv(csv_file)
    captions = []

    for idx in tqdm(range(0, len(df), batch_size), desc="Inference"):
        batch_df = df.iloc[idx:idx + batch_size]
        images = [os.path.join(image_dir, f"{img_id}.jpg") for img_id in batch_df["image_id"]]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=5)
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        captions.extend(decoded)

    df["caption"] = captions
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    generate_captions(**vars(args))