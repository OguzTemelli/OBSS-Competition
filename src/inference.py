import argparse
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from model import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Generate captions for test images')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model checkpoint')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum caption length')
    return parser.parse_args()

def generate_captions(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and processor
    model, processor = load_model(model_name=args.model_dir, device=device)
    model.eval()
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    
    # Generate captions
    captions = []
    for image_id in tqdm(test_df['image_id'], desc='Generating captions'):
        # Load and preprocess image
        image_path = f"{args.image_dir}/{image_id}.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=args.max_length,
                num_beams=5,
                length_penalty=1.0
            )
        
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        captions.append(caption)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'image_id': test_df['image_id'],
        'caption': captions
    })
    
    # Save predictions
    submission_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

if __name__ == '__main__':
    args = parse_args()
    generate_captions(args) 