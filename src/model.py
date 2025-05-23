import os
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

def load_model(model_name="Salesforce/blip-image-captioning-base", device=None):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor

def save_model(model, processor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)