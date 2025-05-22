import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_model(model_name="Salesforce/blip-image-captioning-base", device=None):
    """
    Load BLIP model and processor.
    Returns (model, processor) moved to the specified device.
    
    Args:
        model_name (str): HuggingFace model name or path
        device (torch.device, optional): Device to load model on. If None, uses CUDA if available, else CPU.
    
    Returns:
        tuple: (model, processor) both moved to specified device
    """
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor

def save_model(model, processor, output_dir):
    """
    Save model and processor to output_dir.
    
    Args:
        model (BlipForConditionalGeneration): The model to save
        processor (BlipProcessor): The processor to save
        output_dir (str): Directory to save model and processor
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

def load_checkpoint(model, checkpoint_path):
    """
    Load model checkpoint
    
    Args:
        model: BLIP model
        checkpoint_path (str): Path to the checkpoint file
    
    Returns:
        model: Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model 