import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_model(model_name="Salesforce/blip-image-captioning-base", device=None):
    """
    Load BLIP model and processor
    
    Args:
        model_name (str): Name of the pretrained model
        device (torch.device): Device to load the model on
    
    Returns:
        tuple: (model, processor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    return model, processor

def save_model(model, processor, output_dir):
    """
    Save model and processor
    
    Args:
        model: BLIP model
        processor: BLIP processor
        output_dir (str): Directory to save the model
    """
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