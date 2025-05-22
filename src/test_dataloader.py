from src.data_loader import get_dataloader
from transformers import BlipProcessor

def test_dataloader():
    # Initialize processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Create dataloader with explicit paths and test-friendly settings
    dataloader = get_dataloader(
        csv_file='data/train/train.csv',
        image_dir='data/train/images',
        processor=processor,
        batch_size=2,
        shuffle=True,
        num_workers=0  # Use 0 workers for local testing
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Print shapes
    print("Batch shapes:")
    print(f"pixel_values: {batch['pixel_values'].shape}")
    print(f"input_ids: {batch['input_ids'].shape}")
    print(f"attention_mask: {batch['attention_mask'].shape}")
    
    return batch

if __name__ == '__main__':
    test_dataloader() 