#from src.data_loader import get_dataloader
from data_loader import get_dataloader
from transformers import BlipProcessor

def test_dataloader_train():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    dataloader = get_dataloader(
        csv_file='../data/train/train.csv',
        image_dir='../data/train/images',
        processor=processor,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    batch = next(iter(dataloader))
    print("[TRAIN] Batch shapes:")
    print(f"pixel_values: {batch['pixel_values'].shape}")
    print(f"input_ids: {batch['input_ids'].shape}")
    print(f"attention_mask: {batch['attention_mask'].shape}")
    return batch

def test_dataloader_test():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    dataloader = get_dataloader(
        csv_file='../data/test/test.csv',
        image_dir='../data/test/images',
        processor=processor,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        is_test=True
    )
    batch = next(iter(dataloader))
    print("[TEST] Batch shapes:")
    print(f"pixel_values: {batch['pixel_values'].shape}")
    print(f"image_ids: {batch['image_id']}")
    return batch

if __name__ == '__main__':
    test_dataloader_train()
    test_dataloader_test() 