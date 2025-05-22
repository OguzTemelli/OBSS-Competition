import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor

class ImageCaptioningDataset(Dataset):
    def __init__(self, csv_path, image_dir, processor, max_length=50):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        caption = row['caption']

        # Load and preprocess image
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        # Process image and text
        inputs = self.processor(
            image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }

def get_dataloader(csv_path, image_dir, processor, batch_size=8, shuffle=True):
    dataset = ImageCaptioningDataset(csv_path, image_dir, processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    ) 