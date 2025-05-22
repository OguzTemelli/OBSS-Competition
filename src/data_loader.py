import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor

class CaptionDataset(Dataset):
    def __init__(self, csv_file, image_dir, processor, max_length=50):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_id'])
        image = Image.open(image_path).convert('RGB')
        caption = row['caption']
        inputs = self.processor(
            images=image,
            text=caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Squeeze to remove batch dim
        pixel_values = inputs.pixel_values.squeeze()
        input_ids    = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

def get_dataloader(csv_file, image_dir, processor, batch_size=8, shuffle=True, num_workers=4):
    dataset = CaptionDataset(csv_file, image_dir, processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 