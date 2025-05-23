import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CaptionDataset(Dataset):
    def __init__(self, csv_file, image_dir, processor):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        caption = row['caption']
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['image_id'] = row['image_id']
        return inputs