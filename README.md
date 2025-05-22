# OBSS AI Image Captioning Challenge

This repository contains the implementation for the OBSS AI Image Captioning Challenge, where the goal is to generate accurate and meaningful captions for images using deep learning techniques.

## Task Description

The challenge involves creating an image captioning model that can generate natural language descriptions for images. The model should be able to understand the content of images and generate grammatically correct, meaningful captions that accurately describe the visual content.

## Dataset

The dataset consists of:
- `train.csv`: Contains image IDs and their corresponding captions for training
- `test.csv`: Contains image IDs for which captions need to be generated
- Images from the Open Images dataset

## Environment Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Pillow
- pandas
- numpy
- tqdm
- scikit-learn

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
obss-image-captioning-challenge/
├── data/                   # Data directory
│   ├── train/             # Training images
│   └── test/              # Test images
├── src/                   # Source code
│   ├── models/           # Model architectures
│   ├── utils/            # Utility functions
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── predict.py        # Inference script
├── notebooks/            # Jupyter notebooks for exploration
├── configs/              # Configuration files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### Training

To train the model:
```bash
python src/train.py --config configs/train_config.yaml
```

### Inference

To generate captions for test images:
```bash
python src/predict.py --model_path path/to/model --test_data path/to/test.csv
```

### Evaluation

To evaluate the model's performance:
```bash
python src/evaluate.py --predictions path/to/predictions.csv --ground_truth path/to/ground_truth.csv
```

## Repository Link

[OBSS AI Image Captioning Challenge](https://github.com/yourusername/obss-image-captioning-challenge)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 