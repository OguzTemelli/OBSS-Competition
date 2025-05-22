import argparse
import pandas as pd
import numpy as np
from scipy import linalg
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate image captions using FGD')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth CSV')
    return parser.parse_args()

def calculate_fgd(predictions, references):
    """
    Calculate Fréchet GPT Distance between predicted and reference captions
    
    Args:
        predictions (list): List of predicted captions
        references (list): List of reference captions
    
    Returns:
        float: FGD score
    """
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embeddings
    print("Computing embeddings for predictions...")
    pred_embeddings = model.encode(predictions, show_progress_bar=True)
    
    print("Computing embeddings for references...")
    ref_embeddings = model.encode(references, show_progress_bar=True)
    
    # Calculate mean and covariance
    mu_pred = np.mean(pred_embeddings, axis=0)
    mu_ref = np.mean(ref_embeddings, axis=0)
    sigma_pred = np.cov(pred_embeddings, rowvar=False)
    sigma_ref = np.cov(ref_embeddings, rowvar=False)
    
    # Calculate FGD
    ssdiff = np.sum((mu_pred - mu_ref) ** 2.0)
    covmean = linalg.sqrtm(sigma_pred.dot(sigma_ref))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fgd = ssdiff + np.trace(sigma_pred + sigma_ref - 2.0 * covmean)
    return fgd

def evaluate(args):
    # Load predictions and ground truth
    pred_df = pd.read_csv(args.predictions)
    gt_df = pd.read_csv(args.ground_truth)
    
    # Ensure same order of images
    pred_df = pred_df.sort_values('image_id')
    gt_df = gt_df.sort_values('image_id')
    
    # Get captions
    predictions = pred_df['caption'].tolist()
    references = gt_df['caption'].tolist()
    
    # Calculate FGD
    fgd_score = calculate_fgd(predictions, references)
    print(f"FGD Score: {fgd_score:.4f}")

if __name__ == '__main__':
    args = parse_args()
    evaluate(args) 