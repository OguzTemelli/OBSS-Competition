import numpy as np
import pandas as pd
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from sentence_transformers import SentenceTransformer

def calculate_fgd(solution_embed: np.ndarray, submission_embed: np.ndarray) -> float:
    fgd_list = []
    for sol_emb, sub_emb in zip(solution_embed, submission_embed):
        e1 = np.stack([sol_emb, sol_emb])
        e2 = np.stack([sub_emb, sub_emb])
        mu1, sigma1 = e1.mean(axis=0), cov(e1, rowvar=False)
        mu2, sigma2 = e2.mean(axis=0), cov(e2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if iscomplexobj(covmean):
            covmean = covmean.real
        fgd = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        fgd_list.append(fgd)
    return float(np.mean(fgd_list))

def evaluate_fgd(ground_truth_csv, prediction_csv):
    gt_df = pd.read_csv(ground_truth_csv)
    pred_df = pd.read_csv(prediction_csv)
    merged = pd.merge(gt_df, pred_df, on="image_id", how="inner")

    model = SentenceTransformer("thenlper/gte-small")
    gt_embeds = model.encode(merged["caption_x"].tolist(), convert_to_numpy=True)
    pred_embeds = model.encode(merged["caption_y"].tolist(), convert_to_numpy=True)

    fgd_score = calculate_fgd(gt_embeds, pred_embeds)
    print(f"FGD Score: {fgd_score:.4f}")
    return fgd_score

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_csv", required=True)
    parser.add_argument("--prediction_csv", required=True)
    args = parser.parse_args()
    evaluate_fgd(args.ground_truth_csv, args.prediction_csv)