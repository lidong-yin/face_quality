import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from models import QualityRegressorHeder
from utils import l2norm, is_l2norm


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Init model, input_dim: {args.input_dim}...")
    model = QualityRegressorHeder(input_dim=args.input_dim)
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loading data from {args.feature_file}...")
    try:
        if '.npy' in args.feature_file:
            features = np.load(args.feature_file, mmap_mode='r')
        elif '.parquet' in args.feature_file:
            data = pd.read_parquet(args.feature_file)
            features = np.vstack(data['feature'])
        elif '.pkl' in args.feature_file:
            data = pd.read_pickle(args.feature_file)
            features = np.vstack(data['feature'])
        else:
            print('file error')
    except Exception as e:
        print('input path error.')
        return

    if is_l2norm(features, min(10000, len(features))) == False:
        print('l2norm the feats')
    print(f"Input feats: {len(features)}")

    feature_tensor = torch.from_numpy(features).float()

    all_predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(feature_tensor), args.batch_size), desc="Predicting Quality"):
            batch_features = feature_tensor[i:i+args.batch_size].to(device)
            batch_predictions = model(batch_features)
            all_predictions.append(batch_predictions.cpu().numpy())
    
    predictions = np.vstack(all_predictions).flatten() # flatten()将 (N, 1) 数组变为 (N,)
    np.save(args.output, predictions)
    print(f"Inference done, save file: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Quality Model Inference")

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--feature_file', type=str, required=True, help="feats path or df path")
    parser.add_argument('--output', type=str, default="inference_results.npy")
    parser.add_argument('--input_dim', type=int, default=1024, help='feats dim')
    parser.add_argument('--batch_size', type=int, default=16384)

    args = parser.parse_args()
    run_inference(args)