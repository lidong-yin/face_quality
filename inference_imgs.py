import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from dataset import FaceInferenceDataset
from models import FaceQualityModel


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Init model, backbone: {args.backbone}...")
    model = FaceQualityModel(backbone=args.backbone, pretrained=True)

    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    if os.path.isdir(args.input_dir):
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, '**', ext), recursive=True))
    elif os.path.isfile(args.input_dir):
        if '.pkl' in args.input_dir or '.parquet' in args.input_dir:
            data = pd.read_pickle(args.input_dir) if '.pkl' in args.input_dir else pd.read_parquet(args.input_dir)
            image_paths = data['fpath'].values
        else:
            image_paths = open(args.input_dir).readlines()
            image_paths = [f.strip() for f in image_paths]
    else:
        print('input path error.')
    
    print(f"Input imgs: {len(image_paths)}")

    dataset = FaceInferenceDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    all_predictions = []
    with torch.no_grad():
        for batch_imgs, batch_paths in tqdm(dataloader, desc="Predicting Quality"):
            batch_imgs = batch_imgs.to(device)
            batch_predictions = model(batch_imgs)
            all_predictions.append(batch_predictions.cpu().numpy())

    predictions = np.vstack(all_predictions).flatten() # flatten()将 (N, 1) 数组变为 (N,)
    np.save(args.output, predictions)
    print(f"Inference done, save file: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Quality Model Inference")
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True, help="input fpath list or imgs path")
    parser.add_argument('--output', type=str, default="inference_results.npy")
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    run_inference(args)