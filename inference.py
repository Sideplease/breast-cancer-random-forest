import argparse
import joblib
import pandas as pd
from pathlib import Path

FEATURE_BASE = ['radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points','symmetry','fractal_dimension']
FEATURES = [f'{stat}_{feat}' for stat in ['mean','se','worst'] for feat in FEATURE_BASE]
THRESHOLD = 0.20


def load_wdbc(path):
    return pd.read_csv(path, header=None, names=['id','diagnosis'] + FEATURES)


def main():
    parser = argparse.ArgumentParser(description='WDBC Random Forest inference')
    parser.add_argument('--data', default='data/wdbc.data', help='Path to WDBC .data file')
    parser.add_argument('--row', type=int, default=0, help='Zero-based row to score')
    args = parser.parse_args()

    model = joblib.load('models/rf_model.joblib')
    df = load_wdbc(args.data)
    if args.row < 0 or args.row >= len(df):
        raise ValueError(f'row must be between 0 and {len(df)-1}')
    X = df.loc[[args.row], FEATURES]
    prob_m = float(model.predict_proba(X)[0, 1])
    pred = 'M' if prob_m >= THRESHOLD else 'B'
    print(f'Predicted class (M/B): {pred}')
    print(f'Probability of malignancy: {prob_m:.4f}')
    print(f'Threshold used: {THRESHOLD:.2f}')
    print(f'Class probabilities: B={1-prob_m:.4f}, M={prob_m:.4f}')

if __name__ == '__main__':
    main()
