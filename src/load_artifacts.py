# src/load_artifacts.py
# ============================================================
# Loads all Phase 3 trained artifacts and dataset
# ============================================================

import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Base paths
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR  = BASE_DIR / 'data'


def rebuild_preprocessor(merged, feature_names):
    sector_cols     = [c for c in feature_names if c.startswith('sector_')]
    continuous_cols = [c for c in feature_names if c not in sector_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), continuous_cols),
            ('passthrough', 'passthrough', sector_cols)
        ]
    )
    X = merged[feature_names].copy()
    preprocessor.fit(X)
    return preprocessor


def load_artifacts():
    # --------------------------------------------------------
    # 1. Load models
    # --------------------------------------------------------
    reg_model    = joblib.load(MODEL_DIR / 'regression_model.pkl')
    cls_model    = joblib.load(MODEL_DIR / 'classification_model.pkl')
    le           = joblib.load(MODEL_DIR / 'label_encoder.pkl')

    # --------------------------------------------------------
    # 2. Load metadata
    # --------------------------------------------------------
    with open(MODEL_DIR / 'model_metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']

    # --------------------------------------------------------
    # 3. Load and engineer dataset
    # --------------------------------------------------------
    merged = pd.read_csv(DATA_DIR / 'phase3_merged.csv')

    # Normalize
    scaler = MinMaxScaler()
    merged['patent_volume_norm'] = scaler.fit_transform(
        merged[['patent_volume']]
    )
    merged['avg_breadth_norm'] = scaler.fit_transform(
        merged[['avg_breadth']]
    )

    # LinkedIn signal
    linkedin_signal          = 0.4284
    merged['linkedin_signal'] = linkedin_signal

    # AI Intensity Score
    merged['ai_intensity_score'] = (
        0.5 * merged['patent_volume_norm'] +
        0.3 * merged['avg_breadth_norm']   +
        0.2 * linkedin_signal
    )

    # Sector dummies
    sector_dummies = pd.get_dummies(
        merged['sector'], prefix='sector', drop_first=True
    )
    merged = pd.concat([merged, sector_dummies], axis=1)

    # --------------------------------------------------------
    # 4. Rebuild preprocessor locally to avoid version issues
    # --------------------------------------------------------
    preprocessor = rebuild_preprocessor(merged, feature_names)

    # --------------------------------------------------------
    # 5. Feature matrix
    # --------------------------------------------------------
    X            = merged[feature_names].copy()
    company_list = sorted(merged['conm'].unique().tolist())

    print(f"Artifacts loaded successfully.")
    print(f"  Firms available: {len(company_list)}")
    print(f"  Features:        {len(feature_names)}")

    return {
        'reg_model'      : reg_model,
        'cls_model'      : cls_model,
        'preprocessor'   : preprocessor,
        'le'             : le,
        'metadata'       : metadata,
        'merged'         : merged,
        'X'              : X,
        'feature_names'  : feature_names,
        'company_list'   : company_list,
        'linkedin_signal': linkedin_signal,
    }


if __name__ == '__main__':
    artifacts = load_artifacts()
    print(f"\nCompanies in dataset:")
    for c in artifacts['company_list']:
        print(f"  {c}")