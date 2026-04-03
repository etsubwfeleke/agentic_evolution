# src/predict.py
# ============================================================
# Prediction functions for a given company
# ============================================================

import pandas as pd
import numpy as np


def get_company_profile(company_name, artifacts):
    """
    Retrieves the most recent firm-year record
    for a given company name.
    """
    merged = artifacts['merged']

    # Find matching rows
    matches = merged[
        merged['conm'].str.upper() == company_name.upper()
    ]

    if matches.empty:
        return None

    # Get most recent year
    latest = matches.loc[matches['fyear'].idxmax()]
    return latest


def predict_company(company_name, artifacts):
    """
    Given a company name, returns:
    - Predicted revenue growth
    - Predicted growth tier
    - Tier probabilities
    - AI subfield profile
    - Most recent year in dataset
    """
    profile = get_company_profile(company_name, artifacts)

    if profile is None:
        return {
            'error': f"Company '{company_name}' not found in dataset.",
            'available_companies': artifacts['company_list']
        }

    # Build feature vector
    feature_names = artifacts['feature_names']
    input_df      = pd.DataFrame([profile[feature_names]])
    input_proc    = artifacts['preprocessor'].transform(input_df)

    # Predictions
    reg_pred  = artifacts['reg_model'].predict(input_proc)[0]
    cls_pred  = artifacts['cls_model'].predict(input_proc)[0]
    cls_proba = artifacts['cls_model'].predict_proba(input_proc)[0]
    tier      = artifacts['le'].classes_[cls_pred]

    # AI subfield profile
    subfields = {
        'Machine Learning'          : round(float(profile['share_ml']), 3),
        'NLP'                       : round(float(profile['share_nlp']), 3),
        'Computer Vision'           : round(float(profile['share_vision']), 3),
        'Planning & Reasoning'      : round(float(profile['share_planning']), 3),
        'Hardware'                  : round(float(profile['share_hardware']), 3),
        'Speech Recognition'        : round(float(profile['share_speech']), 3),
        'Knowledge Representation'  : round(float(profile['share_kr']), 3),
        'Evolutionary Computation'  : round(float(profile['share_evo']), 3),
    }

    # Sort by share descending
    subfields = dict(
        sorted(subfields.items(), key=lambda x: x[1], reverse=True)
    )

    # Top 3 subfields
    top_subfields = [k for k, v in subfields.items() if v > 0][:3]

    return {
        'company'          : company_name,
        'year'             : int(profile['fyear']),
        'sector'           : profile['sector'],
        'patent_volume'    : int(profile['patent_volume']),
        'avg_breadth'      : round(float(profile['avg_breadth']), 2),
        'ai_intensity'     : round(float(profile['ai_intensity_score']), 4),
        'predicted_growth' : round(float(reg_pred), 4),
        'growth_tier'      : tier,
        'tier_probabilities': {
            artifacts['le'].classes_[i]: round(float(p), 3)
            for i, p in enumerate(cls_proba)
        },
        'subfield_profile' : subfields,
        'top_subfields'    : top_subfields,
    }

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.load_artifacts import load_artifacts

    artifacts = load_artifacts()
    result    = predict_company('MICROSOFT CORP', artifacts)

    print(f"\nCompany:          {result['company']}")
    print(f"Year:             {result['year']}")
    print(f"Sector:           {result['sector']}")
    print(f"Growth Tier:      {result['growth_tier']}")
    print(f"Predicted Growth: {result['predicted_growth']*100:.1f}%")
    print(f"\nAI Subfield Profile:")
    for k, v in result['subfield_profile'].items():
        if v > 0:
            bar = '█' * int(v * 20)
            print(f"  {k:<30} {bar:<20} {v*100:.1f}%")
    print(f"\nTop Subfields: {result['top_subfields']}")