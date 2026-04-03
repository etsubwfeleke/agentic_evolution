from src.predict import predict_company
from src.skills_mapping import get_skills_for_company
from src.rag import run_rag_pipeline


def run_full_analysis(company_name, artifacts):
    """
    Runs the full career intelligence pipeline for a company.

    Steps:
    1. Predict company profile and growth via `predict_company`
    2. Map AI subfields to market-relevant skills via `get_skills_for_company`
    3. Retrieve news and generate narrative insight via `run_rag_pipeline`

    Returns a single dict with keys:
    - company
    - prediction
    - skills
    - news
    - insight

    Handles missing company or pipeline failures gracefully.
    """
    result = {
        'company': company_name,
        'prediction': None,
        'skills': None,
        'news': [],
        'insight': ''
    }

    try:
        prediction = predict_company(company_name, artifacts)
        result['prediction'] = prediction

        if isinstance(prediction, dict) and prediction.get('error'):
            result['insight'] = prediction['error']
            return result

        skills = get_skills_for_company(prediction)
        result['skills'] = skills

        rag_result = run_rag_pipeline(company_name, prediction, skills)
        if isinstance(rag_result, dict):
            result['news'] = rag_result.get('news', [])
            result['insight'] = rag_result.get('insight', '')

        return result

    except Exception as exc:
        result['insight'] = f"Analysis failed for '{company_name}': {str(exc)}"
        return result
