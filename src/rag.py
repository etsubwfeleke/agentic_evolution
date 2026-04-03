# src/rag.py
# ============================================================
# RAG Layer — Gemini + Live News Retrieval
# ============================================================

import os
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# Gemini Career Intelligence Agent
# ============================================================

def generate_career_insight(
    company_name,
    prediction_result,
    skills_result
):
    """
    Uses Gemini to synthesize patent profile,
    skills, and news into actionable career advice.
    """
    # Build context
    subfield_text = '\n'.join([
        f"  - {k}: {v*100:.1f}%"
        for k, v in prediction_result['subfield_profile'].items()
        if v > 0.05
    ])

    skills_text = ', '.join(skills_result['top_skills'][:8])

    prompt = f"""
You are a career intelligence analyst specializing in AI and 
machine learning job markets. A job seeker wants to know about 
AI opportunities at {company_name}.

Here is what we know about {company_name} from their AI patent data:

COMPANY AI PROFILE:
- Sector: {prediction_result['sector']}
- Total AI Patents: {prediction_result['patent_volume']}
- Average AI Breadth: {prediction_result['avg_breadth']} subfields per patent
- Top AI Research Areas:
{subfield_text}

RECOMMENDED SKILLS TO LEARN:
{skills_text}

Use your own knowledge about this company's recent 
AI initiatives, products, and hiring patterns to 
enrich your response.

Based on this data, provide a career intelligence report with:
1. What kind of AI work is this company actually doing?
2. What are the top 5 skills a job seeker should prioritize 
   to get hired at this company?
3. What role titles are most likely being hired for?
4. One specific career advice tip for this company.

Keep the response concise, practical, and actionable.
Format with clear sections.
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Could not generate insight: {str(e)}"


# ============================================================
# Main RAG Pipeline
# ============================================================

def run_rag_pipeline(company_name, prediction_result, skills_result):
    """
    Full RAG pipeline:
    1. Generate Gemini career insight
    Returns structured result
    """
    print(f"  Generating career insight with Gemini...")
    insight = generate_career_insight(
        company_name,
        prediction_result,
        skills_result
    )

    return {
        'insight': insight
    }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.load_artifacts  import load_artifacts
    from src.predict         import predict_company
    from src.skills_mapping  import get_skills_for_company

    artifacts  = load_artifacts()
    prediction = predict_company('MICROSOFT CORP', artifacts)
    skills     = get_skills_for_company(prediction)
    rag_result = run_rag_pipeline('MICROSOFT CORP', prediction, skills)

    print("\n" + "="*60)
    print("CAREER INTELLIGENCE REPORT — MICROSOFT CORP")
    print("="*60)
    print(f"\nGemini Insight:")
    print(rag_result['insight'])
