# src/skills_mapping.py
# ============================================================
# Maps AI patent subfields to job market skills
# ============================================================

# Subfield to skills mapping
# Based on Phase 2 LinkedIn analysis + industry knowledge
SUBFIELD_SKILLS = {
    'Planning & Reasoning': [
        'LangChain', 'AutoGen', 'CrewAI', 'LangGraph',
        'Prompt Engineering', 'RAG', 'Agentic AI',
        'Multi-Agent Systems', 'Tool Use', 'Function Calling'
    ],
    'NLP': [
        'Transformers', 'HuggingFace', 'LLM Fine-tuning',
        'BERT', 'GPT', 'Text Classification',
        'Named Entity Recognition', 'Semantic Search',
        'Vector Databases', 'Embeddings'
    ],
    'Machine Learning': [
        'Scikit-learn', 'XGBoost', 'Feature Engineering',
        'Model Evaluation', 'MLOps', 'Experiment Tracking',
        'MLflow', 'Weights & Biases', 'A/B Testing',
        'Statistical Modeling'
    ],
    'Hardware': [
        'CUDA', 'GPU Programming', 'TensorRT',
        'Model Optimization', 'Quantization', 'Pruning',
        'ONNX', 'Edge AI', 'Embedded Systems',
        'High Performance Computing'
    ],
    'Computer Vision': [
        'PyTorch', 'OpenCV', 'YOLO', 'Image Segmentation',
        'Object Detection', 'CNNs', 'Vision Transformers',
        'Diffusion Models', 'Image Generation',
        'Video Understanding'
    ],
    'Speech Recognition': [
        'Whisper', 'Speech-to-Text', 'Audio Processing',
        'Text-to-Speech', 'Voice AI', 'Wav2Vec',
        'Speaker Diarization', 'Audio ML',
        'Natural Language Understanding', 'Conversational AI'
    ],
    'Knowledge Representation': [
        'Knowledge Graphs', 'Neo4j', 'Ontologies',
        'Semantic Web', 'RDF', 'SPARQL',
        'Graph Neural Networks', 'Reasoning Systems',
        'Logic Programming', 'Symbolic AI'
    ],
    'Evolutionary Computation': [
        'Genetic Algorithms', 'Reinforcement Learning',
        'Optimization', 'Hyperparameter Tuning',
        'Neural Architecture Search', 'Bayesian Optimization',
        'Evolutionary Strategies', 'Swarm Intelligence',
        'Multi-objective Optimization', 'AutoML'
    ]
}

# Seniority level mapping based on subfield
SENIORITY_SIGNALS = {
    'Planning & Reasoning' : 'Senior / Staff Engineer',
    'NLP'                  : 'Mid-Senior Engineer',
    'Machine Learning'     : 'Mid-level Engineer',
    'Hardware'             : 'Senior / Principal Engineer',
    'Computer Vision'      : 'Mid-Senior Engineer',
    'Speech Recognition'   : 'Mid-Senior Engineer',
    'Knowledge Representation': 'Senior / Research Engineer',
    'Evolutionary Computation': 'Research Scientist',
}


def get_skills_for_company(prediction_result):
    """
    Given a prediction result from predict.py,
    returns ranked skills based on the company's
    AI subfield profile.
    """
    subfields  = prediction_result['subfield_profile']
    top_fields = [k for k, v in subfields.items() if v > 0.05][:3]

    if not top_fields:
        top_fields = list(subfields.keys())[:2]

    # Build weighted skill list
    skill_scores = {}
    for field in subfields:
        weight = subfields[field]
        if weight < 0.01:
            continue
        for skill in SUBFIELD_SKILLS.get(field, []):
            if skill in skill_scores:
                skill_scores[skill] += weight
            else:
                skill_scores[skill] = weight

    # Sort by score
    ranked_skills = sorted(
        skill_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top 10 skills
    top_skills = [skill for skill, _ in ranked_skills[:10]]

    # Seniority signals for top fields
    seniority = [
        SENIORITY_SIGNALS.get(f, 'Mid-level Engineer')
        for f in top_fields
    ]

    return {
        'top_skills'    : top_skills,
        'top_fields'    : top_fields,
        'seniority'     : list(set(seniority)),
        'all_skills'    : dict(ranked_skills[:15])
    }


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.load_artifacts import load_artifacts
    artifacts = load_artifacts()
    # Test
    mock_result = {
        'subfield_profile': {
            'Planning & Reasoning' : 0.40,
            'NLP'                  : 0.30,
            'Hardware'             : 0.15,
            'Machine Learning'     : 0.10,
            'Computer Vision'      : 0.05,
        }
    }
    skills = get_skills_for_company(mock_result)
    print("Top Skills:")
    for skill in skills['top_skills']:
        print(f"  - {skill}")
    print(f"\nSeniority: {skills['seniority']}")