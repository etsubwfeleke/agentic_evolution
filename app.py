import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.load_artifacts import load_artifacts
from src.agent import run_full_analysis


ARTIFACTS = load_artifacts()
COMPANY_LIST = ARTIFACTS['company_list']

SUBFIELD_COLUMN_MAP = {
    'Planning & Reasoning': 'share_planning',
    'Hardware': 'share_hardware',
    'NLP': 'share_nlp',
    'Machine Learning': 'share_ml',
    'Computer Vision': 'share_vision',
    'Speech Recognition': 'share_speech',
    'Knowledge Representation': 'share_kr',
    'Evolutionary Computation': 'share_evo',
}

COLOR_PALETTE = [
    '#2196F3', '#FF9800', '#4CAF50', '#9C27B0',
    '#F44336', '#00BCD4', '#795548', '#607D8B'
]


def build_empty_plot(message, figsize=(14, 10), title='Company Analysis'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
    ax.axis('off')
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig


def build_combined_figure(company_name, prediction, skills):
    merged = ARTIFACTS['merged']
    matches = merged[merged['conm'].str.upper() == company_name.upper()].copy()

    if not prediction or prediction.get('error'):
        error_text = prediction.get('error', 'No analysis available for this company.') if prediction else 'No analysis available for this company.'
        return build_empty_plot(error_text, title=company_name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_pie = axes[0, 0]
    ax_line = axes[0, 1]
    ax_skills = axes[1, 0]
    ax_stack = axes[1, 1]

    subfield_profile = prediction.get('subfield_profile', {})
    pie_data = [(k, v) for k, v in subfield_profile.items() if v > 0.05]

    if pie_data:
        pie_labels = [k for k, _ in pie_data]
        pie_values = [v for _, v in pie_data]
        pie_colors = COLOR_PALETTE[:len(pie_data)]
        ax_pie.pie(
            pie_values,
            labels=pie_labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=pie_colors,
            textprops={'fontsize': 9}
        )
    else:
        ax_pie.text(0.5, 0.5, 'No subfield shares above 5%', ha='center', va='center')
        ax_pie.axis('off')
    ax_pie.set_title('Company AI Research Areas')

    if not matches.empty:
        matches = matches.sort_values('fyear')
        years = matches['fyear'].astype(int).tolist()
        patent_volume = matches['patent_volume'].fillna(0).astype(float).tolist()
        ax_line.plot(years, patent_volume, color='steelblue', marker='o', linewidth=2)
        ax_line.fill_between(years, patent_volume, color='steelblue', alpha=0.15)
        ax_line.set_xlabel('Year')
        ax_line.set_ylabel('Number of Patents')
    else:
        ax_line.text(0.5, 0.5, 'No patent timeline available', ha='center', va='center')
    ax_line.set_title('AI Patent Activity Over Time')
    ax_line.grid(axis='y', linestyle='--', alpha=0.25)

    all_skills = skills.get('all_skills', {}) if skills else {}
    ranked_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:5]
    if not ranked_skills and skills and skills.get('top_skills'):
        ranked_skills = [(skill, 0.0) for skill in skills['top_skills'][:5]]

    if ranked_skills:
        skill_labels = [item[0] for item in ranked_skills]
        skill_values = [float(item[1]) for item in ranked_skills]
        bars = ax_skills.barh(skill_labels, skill_values, color='steelblue')
        ax_skills.invert_yaxis()
        for bar, value in zip(bars, skill_values):
            ax_skills.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f'{value:.3f}', va='center', fontsize=9)
    else:
        ax_skills.text(0.5, 0.5, 'No recommended skills available', ha='center', va='center')
        ax_skills.axis('off')
    ax_skills.set_title('Recommended Skills')
    if ranked_skills:
        ax_skills.grid(axis='x', linestyle='--', alpha=0.25)

    top_subfields = [k for k, v in sorted(subfield_profile.items(), key=lambda x: x[1], reverse=True) if v > 0][:4]
    if not matches.empty and top_subfields:
        bottom = [0.0] * len(matches)
        years = matches['fyear'].astype(int).tolist()
        for idx, subfield in enumerate(top_subfields):
            column = SUBFIELD_COLUMN_MAP.get(subfield)
            if column and column in matches.columns:
                values = (matches[column].fillna(0).astype(float) * 100).tolist()
            else:
                values = [0.0] * len(matches)
            ax_stack.bar(years, values, bottom=bottom, label=subfield, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)])
            bottom = [b + v for b, v in zip(bottom, values)]
        ax_stack.set_xlabel('Year')
        ax_stack.set_ylabel('Share (%)')
        ax_stack.legend(loc='upper right', fontsize=8)
    else:
        ax_stack.text(0.5, 0.5, 'No subfield history available', ha='center', va='center')
    ax_stack.set_title('Research Focus Over Time')
    ax_stack.grid(axis='y', linestyle='--', alpha=0.25)

    plt.suptitle(company_name, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def format_skills(skills):
    if not skills:
        return 'No skill recommendations available.'

    top_skills = skills.get('top_skills', [])
    if not top_skills:
        return 'No skill recommendations available.'

    return '\n'.join([f"{idx}. {skill}" for idx, skill in enumerate(top_skills, start=1)])


def format_market_context(prediction):
    if not prediction or prediction.get('error'):
        return (
            'No market context is available because this company was not found in the dataset. '
            'Please choose a company from the list and try again.'
        )

    sector = prediction.get('sector', 'N/A')
    growth_tier = prediction.get('growth_tier', 'N/A')
    ai_intensity = prediction.get('ai_intensity', 'N/A')
    patent_volume = prediction.get('patent_volume', 'N/A')
    year = prediction.get('year', 'N/A')

    return (
        f"{prediction.get('company', 'This company')} is in the {sector} sector. "
        f"The model classifies expected growth as {growth_tier}, with an AI intensity score of {ai_intensity} on a 0 to 1 scale. "
        f"The company has {patent_volume} AI patents in this dataset, and the most recent record is from {year}. "
        "These estimates are based on historical patent behavior and should be used as one signal among many."
    )


def analyze_company(company_name):
    result = run_full_analysis(company_name, ARTIFACTS)

    prediction = result.get('prediction')
    skills = result.get('skills')
    insight = result.get('insight', 'No insight generated.')

    combined_plot = build_combined_figure(company_name, prediction, skills)
    context_text = format_market_context(prediction)

    return combined_plot, context_text, insight


with gr.Blocks(title='Agentic Evolution — AI Career Intelligence') as demo:
    gr.Markdown('# Agentic Evolution — AI Career Intelligence')
    gr.Markdown('Search a company to discover their AI research focus and what skills you need to get hired there.')

    company_dropdown = gr.Dropdown(
        choices=COMPANY_LIST,
        label='Select a Company',
        value=COMPANY_LIST[0] if COMPANY_LIST else None
    )
    submit_button = gr.Button('Analyze')

    with gr.Accordion('How to read this report', open=False):
        gr.Markdown(
        """
        **AI Research Areas**
        Shows the breakdown of this company's AI patent portfolio by research 
        category. A larger slice means the company has filed more patents in 
        that area — which signals where their engineering teams are focused 
        and where they are most likely hiring.

        **AI Patent Activity Over Time**
        Tracks how many AI patents this company filed each year from 2016 to 
        2023. A rising line means accelerating AI investment. A flat or 
        declining line may indicate the company is shifting from research 
        to deployment, or reducing formal IP activity in favor of 
        open-source contributions.

        **Recommended Skills**
        These are the technical skills most aligned with this company's AI 
        research profile, derived by mapping their top patent categories to 
        real job market demand. Skills at the top of the list appear most 
        frequently in job postings from companies with a similar AI focus.

        **Research Focus Over Time**
        Shows how this company's AI priorities have shifted across categories 
        year by year. A growing slice for a category means increasing 
        investment in that area. Use this to identify emerging focus areas 
        that may represent new hiring opportunities.

        ---
        *All patent data sourced from the USPTO Artificial Intelligence 
        Patent Dataset (AIPD) 2023. Predictions are based on a Random 
        Forest model trained on 57 S&P 500 firms from 2016 to 2023. 
        Use as one signal among many when making career decisions.*
        """
    )

    combined_plot_output = gr.Plot(label='Company AI Analysis')

    gr.Markdown('## Market Context')
    context_output = gr.Markdown()

    gr.Markdown('## Career Intelligence Report')
    insight_output = gr.Markdown()

    gr.Examples(
        examples=[
            ['MICROSOFT CORP'],
            ['GOOGLE'],
            ['NVIDIA CORP'],
            ['META PLATFORMS INC'],
            ['AMAZON.COM INC']
        ],
        inputs=company_dropdown
    )

    submit_button.click(
        fn=analyze_company,
        inputs=[company_dropdown],
        outputs=[combined_plot_output, context_output, insight_output],
        show_progress=True
    )


if __name__ == '__main__':
    demo.launch(share=False, server_port=7860)
