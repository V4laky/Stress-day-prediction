import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import json

def df_to_table(dataframe, index_name):

    df = dataframe.reset_index()
    df.rename(columns={'index': index_name})
    table_data = [df.columns.tolist()] + df.values.tolist()

    return Table(table_data)
    


def make_pdf_report(metrics_df, model_params_df, folds_df, indexes, results_path, filename="report.pdf"):

    doc = SimpleDocTemplate(f'{results_path}/{filename}', pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    from reportlab.platypus import TableStyle
    from reportlab.lib import colors

    tab_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),      # Header background
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),             # Center everything
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),   # Bold header
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),             # Header padding
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),    # Body background
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),     # Border grid
        ])


    # ---- Page 1: Metrics ----
    story.append(Paragraph("<b>Model Evaluation Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    for index in indexes:
        table = df_to_table(metrics_df.xs(index, axis=0).round(4), 'Model')
        table.setStyle(tab_style)

        story.append(Paragraph(f'<b>Model metrics for {index}</b>', styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(table)
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ---- Page 2: parameters and folds ----
    table = df_to_table(folds_df.round(4), 'Fold')
    table.setStyle(tab_style)

    story.append(Paragraph(f'<b>Result in training folds</b>', styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(table)
    story.append(Spacer(1, 6))

    table = df_to_table(model_params_df, 'Parameter')
    table.setStyle(tab_style)

    story.append(Paragraph(f'<b>Model parameters</b>', styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(table)
    story.append(PageBreak())

    # ---- Page 3: PR and ROC curves ----
    for index in indexes:
        img = Image(results_path / f'figures/curves_{index}.png', width=450, height=200)
        story.append(img)
    
    story.append(PageBreak())

    # ---- Page 4+: Permutation Importance ----
    for model in model_params_df.columns:
        story.append(Paragraph(f'<b>Metrics of {model} on indexes</b>', styles['Normal']))    
        story.append(Spacer(1, 6))

        table = df_to_table(metrics_df.xs(model, level='Model').round(4), 'Index')
        table.setStyle(tab_style)
        story.append(table)
        
        story.append(Spacer(1, 12))

        img = Image(results_path / f'figures/perm_imp_{model}.png', width=450, height=400)
        story.append(img)
        story.append(PageBreak())

    # ---- Feature Importances ----
    img = Image(results_path / 'figures/abs_imp_.png', width=450, height=600)
    story.append(img)

    doc.build(story)