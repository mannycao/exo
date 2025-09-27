import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import textwrap

import config

logger = logging.getLogger(__name__)

def generate_report(results, model_results, y_true, y_pred, timestamp, output_dir):
    """
    Generates an HTML report and a JSON summary of the pipeline results,
    including model metrics and classification of individual results.
    """
    report_filename_html = os.path.join(output_dir, f"pipeline_report_{timestamp}.html")
    summary_filename_json = os.path.join(output_dir, f"pipeline_summary_{timestamp}.json")

    # Initialize counts for classification
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Add classification to each result and calculate TP, FP, TN, FN
    # Assuming results, y_true, and y_pred are aligned by index
    for i, result in enumerate(results):
        true_label = y_true[i]
        predicted_label = y_pred[i]

        if true_label == 1 and predicted_label == 1:
            result['classification'] = 'True Positive'
            tp += 1
        elif true_label == 0 and predicted_label == 1:
            result['classification'] = 'False Positive'
            fp += 1
        elif true_label == 0 and predicted_label == 0:
            result['classification'] = 'True Negative'
            tn += 1
        elif true_label == 1 and predicted_label == 0:
            result['classification'] = 'False Negative'
            fn += 1
        else:
            result['classification'] = 'Unknown' # Should not happen with binary classification

    # Update model_results with classification counts
    model_results['classification_counts'] = {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

    # Prepare data for JSON summary
    json_summary = {
        "timestamp": timestamp,
        "total_files": len(results),
        "successful_processing": len(results), # Assuming all passed results were successful
        "transit_detections": len(results), # Assuming all passed results had a detection
        "periodicity_found": len(results), # Assuming all passed results had periodicity found
        "planet_properties": len(results), # Assuming all passed results had planet properties
        "model_metrics": model_results['cnn_metrics'],
        "classification_counts": model_results['classification_counts'],
        "results": results # Now includes classification
    }

    # Write JSON summary
    with open(summary_filename_json, 'w') as f:
        json.dump(json_summary, f, indent=2)

    # Extract CACL explanations if available
    cacl_explanations = model_results.get('cacl_explanations', {})

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>Exoplanet Detection Pipeline Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric-card {{ background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin: 10px; display: inline-block; width: 200px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .gallery {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }}
            .gallery-item {{ flex: 0 0 300px; }}
            .gallery-item img {{ max-width: 100%; border: 1px solid #ddd; }}
            .cacl-explanation {{ background-color: #e6f7ff; border-left: 5px solid #3399ff; padding: 10px; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>Exoplanet Detection Pipeline Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h2>Summary</h2>
        <div style='display: flex; flex-wrap: wrap;'>
            <div class='metric-card'>
                <div class='metric-label'>Light Curves Processed</div>
                <div class='metric-value'>{len(results)}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>True Positives</div>
                <div class='metric-value'>{tp}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>False Positives</div>
                <div class='metric-value'>{fp}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>True Negatives</div>
                <div class='metric-value'>{tn}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>False Negatives</div>
                <div class='metric-value'>{fn}</div>
            </div>
        </div>
    """

    # Re-add Model Metrics section
    if model_results:
        html_content += """
        <h2>AI Model Performance</h2>
        <h3>Multimodal Fusion Model Metrics</h3>
        <div style='display: flex; flex-wrap: wrap;'>
        """
        metrics = model_results['cnn_metrics'] # Assuming 'cnn_metrics' holds the relevant model performance metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                html_content += f"""
            <div class='metric-card'>
                <div class='metric-label'>{metric_name.replace('_', ' ').title()}</div>
                <div class='metric-value'>{metric_value:.4f}</div>
            </div>
                """
        html_content += """
        </div>
        """

        # Add CACL Explanations section
        if cacl_explanations:
            html_content += """
            <h2>CACL Explanations</h2>
            """
            for record_id, explanation_data in cacl_explanations.items():
                html_content += f"""
                <div class='cacl-explanation'>
                    <h3>Record ID: {record_id}</h3>
                    <p><strong>Max Disagreement:</strong> {explanation_data.get('max_disagreement', 'N/A'):.4f}</p>
                    <p><strong>Most Conflicting Partitions:</strong> {explanation_data.get('most_conflicting_partitions', 'N/A')}</p>
                    <p><strong>Violated Dependencies:</strong> {explanation_data.get('violated_dependencies', 'N/A')}</p>
                    <p><strong>Context Similarity:</strong> {explanation_data.get('context_similarity', 'N/A'):.4f}</p>
                    <p><strong>Context Flag:</strong> {explanation_data.get('context_flag', 'N/A')}</p>
                </div>
                """

        # Re-add Learning Curve Visualization
        learning_curve_path = os.path.join(output_dir, f"exo_multimodal_model_learning_curves.png")
        if os.path.exists(learning_curve_path):
            # The path needs to be relative to the HTML file for display
            # Since output_dir is the base for both, we can use the filename directly
            html_content += f"""
        <h2>Learning Curves</h2>
        <div class='gallery'>
            <div class='gallery-item'><img src='{os.path.basename(learning_curve_path)}' alt='Learning Curves'></div>
        </div>
            """

    html_content += """
        <h2>Detected Transits</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Classification</th>
                    <th>Period (days)</th>
                    <th>Planet Radius (Earth)</th>
                    <th>Semi-major Axis (AU)</th>
                    <th>Equilibrium Temp (K)</th>
                </tr>
            </thead>
            <tbody>
    """

    for result in results:
        file_path = result.get('file_path', 'N/A')
        classification = result.get('classification', 'N/A')
        periodicity = result.get('periodicity', 'N/A')
        planet_properties = result.get('planet_properties', {})
        
        radius_earth_val = planet_properties.get('radius_earth', 'N/A')
        radius_earth = round(radius_earth_val, 2) if isinstance(radius_earth_val, (int, float)) else radius_earth_val

        orbital_period_days_val = planet_properties.get('orbital_period_days', 'N/A')
        orbital_period_days = round(orbital_period_days_val, 2) if isinstance(orbital_period_days_val, (int, float)) else orbital_period_days_val

        semi_major_axis_au_val = planet_properties.get('semi_major_axis_au', 'N/A')
        semi_major_axis_au = round(semi_major_axis_au_val, 3) if isinstance(semi_major_axis_au_val, (int, float)) else semi_major_axis_au_val

        equilibrium_temp_k_val = planet_properties.get('equilibrium_temp_k', 'N/A')
        equilibrium_temp_k = round(equilibrium_temp_k_val) if isinstance(equilibrium_temp_k_val, (int, float)) else equilibrium_temp_k_val

        html_content += f"""
                <tr>
                    <td>{file_path.split('/')[-1]}</td>
                    <td>{classification}</td>
                    <td>{periodicity:.2f}</td>
                    <td>{radius_earth}</td>
                    <td>{semi_major_axis_au}</td>
                    <td>{equilibrium_temp_k}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    with open(report_filename_html, 'w') as f:
        f.write(html_content)

    logger.info(f"Report generated: {report_filename_html}")
    logger.info(f"Summary generated: {summary_filename_json}")

    return report_filename_html