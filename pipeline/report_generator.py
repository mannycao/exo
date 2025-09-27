import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

logger = logging.getLogger(__name__)

def create_spider_chart(disagreement_matrix, record_id, output_dir):
    labels = [f'P{i+1}' for i in range(len(disagreement_matrix))]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot disagreement values for each partition
    for i in range(num_vars):
        values = disagreement_matrix[i].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Partition {i+1}')

    # Add a circle for the mean disagreement
    mean_disagreement = np.mean(disagreement_matrix)
    ax.plot(angles, [mean_disagreement] * (num_vars + 1), color='r', linestyle='--', label='Mean Disagreement')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'CACL Disagreement for Record {record_id}')
    
    chart_filename = f'cacl_spider_{record_id}.png'
    chart_filepath = os.path.join(output_dir, chart_filename)
    plt.savefig(chart_filepath)
    plt.close(fig)
    return chart_filename

def create_disagreement_similarity_plot(cacl_explanations, output_dir):
    record_ids = []
    max_disagreements = []
    context_similarities = []
    classifications = []

    for record_id, exp_data in cacl_explanations.items():
        record_ids.append(record_id)
        max_disagreements.append(exp_data['max_disagreement'])
        context_similarities.append(exp_data['context_similarity'])
        classifications.append(exp_data['classification'])

    df = pd.DataFrame({
        'Record ID': record_ids,
        'Max Disagreement': max_disagreements,
        'Context Similarity': context_similarities,
        'Classification': classifications
    })

    plt.figure(figsize=(10, 6))
    # Define a color map for classifications
    color_map = {
        'True Positive': 'green',
        'False Positive': 'red',
        'True Negative': 'blue',
        'False Negative': 'orange',
        'Unknown': 'gray'
    }
    for name, group in df.groupby('Classification'):
        plt.scatter(group['Max Disagreement'], group['Context Similarity'], label=name, color=color_map.get(name, 'black'))
    plt.xlabel('Max Disagreement')
    plt.ylabel('Context Similarity')
    plt.title('Max Disagreement vs. Context Similarity by Classification')
    plt.legend()
    plt.grid(True)
    plot_filename = 'disagreement_similarity_plot.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    return plot_filename

def generate_report(results, model_results, y_true, y_pred, timestamp, output_dir, validation_original_indices, validation_classifications):
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

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred).astype(int)

    # Add classification to each result and calculate TP, FP, TN, FN
    # Assuming results, y_true, and y_pred are aligned by index
    for i, result in enumerate(results):
        true_label = y_true[i]
        predicted_label = y_pred_binary[i]

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
        "model_metrics": model_results.get('cnn_metrics', {}),
        "classification_counts": model_results['classification_counts'],
        "results": results # Now includes classification
    }

    # Write JSON summary
    with open(summary_filename_json, 'w') as f:
        json.dump(json_summary, f, indent=2)

    # Extract CACL explanations if available
    cacl_explanations = model_results.get('cacl_explanations', {})

    # Filter results to only include those with CACL explanations
    filtered_results = [res for res in results if res['original_index'] in cacl_explanations]

    # Generate CACL plots
    disagreement_similarity_plot_filename = None
    if cacl_explanations:
        disagreement_similarity_plot_filename = create_disagreement_similarity_plot(cacl_explanations, output_dir)

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
            .cacl-explanation {{ background-color: #e6f7ff; border-left: 5px solid #3399ff; padding: 10px; margin-bottom: 10px; display: flex; align-items: center; }}
            .cacl-text {{ flex: 1; }}
            .cacl-chart {{ flex: 1; }}
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
    if model_results and 'cnn_metrics' in model_results:
        html_content += """
        <h2>AI Model Performance</h2>
        <h3>Multimodal Fusion Model Metrics</h3>
        <div style='display: flex; flex-wrap: wrap;'>
        """
        metrics = model_results['cnn_metrics']
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
            <h2>How to Interpret CACL Explanations</h2>
            <div class='cacl-explanation'>
                <p><strong>What is a Partition?</strong> For a time series like a light curve, a "partition" is simply a segment of the time series. For example, if we have a light curve with 2048 data points, we can split it into 4 non-overlapping partitions of 512 data points each. Each partition is a "view" of the light curve from a different time window, where P1 represents the first segment of the observation, P2 the second, and so on.</p>
                <p><strong>Analogy:</strong> Imagine you have a picture of a cat, and you show it to four different people (the "partitions"). If they all agree it's a cat, that's high agreement. If one person says it's a dog, another says it's a car, and so on, that's high disagreement, and something is likely "weird" or anomalous about the picture. CACL does something similar with the light curve data. It splits the data into different "views" (partitions) and checks if the model's interpretation of these views is consistent.</p>
                <p>The <strong>Record ID</strong> corresponds to the index of the sample in the validation set.</p>
                <p>The <strong>Spider Chart</strong> visualizes the disagreement matrix. Each axis represents a partition of the light curve, and the lines show the disagreement of each partition with the others. The red dotted line shows the mean disagreement for the sample.</p>
                <ul>
                    <li><strong>Good (Normal):</strong> A small, centrally located, and mostly regular shape. This means all parts of the light curve are in agreement. The disagreement values are low and close to the mean.</li>
                    <li><strong>Bad (Anomaly/Warning):</strong> A large, irregular, and skewed shape. This means some parts of the light curve are "confusing" the model, suggesting a potential anomaly or an unusual signal. The disagreement values are high and far from the mean.</li>
                </ul>
                <p><strong>Max Disagreement:</strong> The highest level of confusion between any two parts of the light curve. Higher values are a stronger warning sign.</p>
                <p><strong>Most Conflicting Partitions:</strong> The two parts of the light curve that are causing the most confusion.</p>
                <p><strong>Violated Dependencies:</strong> This indicates a breakdown in the expected relationships between different parts of the light curve. Think of it as a 'rule' being broken, which is a strong indicator of an anomaly.</p>
                <p><strong>Context Similarity:</strong> How 'normal' this light curve looks compared to its neighbors. A low score means it's an outlier.</p>
                <p><strong>Context Flag:</strong> If this is 'True', the light curve is considered an outlier compared to its neighbors.</p>
            </div>
            """
            if disagreement_similarity_plot_filename:
                html_content += f"""
                <h2>Max Disagreement vs. Context Similarity Plot</h2>
                <img src='{disagreement_similarity_plot_filename}' alt='Max Disagreement vs. Context Similarity Plot'>
                """

            for record_id, explanation_data in cacl_explanations.items():
                chart_filename = create_spider_chart(np.array(explanation_data['disagreement_matrix']), record_id, output_dir)
                # Find the original filename from the results
                original_filename = "Unknown"
                if int(record_id) < len(results):
                    original_filename = results[int(record_id)].get('file_path', 'Unknown').split('/')[-1]

                html_content += f"""
                <div class='cacl-explanation'>
                    <div class='cacl-text'>
                        <h3>Record ID: {record_id} (File: {original_filename})</h3>
                        <p><strong>Max Disagreement:</strong> {explanation_data.get('max_disagreement', 'N/A'):.4f}</p>
                        <p><strong>Most Conflicting Partitions:</strong> {explanation_data.get('most_conflicting_partitions', 'N/A')}</p>
                        <p><strong>Violated Dependencies:</strong> {explanation_data.get('violated_dependencies', 'N/A')}</p>
                        <p><strong>Context Similarity:</strong> {explanation_data.get('context_similarity', 'N/A'):.4f}</p>
                        <p><strong>Context Flag:</strong> {explanation_data.get('context_flag', 'N/A')}</p>
                        <p><strong>Max Disagreement (Mean):</strong> {model_results.get('cacl_stats', {}).get('max_disagreement_mean', 'N/A'):.4f} (Range: {model_results.get('cacl_stats', {}).get('max_disagreement_min', 'N/A'):.4f} - {model_results.get('cacl_stats', {}).get('max_disagreement_max', 'N/A'):.4f})</p>
                        <p><strong>Context Similarity (Mean):</strong> {model_results.get('cacl_stats', {}).get('context_similarity_mean', 'N/A'):.4f} (Range: {model_results.get('cacl_stats', {}).get('context_similarity_min', 'N/A'):.4f} - {model_results.get('cacl_stats', {}).get('context_similarity_max', 'N/A'):.4f})</p>
                    </div>
                    <div class='cacl-chart'>
                        <img src='{chart_filename}' alt='CACL Disagreement Matrix'>
                    </div>
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
                    <th>ID</th>
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
                    <td>{result.get('original_index', i)}</td>
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