"""
Report generation for the exoplanet detection pipeline.
"""

import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


def generate_report(results, model_results, timestamp, output_dir):
    """
    Generate a comprehensive report of the pipeline results.
    
    Args:
        results: List of light curve processing results
        model_results: AI model training results (optional)
        timestamp: Timestamp for the report
        output_dir: Directory to save the report
    
    Returns:
        str: Path to the generated report
    """
    # Generate HTML report
    html_path = generate_html_report(results, model_results, timestamp, output_dir)
    
    # Generate JSON summary
    json_path = generate_json_summary(results, model_results, timestamp, output_dir)
    
    return html_path


def generate_json_summary(results, model_results, timestamp, output_dir):
    """
    Generate a JSON summary of the pipeline results.
    
    Args:
        results: List of light curve processing results
        model_results: AI model training results (optional)
        timestamp: Timestamp for the report
        output_dir: Directory to save the report
    
    Returns:
        str: Path to the generated JSON file
    """
    summary_file = os.path.join(output_dir, f"pipeline_summary_{timestamp}.json")
    
    # Extract successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    # Create summary dictionary
    summary = {
        'timestamp': timestamp,
        'total_files': len(results),
        'successful_processing': len(successful_results),
        'transit_detections': sum(1 for r in successful_results if r.get('transit_count', 0) > 0),
        'periodicity_found': sum(1 for r in successful_results if r.get('periodicity') is not None),
        'planet_properties': sum(1 for r in successful_results if r.get('planet_properties') is not None)
    }
    
    # Add model metrics if available
    if model_results:
        model_summary = {}
        
        if 'cnn_metrics' in model_results:
            # Extract numeric metrics only
            cnn_metrics = {k: float(v) for k, v in model_results['cnn_metrics'].items() 
                          if isinstance(v, (int, float))}
            model_summary['cnn_metrics'] = cnn_metrics
        
        if 'multimodal_metrics' in model_results:
            mm_metrics = {k: float(v) for k, v in model_results['multimodal_metrics'].items() 
                         if isinstance(v, (int, float))}
            model_summary['multimodal_metrics'] = mm_metrics
            
        summary['model_metrics'] = model_summary
    
    # Add result details (cleaning up non-serializable objects)
    clean_results = []
    for result in successful_results:
        clean_result = {
            'file_path': result.get('file_path', ''),
            'transit_count': result.get('transit_count', 0),
            'result_dir': result.get('result_dir', '')
        }
        
        # Add periodicity if available
        if result.get('periodicity') is not None:
            clean_result['periodicity'] = float(result['periodicity'])
        
        # Add planet properties if available
        if result.get('planet_properties') is not None:
            props = result['planet_properties']
            clean_result['planet_properties'] = {
                'radius_earth': float(props.get('radius_earth', 0)),
                'orbital_period_days': float(props.get('orbital_period_days', 0)),
                'semi_major_axis_au': float(props.get('semi_major_axis_au', 0)),
                'equilibrium_temp_k': float(props.get('equilibrium_temp_k', 0))
            }
        
        clean_results.append(clean_result)
    
    summary['results'] = clean_results
    
    # Write to file
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"JSON summary generated: {summary_file}")
    return summary_file


def generate_html_report(results, model_results, timestamp, output_dir):
    """
    Generate an HTML report of the pipeline results.
    
    Args:
        results: List of light curve processing results
        model_results: AI model training results (optional)
        timestamp: Timestamp for the report
        output_dir: Directory to save the report
    
    Returns:
        str: Path to the generated report
    """
    report_file = os.path.join(output_dir, f"pipeline_report_{timestamp}.html")
    
    # Extract successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    with open(report_file, 'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html lang='en'>\n")
        f.write("<head>\n")
        f.write("    <meta charset='UTF-8'>\n")
        f.write("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
        f.write("    <title>Exoplanet Detection Pipeline Report</title>\n")
        f.write("    <style>\n")
        f.write("        body { font-family: Arial, sans-serif; max-width: 1200px; margin: auto; padding: 20px; }\n")
        f.write("        table { width: 100%; border-collapse: collapse; margin: 20px 0; }\n")
        f.write("        th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }\n")
        f.write("        th { background-color: #f2f2f2; }\n")
        f.write("        .metric-card { background-color: #f9f9f9; border-radius: 5px; padding: 15px; margin: 10px; display: inline-block; width: 200px; text-align: center; }\n")
        f.write("        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }\n")
        f.write("        .metric-label { font-size: 14px; color: #666; }\n")
        f.write("        .gallery { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }\n")
        f.write("        .gallery-item { flex: 0 0 300px; }\n")
        f.write("        .gallery-item img { max-width: 100%; border: 1px solid #ddd; }\n")
        f.write("    </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        
        # Header
        f.write(f"    <h1>Exoplanet Detection Pipeline Report</h1>\n")
        f.write(f"    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        
        # Summary metrics
        f.write("    <h2>Summary</h2>\n")
        f.write("    <div style='display: flex; flex-wrap: wrap;'>\n")
        
        # Total files
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>Light Curves Processed</div>\n")
        f.write(f"            <div class='metric-value'>{len(results)}</div>\n")
        f.write("        </div>\n")
        
        # Successful processing
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>Successfully Processed</div>\n")
        f.write(f"            <div class='metric-value'>{len(successful_results)}</div>\n")
        f.write("        </div>\n")
        
        # Transit detections
        transit_count = sum(1 for r in successful_results if r.get('transit_count', 0) > 0)
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>Transit Detections</div>\n")
        f.write(f"            <div class='metric-value'>{transit_count}</div>\n")
        f.write("        </div>\n")
        
        # Periodicity found
        periodicity_count = sum(1 for r in successful_results if r.get('periodicity') is not None)
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>Periodicity Found</div>\n")
        f.write(f"            <div class='metric-value'>{periodicity_count}</div>\n")
        f.write("        </div>\n")
        
        # Planet properties
        planet_count = sum(1 for r in successful_results if r.get('planet_properties') is not None)
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>Planet Properties</div>\n")
        f.write(f"            <div class='metric-value'>{planet_count}</div>\n")
        f.write("        </div>\n")
        
        f.write("    </div>\n")
        
        # Detailed results table
        f.write("    <h2>Detected Transits</h2>\n")
        f.write("    <table>\n")
        f.write("        <thead>\n")
        f.write("            <tr>\n")
        f.write("                <th>File</th>\n")
        f.write("                <th>Transit Count</th>\n")
        f.write("                <th>Period (days)</th>\n")
        f.write("                <th>Planet Radius (Earth)</th>\n")
        f.write("                <th>Semi-major Axis (AU)</th>\n")
        f.write("                <th>Equilibrium Temp (K)</th>\n")
        f.write("            </tr>\n")
        f.write("        </thead>\n")
        f.write("        <tbody>\n")
        
        # Only include results with transits
        transit_results = [r for r in successful_results if r.get('transit_count', 0) > 0]
        for result in transit_results:
            file_name = os.path.basename(result.get('file_path', ''))
            transit_count = result.get('transit_count', 0)
            period = result.get('periodicity', 'N/A')
            if period != 'N/A':
                period = f"{period:.2f}" if period is not None else "N/A"
            
            # Planet properties
            radius = "N/A"
            semi_major = "N/A"
            temp = "N/A"
            
            if result.get('planet_properties'):
                props = result['planet_properties']
                radius = f"{props.get('radius_earth', 0):.2f}" if 'radius_earth' in props else "N/A"
                semi_major = f"{props.get('semi_major_axis_au', 0):.3f}" if 'semi_major_axis_au' in props else "N/A"
                temp = f"{props.get('equilibrium_temp_k', 0):.0f}" if 'equilibrium_temp_k' in props else "N/A"
            
            f.write("            <tr>\n")
            f.write(f"                <td>{file_name}</td>\n")
            f.write(f"                <td>{transit_count}</td>\n")
            f.write(f"                <td>{period}</td>\n")
            f.write(f"                <td>{radius}</td>\n")
            f.write(f"                <td>{semi_major}</td>\n")
            f.write(f"                <td>{temp}</td>\n")
            f.write("            </tr>\n")
        
        f.write("        </tbody>\n")
        f.write("    </table>\n")
        
        # Image gallery
        f.write("    <h2>Visualization Gallery</h2>\n")
        f.write("    <div class='gallery'>\n")
        
        # Show a selection of the best light curves (up to 10)
        best_results = sorted(
            [r for r in successful_results if r.get('transit_count', 0) > 0],
            key=lambda r: r.get('transit_count', 0),
            reverse=True
        )[:10]
        
        for result in best_results:
            light_curve_plot_path = result.get('light_curve_plot_path', '')
            if light_curve_plot_path:
                if os.path.exists(light_curve_plot_path):
                    rel_path = os.path.relpath(light_curve_plot_path, output_dir)
                    file_name = os.path.basename(result.get('file_path', ''))
                    f.write(f"        <div class='gallery-item'><img src='{rel_path}' alt='{file_name}'><p>{file_name}</p></div>\n")
        
        f.write("    </div>\n")
        
        # Model results
        if model_results:
            f.write("    <h2>AI Model Performance</h2>\n")
            
            # CNN model
            if 'cnn_metrics' in model_results:
                f.write("    <h3>Transit Detection CNN</h3>\n")
                f.write("    <div style='display: flex; flex-wrap: wrap;'>\n")
                
                metrics = model_results['cnn_metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        f.write("        <div class='metric-card'>\n")
                        f.write(f"            <div class='metric-label'>{metric_name.replace('_', ' ').title()}</div>\n")
                        f.write(f"            <div class='metric-value'>{metric_value:.4f}</div>\n")
                        f.write("        </div>\n")
                
                f.write("    </div>\n")
                
                # Add model images
                f.write("    <div class='gallery'>\n")
                roc_path = os.path.join(config.MODEL_DIR, 'transit_detector_roc_curve.png')
                if os.path.exists(roc_path):
                    rel_path = os.path.relpath(roc_path, output_dir)
                    f.write(f"        <div class='gallery-item'><img src='{rel_path}' alt='ROC Curve'></div>\n")
                
                pr_path = os.path.join(config.MODEL_DIR, 'transit_detector_precision_recall_curve.png')
                if os.path.exists(pr_path):
                    rel_path = os.path.relpath(pr_path, output_dir)
                    f.write(f"        <div class='gallery-item'><img src='{rel_path}' alt='Precision-Recall Curve'></div>\n")
                
                f.write("    </div>\n")
                
            # Multimodal model
            if 'multimodal_metrics' in model_results:
                f.write("    <h3>Multimodal Fusion Model</h3>\n")
                f.write("    <div style='display: flex; flex-wrap: wrap;'>\n")
                
                metrics = model_results['multimodal_metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        f.write("        <div class='metric-card'>\n")
                        f.write(f"            <div class='metric-label'>{metric_name.replace('_', ' ').title()}</div>\n")
                        f.write(f"            <div class='metric-value'>{metric_value:.4f}</div>\n")
                        f.write("        </div>\n")
                
                f.write("    </div>\n")
                
                # Add model images
                f.write("    <div class='gallery'>\n")
                roc_path = os.path.join(config.MODEL_DIR, 'multimodal_classifier_roc_curve.png')
                if os.path.exists(roc_path):
                    rel_path = os.path.relpath(roc_path, output_dir)
                    f.write(f"        <div class='gallery-item'><img src='{rel_path}' alt='ROC Curve'></div>\n")
                
                pr_path = os.path.join(config.MODEL_DIR, 'multimodal_classifier_precision_recall_curve.png')
                if os.path.exists(pr_path):
                    rel_path = os.path.relpath(pr_path, output_dir)
                    f.write(f"        <div class='gallery-item'><img src='{rel_path}' alt='Precision-Recall Curve'></div>\n")
                
                f.write("    </div>\n")
                
            # Model comparison
            comp_path = os.path.join(config.MODEL_DIR, 'model_comparison.png')
            if os.path.exists(comp_path):
                f.write("    <h3>Model Comparison</h3>\n")
                f.write("    <div class='gallery'>\n")
                rel_path = os.path.relpath(comp_path, output_dir)
                f.write(f"        <div class='gallery-item'><img src='{rel_path}' alt='Model Comparison'></div>\n")
                f.write("    </div>\n")