"""
Analysis of exoplanet detection pipeline results.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

import config
from utils.plotting import visualize_detection_results

logger = logging.getLogger(__name__)


def analyze_pipeline_results(results, output_dir=None):
    """
    Analyze results from the exoplanet detection pipeline.
    
    Args:
        results: List of processing results from pipeline
        output_dir: Directory to save analysis outputs (optional)
    
    Returns:
        dict: Analysis statistics and insights
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        logger.warning("No successful results to analyze")
        return {"error": "No successful results to analyze"}
    
    # Extract key metrics
    transit_counts = [r.get('transit_count', 0) for r in successful_results]
    periods = [r.get('periodicity') for r in successful_results if r.get('periodicity') is not None]
    
    # Calculate basic statistics
    stats_summary = {
        'total_processed': len(results),
        'successful_processing': len(successful_results),
        'success_rate': len(successful_results) / len(results) if results else 0,
        'transit_detection_rate': sum(1 for c in transit_counts if c > 0) / len(transit_counts) if transit_counts else 0,
        'mean_transits_per_curve': np.mean(transit_counts) if transit_counts else 0,
        'median_transits_per_curve': np.median(transit_counts) if transit_counts else 0,
        'max_transits': max(transit_counts) if transit_counts else 0,
        'period_detection_rate': len(periods) / sum(1 for c in transit_counts if c > 0) if transit_counts and sum(1 for c in transit_counts if c > 0) > 0 else 0,
        'mean_period': np.mean(periods) if periods else 0,
        'median_period': np.median(periods) if periods else 0
    }
    
    # Extract planet properties where available
    planet_properties = []
    for result in successful_results:
        if result.get('planet_properties') is not None:
            props = result.get('planet_properties')
            planet_properties.append({
                'file': os.path.basename(result.get('file_path', '')),
                'transit_count': result.get('transit_count', 0),
                'period': result.get('periodicity'),
                'radius_earth': props.get('radius_earth'),
                'semi_major_axis_au': props.get('semi_major_axis_au'),
                'equilibrium_temp_k': props.get('equilibrium_temp_k')
            })
    
    # Create a DataFrame for planet properties
    planet_df = pd.DataFrame(planet_properties) if planet_properties else None
    
    # Analyze planet size distribution
    size_distribution = None
    if planet_df is not None and not planet_df.empty and 'radius_earth' in planet_df.columns:
        # Categorize planets by size
        def size_category(radius):
            if radius < 1.25:
                return 'Earth-sized'
            elif radius < 2.0:
                return 'Super-Earth'
            elif radius < 4.0:
                return 'Neptune-sized'
            elif radius < 11.0:
                return 'Jupiter-sized'
            else:
                return 'Super-Jupiter'
        
        planet_df['size_category'] = planet_df['radius_earth'].apply(size_category)
        size_distribution = planet_df['size_category'].value_counts().to_dict()
        
        # Generate planet size distribution plot
        plt.figure(figsize=(10, 6))
        planet_df['size_category'].value_counts().plot(kind='bar')
        plt.title('Planet Size Distribution')
        plt.xlabel('Size Category')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'planet_size_distribution.png'))
        plt.close()
    
    # Analyze period distribution
    period_distribution = None
    if planet_df is not None and not planet_df.empty and 'period' in planet_df.columns:
        # Create period bins
        bins = [0, 3, 10, 30, 100, 365, 1000]
        labels = ['<3 days', '3-10 days', '10-30 days', '30-100 days', '100-365 days', '>365 days']
        planet_df['period_category'] = pd.cut(planet_df['period'], bins=bins, labels=labels)
        period_distribution = planet_df['period_category'].value_counts().to_dict()
        
        # Generate period distribution plot
        plt.figure(figsize=(10, 6))
        planet_df['period_category'].value_counts().plot(kind='bar')
        plt.title('Orbital Period Distribution')
        plt.xlabel('Period Range')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'period_distribution.png'))
        plt.close()
    
    # Analyze habitable zone potential
    habitable_candidates = None
    if planet_df is not None and not planet_df.empty:
        if all(col in planet_df.columns for col in ['equilibrium_temp_k', 'radius_earth']):
            # Filter for potentially habitable planets
            # (Equilibrium temperature between 200-350K and radius < 2 Earth radii)
            habitable_mask = (
                (planet_df['equilibrium_temp_k'] >= 200) & 
                (planet_df['equilibrium_temp_k'] <= 350) & 
                (planet_df['radius_earth'] < 2.0)
            )
            habitable_candidates = planet_df[habitable_mask].to_dict('records')
    
    # Analyze transit model quality
    model_quality = []
    for result in successful_results:
        if result.get('transit_model') is not None:
            model = result.get('transit_model')
            if isinstance(model, dict) and 'fit_quality' in model:
                model_quality.append({
                    'file': os.path.basename(result.get('file_path', '')),
                    'fit_quality': model['fit_quality'],
                    'residual_std': model.get('residual_std')
                })
    
    # Save comprehensive visualization of results
    visualize_detection_results(
        successful_results, 
        title="Pipeline Results Summary",
        filename=os.path.join(output_dir, 'results_summary.png')
    )
    
    # Create summary DataFrame and save to CSV
    summary_df = pd.DataFrame([stats_summary])
    summary_df.to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)
    
    if planet_df is not None and not planet_df.empty:
        planet_df.to_csv(os.path.join(output_dir, 'planet_properties.csv'), index=False)
    
    # Return comprehensive analysis results
    analysis_results = {
        'stats_summary': stats_summary,
        'size_distribution': size_distribution,
        'period_distribution': period_distribution,
        'habitable_candidates': habitable_candidates,
        'model_quality': model_quality
    }
    
    logger.info(f"Result analysis completed and saved to {output_dir}")
    return analysis_results


def analyze_model_performance(model_results, test_data=None, output_dir=None):
    """
    Analyze performance of the trained AI models.
    
    Args:
        model_results: Dictionary containing model evaluation results
        test_data: Optional test dataset for additional analysis
        output_dir: Directory to save analysis outputs (optional)
    
    Returns:
        dict: Model performance analysis
    """
    if output_dir is None:
        output_dir = config.MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if not model_results:
        logger.warning("No model results to analyze")
        return {"error": "No model results to analyze"}
    
    # Extract metrics
    performance = {}
    
    # CNN model analysis
    if 'cnn_metrics' in model_results:
        cnn_metrics = model_results['cnn_metrics']
        performance['cnn'] = {
            key: value for key, value in cnn_metrics.items()
            if isinstance(value, (int, float))
        }
    
    # Multimodal model analysis
    if 'multimodal_metrics' in model_results:
        mm_metrics = model_results['multimodal_metrics']
        performance['multimodal'] = {
            key: value for key, value in mm_metrics.items()
            if isinstance(value, (int, float))
        }
    
    # Compare models if both are available
    if 'cnn' in performance and 'multimodal' in performance:
        # Calculate improvement percentages
        common_metrics = set(performance['cnn'].keys()) & set(performance['multimodal'].keys())
        improvements = {}
        
        for metric in common_metrics:
            cnn_value = performance['cnn'][metric]
            mm_value = performance['multimodal'][metric]
            if cnn_value != 0:  # Avoid division by zero
                pct_change = ((mm_value - cnn_value) / cnn_value) * 100
                improvements[metric] = pct_change
        
        performance['improvements'] = improvements
        
        # Create improvement visualization
        plt.figure(figsize=(12, 6))
        metrics = list(improvements.keys())
        values = list(improvements.values())
        
        # Color bars based on improvement (green) or decline (red)
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        plt.bar(metrics, values, color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Multimodal Model Improvement Over CNN (%)')
        plt.ylabel('Percentage Change')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_improvement.png'))
        plt.close()
    
    # Analyze false positives and false negatives if test data is available
    error_analysis = {}
    if test_data and 'cnn_model' in model_results:
        # Extract model and test data
        cnn_model = model_results.get('cnn_model')
        X_test, y_test = test_data.get('X_test'), test_data.get('y_test')
        
        if cnn_model is not None and X_test is not None and y_test is not None:
            # Get predictions
            y_pred = (cnn_model.predict(X_test) > 0.5).astype(int).flatten()
            
            # Identify false positives and false negatives
            false_positives = np.where((y_pred == 1) & (y_test == 0))[0]
            false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]
            
            error_analysis['false_positive_count'] = len(false_positives)
            error_analysis['false_negative_count'] = len(false_negatives)
            error_analysis['false_positive_rate'] = len(false_positives) / sum(y_test == 0) if sum(y_test == 0) > 0 else 0
            error_analysis['false_negative_rate'] = len(false_negatives) / sum(y_test == 1) if sum(y_test == 1) > 0 else 0
    
    # Save summary to CSV
    performance_df = pd.DataFrame({
        'Model': list(performance.keys()),
        'Metrics': [str(metrics) for metrics in performance.values()]
    })
    performance_df.to_csv(os.path.join(output_dir, 'model_performance.csv'), index=False)
    
    # Return comprehensive performance analysis
    analysis_results = {
        'performance': performance,
        'error_analysis': error_analysis
    }
    
    logger.info(f"Model performance analysis completed and saved to {output_dir}")
    return analysis_results


def compare_with_ground_truth(pipeline_results, ground_truth_data, output_dir=None):
    """
    Compare pipeline detection results with ground truth data.
    
    Args:
        pipeline_results: Results from the exoplanet detection pipeline
        ground_truth_data: Known exoplanet data for comparison
        output_dir: Directory to save analysis outputs (optional)
    
    Returns:
        dict: Comparison metrics and analysis
    """
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if not pipeline_results or not ground_truth_data:
        logger.warning("Missing data for ground truth comparison")
        return {"error": "Missing data for comparison"}
    
    # Extract successful detections
    detections = [r for r in pipeline_results if r.get('success', False) and r.get('transit_count', 0) > 0]
    
    # Extract stellar names from file paths
    # This assumes file names contain the stellar name
    # Modify the extraction logic based on your actual file naming convention
    detection_stars = []
    for result in detections:
        file_path = result.get('file_path', '')
        file_name = os.path.basename(file_path)
        # Example: extract stellar name from file name (adjust as needed)
        stellar_name = file_name.split('_')[0]
        detection_stars.append(stellar_name)
    
    # Get known host stars from ground truth
    if isinstance(ground_truth_data, pd.DataFrame):
        known_hosts = set(ground_truth_data['hostname'].unique())
    else:
        # Assuming ground_truth_data has a 'hostname' field
        known_hosts = set(item.get('hostname') for item in ground_truth_data if 'hostname' in item)
    
    # Calculate detection metrics
    detected_hosts = set(detection_stars)
    true_positives = detected_hosts.intersection(known_hosts)
    false_positives = detected_hosts - known_hosts
    false_negatives = known_hosts - detected_hosts
    
    # Calculate precision, recall, and F1 score
    precision = len(true_positives) / len(detected_hosts) if detected_hosts else 0
    recall = len(true_positives) / len(known_hosts) if known_hosts else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compare planet properties for true positives
    property_comparisons = []
    if isinstance(ground_truth_data, pd.DataFrame) and 'pl_orbper' in ground_truth_data.columns:
        for star in true_positives:
            # Find the pipeline result for this star
            pipeline_result = next((r for r in detections if os.path.basename(r.get('file_path', '')).split('_')[0] == star), None)
            
            if pipeline_result and pipeline_result.get('planet_properties') and pipeline_result.get('periodicity'):
                # Get ground truth properties
                ground_truth = ground_truth_data[ground_truth_data['hostname'] == star]
                
                if not ground_truth.empty:
                    # Get the first planet if multiple exist
                    gt_period = ground_truth.iloc[0]['pl_orbper']
                    gt_radius = ground_truth.iloc[0]['pl_rade'] if 'pl_rade' in ground_truth.columns else None
                    
                    # Get detected properties
                    detected_period = pipeline_result.get('periodicity')
                    detected_props = pipeline_result.get('planet_properties', {})
                    detected_radius = detected_props.get('radius_earth') if detected_props else None
                    
                    # Calculate errors
                    period_error_pct = abs(detected_period - gt_period) / gt_period * 100 if gt_period else None
                    radius_error_pct = abs(detected_radius - gt_radius) / gt_radius * 100 if gt_radius and detected_radius else None
                    
                    property_comparisons.append({
                        'star': star,
                        'gt_period': gt_period,
                        'detected_period': detected_period,
                        'period_error_pct': period_error_pct,
                        'gt_radius': gt_radius,
                        'detected_radius': detected_radius,
                        'radius_error_pct': radius_error_pct
                    })
    
    # Create comparison DataFrame and save to CSV
    if property_comparisons:
        comparison_df = pd.DataFrame(property_comparisons)
        comparison_df.to_csv(os.path.join(output_dir, 'property_comparison.csv'), index=False)
        
        # Calculate average errors
        avg_period_error = comparison_df['period_error_pct'].mean() if 'period_error_pct' in comparison_df else None
        avg_radius_error = comparison_df['radius_error_pct'].mean() if 'radius_error_pct' in comparison_df else None
    else:
        avg_period_error = None
        avg_radius_error = None
    
    # Plot comparison results
    plt.figure(figsize=(10, 6))
    categories = ['True Positives', 'False Positives', 'False Negatives']
    values = [len(true_positives), len(false_positives), len(false_negatives)]
    
    plt.bar(categories, values, color=['green', 'red', 'orange'])
    plt.title('Detection Performance')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_comparison.png'))
    plt.close()
    
    # Return comparison results
    comparison_results = {
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_period_error_pct': avg_period_error,
        'avg_radius_error_pct': avg_radius_error,
        'property_comparisons': property_comparisons
    }
    
    logger.info(f"Ground truth comparison completed and saved to {output_dir}")
    return comparison_results
