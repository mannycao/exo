"""
Statistical analysis tools for periodicity detection results.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def convert_numpy_int_to_int(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_int_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_int_to_int(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def create_period_distribution_analysis(results: List[Dict]) -> Dict:
    """
    Analyze the distribution of detected periods and their confidence scores.
    
    Args:
        results: List of periodicity analysis results
        
    Returns:
        Dictionary containing statistical analysis results
    """
    # Extract periods and confidence scores
    periods = []
    confidences = []
    valid_flags = []
    n_transits = []
    data_spans = []
    
    for result in results:
        if not result.get('median_period'):
            continue
            
        periods.append(result['median_period'])
        confidences.append(result.get('confidence', 0.0))
        valid_flags.append(result.get('validation', {}).get('is_valid', False))
        n_transits.append(result.get('n_transits', 0))
        data_spans.append(result.get('data_span', 0.0))
    
    if not periods:
        return {}
        
    periods = np.array(periods)
    confidences = np.array(confidences)
    valid_flags = np.array(valid_flags)
    n_transits = np.array(n_transits)
    data_spans = np.array(data_spans)
    
    # Basic statistics
    stats_dict = {
        'n_total': len(periods),
        'n_valid': sum(valid_flags),
        'mean_period': float(np.mean(periods)),
        'median_period': float(np.median(periods)),
        'std_period': float(np.std(periods)),
        'mean_confidence': float(np.mean(confidences)),
        'median_confidence': float(np.median(confidences)),
        'period_range': (float(np.min(periods)), float(np.max(periods))),
        'mean_transits': float(np.mean(n_transits)),
        'median_transits': float(np.median(n_transits))
    }
    
    # Period distribution analysis
    period_bins = np.logspace(np.log10(0.5), np.log10(200), 50)
    hist, bin_edges = np.histogram(periods, bins=period_bins)
    stats_dict['period_distribution'] = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    # Confidence correlation analysis
    stats_dict['correlations'] = {
        'period_confidence': float(stats.pearsonr(periods, confidences)[0]),
        'transits_confidence': float(stats.pearsonr(n_transits, confidences)[0]),
        'span_confidence': float(stats.pearsonr(data_spans, confidences)[0])
    }
    
    # Distribution tests
    valid_periods = periods[valid_flags]
    if len(valid_periods) > 3:
        # Test for log-normal distribution of periods
        logperiods = np.log10(valid_periods)
        _, pval_normal = stats.normaltest(logperiods)
        stats_dict['distribution_tests'] = {
            'log_normal_pvalue': float(pval_normal)
        }
    
    return stats_dict

def plot_period_statistics(stats_dict: Dict, output_dir: Optional[str] = None) -> None:
    """
    Create visualizations of period statistics.
    
    Args:
        stats_dict: Dictionary of statistical analysis results
        output_dir: Directory to save plots (optional)
    """
    if not stats_dict:
        logger.warning("No statistics available for plotting")
        return
        
    # Create period distribution plot
    if 'period_distribution' in stats_dict:
        plt.figure(figsize=(12, 6))
        bin_edges = np.array(stats_dict['period_distribution']['bin_edges'])
        counts = np.array(stats_dict['period_distribution']['counts'])
        
        plt.semilogx(bin_edges[:-1], counts, 'k-', drawstyle='steps-post')
        plt.fill_between(bin_edges[:-1], counts, step='post', alpha=0.3)
        
        plt.xlabel('Period (days)')
        plt.ylabel('Number of Detections')
        plt.title('Distribution of Detected Periods')
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(f"{output_dir}/period_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def analyze_period_reliability(results: List[Dict]) -> Dict:
    """
    Analyze the reliability of period detections based on multiple metrics.
    
    Args:
        results: List of periodicity analysis results
        
    Returns:
        Dictionary containing reliability analysis results
    """
    reliability_scores = []
    period_ratios = []
    
    for result in results:
        if not result.get('median_period') or not result.get('quality_metrics'):
            continue
            
        # Calculate period ratio relative to data span
        period = result['median_period']
        data_span = result.get('data_span', 0.0)
        if data_span > 0:
            period_ratios.append(period / data_span)
        
        # Get quality metrics
        metrics = result['quality_metrics']
        confidence = metrics.get('confidence_score', 0.0)
        fap = metrics.get('fap', 1.0)
        power_snr = metrics.get('power_snr', 0.0)
        
        # Calculate reliability score
        reliability = confidence * (1 - fap) * min(1.0, power_snr / 10.0)
        reliability_scores.append(reliability)
    
    if not reliability_scores:
        return {}
    
    reliability_stats = {
        'mean_reliability': float(np.mean(reliability_scores)),
        'median_reliability': float(np.median(reliability_scores)),
        'std_reliability': float(np.std(reliability_scores)),
        'reliability_percentiles': {
            '10th': float(np.percentile(reliability_scores, 10)),
            '25th': float(np.percentile(reliability_scores, 25)),
            '75th': float(np.percentile(reliability_scores, 75)),
            '90th': float(np.percentile(reliability_scores, 90))
        }
    }
    
    if period_ratios:
        reliability_stats['period_span_ratios'] = {
            'mean': float(np.mean(period_ratios)),
            'median': float(np.median(period_ratios)),
            'std': float(np.std(period_ratios))
        }
    
    return reliability_stats

def plot_reliability_analysis(reliability_stats: Dict, output_dir: Optional[str] = None) -> None:
    """
    Create visualizations of period detection reliability.
    
    Args:
        reliability_stats: Dictionary of reliability analysis results
        output_dir: Directory to save plots (optional)
    """
    if not reliability_stats:
        logger.warning("No reliability statistics available for plotting")
        return
    
    # Create reliability percentile plot
    plt.figure(figsize=(10, 6))
    percentiles = reliability_stats.get('reliability_percentiles', {})
    if percentiles:
        values = [percentiles['10th'], percentiles['25th'], 
                 reliability_stats['median_reliability'],
                 percentiles['75th'], percentiles['90th']]
        labels = ['10th', '25th', 'Median', '75th', '90th']
        
        plt.boxplot([values], labels=['Period Detection Reliability'],
                   whis=[10, 90], meanline=True, showmeans=True)
        plt.ylabel('Reliability Score')
        plt.title('Period Detection Reliability Distribution')
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(f"{output_dir}/reliability_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def generate_analysis_report(results: List[Dict], output_dir: Optional[str] = None) -> Dict:
    """
    Generate a comprehensive analysis report of periodicity detection results.
    
    Args:
        results: List of periodicity analysis results
        output_dir: Directory to save plots and report (optional)
        
    Returns:
        Dictionary containing all analysis results
    """
    # Create output directory if needed
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # Perform all analyses
    period_stats = create_period_distribution_analysis(results)
    reliability_stats = analyze_period_reliability(results)
    
    # Generate visualizations
    if output_dir:
        plot_period_statistics(period_stats, output_dir)
        plot_reliability_analysis(reliability_stats, output_dir)
    
    # Combine all results
    report = {
        'period_statistics': period_stats,
        'reliability_analysis': reliability_stats,
        'metadata': {
            'total_samples': len(results),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save report if output directory provided
    if output_dir:
        import json
        report_path = f"{output_dir}/period_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(convert_numpy_int_to_int(report), f, indent=2)
        logger.info(f"Analysis report saved to {report_path}")
    
    return report