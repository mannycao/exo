"""
Cross-validation utilities for exoplanet detection.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

import config
from data.real_data_fetcher import load_rv_catalog, extract_star_name

logger = logging.getLogger(__name__)


def cross_validate_with_rv_data(transit_results, catalog=None, output_dir=None):
    """
    Cross-validate transit detections with radial velocity data.
    
    Args:
        transit_results: List of transit detection results
        catalog: RV catalog DataFrame (optional, will be loaded if None)
        output_dir: Directory to save validation reports
    
    Returns:
        dict: Enhanced results with RV validation
    """
    output_dir = output_dir or config.RESULTS_DIR
    
    # Load radial velocity catalog if not provided
    if catalog is None:
        catalog = load_rv_catalog()
    
    if catalog is None or len(catalog) == 0:
        logger.warning("No RV catalog available for cross-validation")
        return {
            'enhanced_results': transit_results,
            'validation_stats': {
                'total_detections': len(transit_results),
                'rv_validated': 0,
                'validation_rate': 0
            }
        }
    
    logger.info(f"Cross-validating {len(transit_results)} transit detections with {len(catalog)} RV planets")
    
    # Create lookup dictionaries
    rv_by_name = {name.lower(): entry for name, entry in catalog.groupby('star_name').first().iterrows()}
    
    # Match coordinates if available
    coord_matches = {}
    if 'ra' in catalog.columns and 'dec' in catalog.columns:
        # Create SkyCoord object for RV stars
        rv_coords = []
        rv_indices = []
        
        for i, row in catalog.iterrows():
            if pd.notna(row['ra']) and pd.notna(row['dec']):
                try:
                    rv_coords.append(SkyCoord(row['ra'], row['dec'], unit=(u.deg, u.deg)))
                    rv_indices.append(i)
                except:
                    continue
        
        if rv_coords:
            rv_catalog_coords = SkyCoord(rv_coords)
            
            # Will store coordinate matches
            coord_matches = {}
    
    # Enhance results with RV validation
    for result in transit_results:
        # Extract star name from file path
        file_name = os.path.basename(result.get('file_path', ''))
        star_name = extract_star_name(file_name)
        result['star_name'] = star_name
        
        # Initialize RV validation fields
        result['rv_validated'] = False
        result['rv_match_method'] = None
        result['rv_match_details'] = {}
        
        # Check if star is in RV catalog by name
        name_match = False
        for name_variant in generate_name_variants(star_name):
            if name_variant.lower() in rv_by_name:
                name_match = True
                rv_data = rv_by_name[name_variant.lower()]
                break
        
        if name_match:
            result['rv_match_method'] = 'name'
            
            # Check for period match (within 10%)
            if result.get('periodicity') and not pd.isna(rv_data.get('orbital_period')):
                rv_period = rv_data['orbital_period']
                transit_period = result['periodicity']
                period_diff_pct = abs(transit_period - rv_period) / rv_period * 100
                
                result['rv_match_details']['catalog_period'] = float(rv_period)
                result['rv_match_details']['detected_period'] = float(transit_period)
                result['rv_match_details']['period_difference_pct'] = float(period_diff_pct)
                
                # Validate if period difference is within threshold
                period_match = period_diff_pct < 10  # 10% threshold
                result['rv_validated'] = period_match
                
                # If RV data has planet properties, add them
                for prop in ['planet_mass', 'planet_radius']:
                    if prop in rv_data and not pd.isna(rv_data[prop]):
                        if 'planet_properties' not in result:
                            result['planet_properties'] = {}
                        result['rv_match_details'][prop] = float(rv_data[prop])
        
        # TODO: Add coordinate matching if needed in the future
    
    # Calculate validation statistics
    validated_count = sum(1 for r in transit_results if r.get('rv_validated', False))
    validation_rate = validated_count / len(transit_results) if transit_results else 0
    
    stats = {
        'total_detections': len(transit_results),
        'rv_validated': validated_count,
        'validation_rate': validation_rate
    }
    
    logger.info(f"RV validation results: {validated_count}/{len(transit_results)} validated ({validation_rate:.1%})")
    
    # Generate validation report
    if output_dir:
        generate_validation_report(transit_results, stats, output_dir)
    
    return {
        'enhanced_results': transit_results,
        'validation_stats': stats
    }


def generate_name_variants(star_name):
    """
    Generate possible variants of a star name for matching.
    
    Args:
        star_name: Original star name
    
    Returns:
        list: Variants of the star name
    """
    variants = [star_name]
    
    # Remove spaces
    variants.append(star_name.replace(' ', ''))
    
    # Handle Kepler IDs
    if 'KIC' in star_name.upper():
        # KIC 12345678 -> Kepler-12345678
        kic_num = star_name.upper().replace('KIC', '').strip()
        variants.append(f"Kepler-{kic_num}")
        variants.append(f"Kepler {kic_num}")
        variants.append(f"KOI-{kic_num}")
        variants.append(kic_num)
    
    # Handle Kepler names
    if 'KEPLER' in star_name.upper():
        # Kepler-12345678 -> KIC 12345678
        kepler_num = star_name.upper().replace('KEPLER-', '').replace('KEPLER', '').strip()
        variants.append(f"KIC {kepler_num}")
        variants.append(f"KOI-{kepler_num}")
        variants.append(kepler_num)
    
    # Handle TESS IDs
    if 'TIC' in star_name.upper():
        # TIC 12345678 -> TOI-12345678
        tic_num = star_name.upper().replace('TIC', '').strip()
        variants.append(f"TOI-{tic_num}")
        variants.append(f"TESS {tic_num}")
        variants.append(tic_num)
    
    return variants

def verify_detections_against_catalogs(detection_results, output_dir=None):
    """
    Verify detected transits against catalogs of confirmed planets and known false positives.
    
    Args:
        detection_results: List of light curve processing results
        output_dir: Directory to save verification results
    
    Returns:
        dict: Enhanced results with verification labels and metrics
    """
    import os
    import logging
    import pandas as pd
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import matplotlib.pyplot as plt
    
    logger = logging.getLogger(__name__)
    
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    
    verification_dir = os.path.join(output_dir, "verification")
    os.makedirs(verification_dir, exist_ok=True)
    
    # Load catalogs
    logger.info("Loading verification catalogs")
    
    # 1. Load confirmed planets catalog
    try:
        from data.data_fetcher import fetch_exoplanet_labels
        confirmed_planets = fetch_exoplanet_labels(use_cache=True)
        confirmed_hosts = set(confirmed_planets['hostname'].str.lower())
        logger.info(f"Loaded {len(confirmed_planets)} confirmed planets")
    except Exception as e:
        logger.error(f"Error loading confirmed planets: {e}")
        confirmed_planets = pd.DataFrame()
        confirmed_hosts = set()
    
    # 2. Load Kepler Objects of Interest (KOI) catalog with dispositions
    try:
        from data.data_fetcher import get_kepler_koi_targets
        confirmed_koi, false_positive_koi = get_kepler_koi_targets(use_cache=True)
        confirmed_koi_set = set(koi.lower() for koi in confirmed_koi)
        false_positive_set = set(fp.lower() for fp in false_positive_koi)
        logger.info(f"Loaded {len(confirmed_koi)} confirmed KOIs and {len(false_positive_koi)} false positives")
    except Exception as e:
        logger.error(f"Error loading KOI catalog: {e}")
        confirmed_koi_set = set()
        false_positive_set = set()
    
    # 3. Load RV planet catalog
    try:
        from data.real_data_fetcher import load_rv_catalog
        rv_catalog = load_rv_catalog(cache=True)
        rv_hosts = set(rv_catalog['star_name'].str.lower())
        logger.info(f"Loaded {len(rv_catalog)} RV-confirmed planets")
    except Exception as e:
        logger.error(f"Error loading RV catalog: {e}")
        rv_catalog = pd.DataFrame()
        rv_hosts = set()
    
    # Generate name variants for matching
    from validation.cross_validator import generate_name_variants
    
    # Process each detection result
    verified_results = []
    
    verification_stats = {
        'total': len(detection_results),
        'transit_detections': 0,
        'confirmed_true_positives': 0,
        'confirmed_false_positives': 0,
        'unverified': 0
    }
    
    for result in detection_results:
        # Skip non-successful results
        if not result.get('success', False):
            continue
            
        # Extract basic info
        has_transit = result.get('transit_count', 0) > 0
        file_path = result.get('file_path', '')
        star_name = os.path.basename(file_path).split('_')[0]  # Modify as needed
        
        # Initialize verification fields
        result['verification'] = {
            'verified': False,
            'label': 'unknown',
            'confidence': 0.0,
            'method': None,
            'details': {}
        }
        
        # Only verify transit detections
        if has_transit:
            verification_stats['transit_detections'] += 1
            
            # Match against catalogs using multiple methods
            matched = False
            
            # 1. Try direct name matching
            name_variants = generate_name_variants(star_name)
            
            # Check against confirmed planets
            for variant in name_variants:
                variant_lower = variant.lower()
                # Try different catalog matches
                if variant_lower in confirmed_hosts:
                    # Match in exoplanet archive
                    result['verification']['verified'] = True
                    result['verification']['label'] = 'true_positive'
                    result['verification']['confidence'] = 0.9
                    result['verification']['method'] = 'confirmed_planet_catalog'
                    matched = True
                    verification_stats['confirmed_true_positives'] += 1
                    break
                elif variant_lower in confirmed_koi_set:
                    # Match in KOI confirmed list
                    result['verification']['verified'] = True
                    result['verification']['label'] = 'true_positive'
                    result['verification']['confidence'] = 0.8
                    result['verification']['method'] = 'confirmed_koi'
                    matched = True
                    verification_stats['confirmed_true_positives'] += 1
                    break
                elif variant_lower in rv_hosts:
                    # Match in RV catalog
                    result['verification']['verified'] = True
                    result['verification']['label'] = 'true_positive'
                    result['verification']['confidence'] = 0.85
                    result['verification']['method'] = 'rv_confirmed'
                    matched = True
                    verification_stats['confirmed_true_positives'] += 1
                    break
                elif variant_lower in false_positive_set:
                    # Known false positive
                    result['verification']['verified'] = True
                    result['verification']['label'] = 'false_positive'
                    result['verification']['confidence'] = 0.85
                    result['verification']['method'] = 'known_false_positive'
                    matched = True
                    verification_stats['confirmed_false_positives'] += 1
                    break
            
            # 2. If not matched by name, check for period matches with RV planets
            if not matched and result.get('periodicity') and not rv_catalog.empty:
                period = result.get('periodicity')
                # Find potential period matches within 10%
                for _, row in rv_catalog.iterrows():
                    if pd.notna(row.get('orbital_period')):
                        rv_period = row['orbital_period']
                        period_diff = abs(period - rv_period) / rv_period
                        
                        if period_diff < 0.1:  # Within 10%
                            result['verification']['verified'] = True
                            result['verification']['label'] = 'true_positive'
                            result['verification']['confidence'] = 0.7
                            result['verification']['method'] = 'period_match'
                            result['verification']['details'] = {
                                'detected_period': period,
                                'catalog_period': rv_period,
                                'period_diff': period_diff,
                                'matching_star': row['star_name']
                            }
                            matched = True
                            verification_stats['confirmed_true_positives'] += 1
                            break
            
            # 3. If still not matched, use transit properties to estimate likelihood
            if not matched:
                # Analyze transit properties
                verification_score = 0.0
                
                # a. Check if period is in reasonable range for exoplanets
                if result.get('periodicity'):
                    period = result.get('periodicity')
                    if 0.5 <= period <= 100:
                        verification_score += 0.1
                    elif period > 100:
                        verification_score -= 0.1  # Less common for exoplanets
                
                # b. Check if planet properties are in reasonable ranges
                if result.get('planet_properties'):
                    props = result.get('planet_properties')
                    
                    # Check radius (if available)
                    if 'radius_earth' in props:
                        radius = props['radius_earth']
                        if 0.5 <= radius <= 4.0:  # Common range for exoplanets
                            verification_score += 0.1
                        elif radius > 15.0:  # Very large, likely not a planet
                            verification_score -= 0.2
                    
                    # Check equilibrium temperature
                    if 'equilibrium_temp_k' in props:
                        temp = props['equilibrium_temp_k']
                        if 200 <= temp <= 1500:  # Common range
                            verification_score += 0.1
                
                # c. Add verification score to result
                result['verification']['confidence'] = max(0.0, min(0.5, verification_score))
                result['verification']['method'] = 'property_analysis'
                result['verification']['details'] = {
                    'verification_score': verification_score
                }
                
                verification_stats['unverified'] += 1
        
        verified_results.append(result)
    
    # Calculate verification statistics
    verification_stats['verified_percent'] = (
        (verification_stats['confirmed_true_positives'] + verification_stats['confirmed_false_positives']) / 
        verification_stats['transit_detections'] * 100 if verification_stats['transit_detections'] > 0 else 0
    )
    
    # Create verification summary visualizations
    plt.figure(figsize=(10, 6))
    categories = ['True Positives', 'False Positives', 'Unverified']
    counts = [
        verification_stats['confirmed_true_positives'],
        verification_stats['confirmed_false_positives'],
        verification_stats['unverified']
    ]
    colors = ['green', 'red', 'gray']
    
    plt.bar(categories, counts, color=colors)
    plt.title('Transit Detection Verification Results')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(verification_dir, 'verification_summary.png'))
    plt.close()
    
    # Generate verification report
    with open(os.path.join(verification_dir, 'verification_report.txt'), 'w') as f:
        f.write("TRANSIT DETECTION VERIFICATION REPORT\n")
        f.write("====================================\n\n")
        f.write(f"Total detections analyzed: {verification_stats['transit_detections']}\n")
        f.write(f"Confirmed true positives: {verification_stats['confirmed_true_positives']}\n")
        f.write(f"Confirmed false positives: {verification_stats['confirmed_false_positives']}\n")
        f.write(f"Unverified detections: {verification_stats['unverified']}\n")
        f.write(f"Verification rate: {verification_stats['verified_percent']:.1f}%\n\n")
        
        f.write("VERIFICATION DETAILS:\n")
        f.write("--------------------\n")
        for result in verified_results:
            if result.get('transit_count', 0) > 0:
                f.write(f"\nFile: {os.path.basename(result.get('file_path', 'unknown'))}\n")
                f.write(f"Transits: {result.get('transit_count', 0)}\n")
                f.write(f"Period: {result.get('periodicity', 'unknown')}\n")
                f.write(f"Verification label: {result['verification']['label']}\n")
                f.write(f"Verification method: {result['verification']['method']}\n")
                f.write(f"Confidence: {result['verification']['confidence']:.2f}\n")
    
    logger.info(f"Verification complete: {verification_stats['confirmed_true_positives']} true positives, "
                f"{verification_stats['confirmed_false_positives']} false positives, "
                f"{verification_stats['unverified']} unverified")
    
    return {
        'verified_results': verified_results,
        'verification_stats': verification_stats
    }

def multi_method_validation(transit_results, output_dir=None):
    """
    Validate transit detections using multiple methods.
    
    Args:
        transit_results: List of transit detection results
        output_dir: Directory to save validation reports
    
    Returns:
        dict: Enhanced results with multi-method validation
    """
    output_dir = output_dir or config.RESULTS_DIR
    
    # Start with RV validation
    rv_validation = cross_validate_with_rv_data(transit_results, output_dir=output_dir)

    # Add catalog verification
    catalog_verification = verify_detections_against_catalogs(rv_validation['enhanced_results'], output_dir=output_dir)
    
    combined_results = {
        'enhanced_results': catalog_verification['verified_results'],
        'validation_stats': {
            **rv_validation['validation_stats'],
            **catalog_verification['verification_stats']
        }
    }
    
    return combined_results


def generate_validation_report(results, stats, output_dir):
    """
    Generate validation report with visualizations.
    
    Args:
        results: Enhanced transit detection results
        stats: Validation statistics
        output_dir: Output directory for report files
    """
    # Create validation directory
    validation_dir = os.path.join(output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    # Extract validation data
    validated = [r for r in results if r.get('rv_validated', False)]
    non_validated = [r for r in results if not r.get('rv_validated', False)]
    
    # Get periods for comparison
    validated_periods = [r.get('periodicity') for r in validated if r.get('periodicity') is not None]
    non_validated_periods = [r.get('periodicity') for r in non_validated if r.get('periodicity') is not None]
    
    # Create period comparison chart
    plt.figure(figsize=(10, 6))
    
    n_bins = 20
    if validated_periods:
        plt.hist(validated_periods, bins=n_bins, alpha=0.5, label='RV Validated')
    if non_validated_periods:
        plt.hist(non_validated_periods, bins=n_bins, alpha=0.5, label='Not Validated')
    
    plt.xlabel('Orbital Period (days)')
    plt.ylabel('Number of Detections')
    plt.title('Period Distribution: Validated vs. Non-validated')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(validation_dir, 'period_validation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create period comparison scatter plot for validated planets
    if validated and any('rv_match_details' in r for r in validated):
        # Extract period comparisons
        rv_periods = []
        transit_periods = []
        
        for r in validated:
            if 'rv_match_details' in r and 'catalog_period' in r['rv_match_details']:
                rv_periods.append(r['rv_match_details']['catalog_period'])
                transit_periods.append(r['rv_match_details']['detected_period'])
        
        if rv_periods:
            plt.figure(figsize=(8, 8))
            
            plt.scatter(rv_periods, transit_periods, alpha=0.7)
            
            # Add perfect match line
            max_val = max(max(rv_periods), max(transit_periods)) * 1.1
            min_val = min(min(rv_periods), min(transit_periods)) * 0.9
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Add 10% tolerance lines
            plt.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'r--', alpha=0.3)
            plt.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'r--', alpha=0.3)
            
            plt.xlabel('RV Catalog Period (days)')
            plt.ylabel('Transit Detection Period (days)')
            plt.title('Period Comparison: RV vs Transit Detection')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(validation_dir, 'period_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create HTML report
    with open(os.path.join(validation_dir, 'validation_report.html'), 'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html lang='en'>\n")
        f.write("<head>\n")
        f.write("    <meta charset='UTF-8'>\n")
        f.write("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
        f.write("    <title>Transit Cross-Validation Report</title>\n")
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
        f.write("        .validated { background-color: #d4edda; }\n")
        f.write("        .not-validated { background-color: #f8d7da; }\n")
        f.write("    </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write(f"    <h1>Transit Detection Cross-Validation Report</h1>\n")
        
        # Summary section
        f.write("    <h2>Validation Summary</h2>\n")
        f.write("    <div style='display: flex; flex-wrap: wrap;'>\n")
        
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>Total Detections</div>\n")
        f.write(f"            <div class='metric-value'>{stats['total_detections']}</div>\n")
        f.write("        </div>\n")
        
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>RV Validated</div>\n")
        f.write(f"            <div class='metric-value'>{stats['rv_validated']}</div>\n")
        f.write("        </div>\n")
        
        f.write("        <div class='metric-card'>\n")
        f.write("            <div class='metric-label'>Validation Rate</div>\n")
        f.write(f"            <div class='metric-value'>{stats['validation_rate']:.1%}</div>\n")
        f.write("        </div>\n")
        
        f.write("    </div>\n")
        
        # Visualization section
        f.write("    <h2>Validation Visualizations</h2>\n")
        f.write("    <div class='gallery'>\n")
        
        period_validation_path = 'period_validation.png'
        if os.path.exists(os.path.join(validation_dir, period_validation_path)):
            f.write(f"        <div class='gallery-item'><img src='{period_validation_path}' alt='Period Distribution'><p>Period Distribution: Validated vs Non-validated</p></div>\n")
        
        period_comparison_path = 'period_comparison.png'
        if os.path.exists(os.path.join(validation_dir, period_comparison_path)):
            f.write(f"        <div class='gallery-item'><img src='{period_comparison_path}' alt='Period Comparison'><p>Period Comparison: RV vs Transit Detection</p></div>\n")
        
        f.write("    </div>\n")
        
        # Validated results table
        f.write("    <h2>RV-Validated Detections</h2>\n")
        if validated:
            f.write("    <table>\n")
            f.write("        <thead>\n")
            f.write("            <tr>\n")
            f.write("                <th>Star</th>\n")
            f.write("                <th>Detected Period (days)</th>\n")
            f.write("                <th>RV Period (days)</th>\n")
            f.write("                <th>Difference (%)</th>\n")
            f.write("                <th>Transit Count</th>\n")
            f.write("            </tr>\n")
            f.write("        </thead>\n")
            f.write("        <tbody>\n")
            
            for result in validated:
                star = result.get('star_name', 'Unknown')
                detected_period = result.get('periodicity', 'N/A')
                if detected_period != 'N/A':
                    detected_period = f"{detected_period:.2f}"
                
                rv_period = 'N/A'
                diff_pct = 'N/A'
                
                if 'rv_match_details' in result:
                    details = result['rv_match_details']
                    if 'catalog_period' in details:
                        rv_period = f"{details['catalog_period']:.2f}"
                    if 'period_difference_pct' in details:
                        diff_pct = f"{details['period_difference_pct']:.1f}"
                
                transit_count = result.get('transit_count', 0)
                
                f.write("            <tr class='validated'>\n")
                f.write(f"                <td>{star}</td>\n")
                f.write(f"                <td>{detected_period}</td>\n")
                f.write(f"                <td>{rv_period}</td>\n")
                f.write(f"                <td>{diff_pct}</td>\n")
                f.write(f"                <td>{transit_count}</td>\n")
                f.write("            </tr>\n")
            
            f.write("        </tbody>\n")
            f.write("    </table>\n")
        else:
            f.write("    <p>No validated detections found.</p>\n")
        
        f.write("</body>\n")
        f.write("</html>\n")