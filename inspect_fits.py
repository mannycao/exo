from astropy.io import fits
from pathlib import Path

def inspect_fits_header(file_path):
    """
    Inspects the header of a FITS file for period-related keywords.
    """
    try:
        with fits.open(file_path, mode='readonly') as hdul:
            print(f"Inspecting FITS header for: {file_path}")
            print("\n--- Primary Header (HDU 0) ---")
            found_period_in_primary = False
            for key, value in hdul[0].header.items():
                print(f"{key}: {value}")
                if key == 'PERIOD' or key == 'ORBPER':
                    print(f"!!! Found period keyword in primary header: {key} = {value}")
                    found_period_in_primary = True
            
            if not found_period_in_primary:
                print("No PERIOD or ORBPER found in primary header.")

            if len(hdul) > 1:
                print("\n--- Data Header (HDU 1) ---")
                found_period_in_data = False
                for key, value in hdul[1].header.items():
                    print(f"{key}: {value}")
                    if key == 'PERIOD' or key == 'ORBPER':
                        print(f"!!! Found period keyword in data header: {key} = {value}")
                        found_period_in_data = True
                if not found_period_in_data:
                    print("No PERIOD or ORBPER found in data header.")
            else:
                print("\nNo Data HDU (HDU 1) found.")
            
            print("\n--- End of Inspection ---")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the FITS file: {e}")

if __name__ == "__main__":
    test_file_path = "/Users/emmanuel/proj/exo/kepler_local_data/confirmed_planets/mastDownload/TESS/tess2024003055635-s0074-0000000048503881-0269-s/tess2024003055635-s0074-0000000048503881-0269-s_lc.fits"
    inspect_fits_header(test_file_path)