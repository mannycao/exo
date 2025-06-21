import sys
import importlib
from pprint import pprint

# --- Start of Diagnostic Script ---

print("--- Python Environment Diagnostics ---")
print(f"Python Executable: {sys.executable}")
print("\nPython Search Path (sys.path):")
pprint(sys.path)
print("-" * 35)

# List of essential packages to check
packages_to_check = [
    # Core Libraries
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    
    # Machine Learning
    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
    ("sklearn", "scikit-learn"),
    ("imblearn", "imbalanced-learn"),

    # Astronomy Specific
    ("astropy", "astropy"),
    ("astroquery", "astroquery"),
    ("skimage", "scikit-image"), # scikit-image is imported as 'skimage'
    ("batman", "batman-package"),
]

print("\n--- Checking Package Imports ---")
all_successful = True

for import_name, package_name in packages_to_check:
    try:
        # Try to import the library
        module = importlib.import_module(import_name)
        
        # Get version and location
        version = getattr(module, '__version__', 'N/A')
        location = getattr(module, '__file__', 'N/A')
        
        print(f"✅ SUCCESS: Imported '{import_name}' (from package '{package_name}')")
        print(f"    - Version: {version}")
        print(f"    - Location: {location}\n")
        
    except ImportError:
        print(f"❌ FAILED: Could not import '{import_name}' (from package '{package_name}')\n")
        all_successful = False

print("-" * 35)

if all_successful:
    print("✅ All essential libraries were imported successfully.")
else:
    print("❌ One or more essential libraries failed to import.")

# --- End of Diagnostic Script ---
