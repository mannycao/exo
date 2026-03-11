from setuptools import setup, find_packages

setup(
    name="multimodal-disagreement",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "tensorflow",
        "statsmodels",
        "scikit-posthocs"
    ],
    author="Emmanuel Cao",
    description="A Python package for quantifying inter-modal disagreement in deep learning exoplanet classifiers",
    python_requires=">=3.8",
)
