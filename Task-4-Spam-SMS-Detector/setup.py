from setuptools import setup, find_packages

setup(
    name="spam_sms_detection",
    version="1.0.0",
    description="AI-powered SMS Spam Detector using TF-IDF + ML classifiers",
    author="Your Name",
    packages=find_packages(include=["src", "app"]),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.4",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "flask>=3.0",
        "nltk>=3.8",
        "joblib>=1.3",
        "gunicorn>=22.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "spam-train=src.train:train",
            "spam-predict=src.predict:main",
        ]
    },
)
