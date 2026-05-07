from setuptools import setup, find_packages

setup(
    name='movie-genre-classifier',
    version='1.0.0',
    description='Classify movie genres from plot summaries using ML/NLP',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.3.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'nltk>=3.8.1',
        'scipy>=1.11.0',
        'gensim>=4.3.0',
        'flask>=2.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'joblib>=1.3.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'genre-train=src.train:main',
            'genre-predict=src.predict:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
