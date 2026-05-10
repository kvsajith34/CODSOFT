"""
balancer.py

Fraud datasets are always ridiculously imbalanced. On this one it's like 0.5% fraud 
(which is still tiny). So I put together three different ways to deal with it using 
imbalanced-learn. The key thing is that we ONLY resample inside the cross-validation 
loops – never on the whole dataset. Otherwise you get leaky, overly optimistic results.

I went with the 'combined' strategy by default because it gave me the best balance 
between performance and training time during my experiments.
"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


STRATEGIES = ("smote", "undersample", "combined")


def make_resampler(strategy: str = "combined"):
    """Builds the appropriate resampling pipeline for the chosen strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got '{strategy}'")

    if strategy == "smote":
        return ImbPipeline(
            steps=[("smote", SMOTE(random_state=42, k_neighbors=5))]
        )

    if strategy == "undersample":
        return ImbPipeline(
            steps=[("under", RandomUnderSampler(random_state=42))]
        )

    # combined approach: first bring the majority class down to ~10x the minority,
    # then SMOTE the minority back up to equal. Works pretty well in practice.
    return ImbPipeline(
        steps=[
            ("under", RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
            ("smote", SMOTE(sampling_strategy=1.0, random_state=42, k_neighbors=5)),
        ]
    )


def resample(X, y, strategy: str = "combined"):
    """Convenience wrapper – just makes the resampler and fits it."""
    resampler = make_resampler(strategy)
    return resampler.fit_resample(X, y)