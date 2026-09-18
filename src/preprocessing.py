from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """
    Remove features whose missing-value percentage exceeds
    a threshold learned from the training data.

    The transformer is fit only on the training set to avoid
    data leakage.
    """

    def __init__(self, threshold=0.50):
        self.threshold = threshold

    def fit(self, X, y=None):

        missing_fraction = X.isna().mean()

        self.columns_to_keep_ = missing_fraction[
            missing_fraction <= self.threshold
        ].index.tolist()

        self.columns_to_drop_ = missing_fraction[
            missing_fraction > self.threshold
        ].index.tolist()

        return self

    def transform(self, X):

        return X[self.columns_to_keep_].copy()


def build_preprocessor(scale=True):
    """
    Build the preprocessing pipeline.

    Steps
    -----
    1. Remove features with >50% missing data.
    2. Median-impute remaining missing values.
    3. Remove zero-variance features.
    4. Optionally standardize predictors.
    """

    steps = [
        (
            "missingness_filter",
            MissingnessFilter(
                threshold=0.50
            ),
        ),
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "variance_filter",
            VarianceThreshold(
                threshold=0.0
            ),
        ),
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(steps)