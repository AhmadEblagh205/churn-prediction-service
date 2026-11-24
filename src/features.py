"""Stage 4: features as a pipeline, not a script. Scaling and one hot
encoding live inside a ColumnTransformer so exactly the same
transformation runs in training, in tests and behind the API, and unseen
categories at serving time are ignored rather than fatal."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from validate import CATEGORICAL, NUMERIC


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
    ])


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", model)])
