"""
FMCG Promotional Analytics — Custom Vertex AI Predictor

Implements the custom prediction interface required by Vertex AI
when serving a model via a custom container.

The XGBPredictor class is loaded by the Vertex AI prediction server
at endpoint initialisation. It handles:
- Loading the serialised XGBoost model from the GCS artefact URI
- Receiving inference requests as a list of feature vectors
- Returning predicted sell-out volumes as a list

Note: Model was trained as a raw XGBoost Booster and serialised
with pickle — inputs must match the exact 97-feature order used
during training. See deploy.py for the full feature list.

Author: Panos Emmanouilidis
"""

import pickle
import numpy as np
import xgboost as xgb
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor


class XGBPredictor:
    """
    Custom predictor class for Vertex AI online prediction.
    Wraps a serialised XGBoost Booster model and exposes
    a predict() method compatible with the Vertex AI serving interface.
    """

    def __init__(self):
        self._model = None

    def load(self, artifacts_uri: str):
        """
        Loads the serialised model from the Vertex AI artefact URI.
        Called once at container startup — not on every prediction request.
        """
        import joblib, os
        model_path = os.path.join(artifacts_uri, "model.pkl")
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

    def predict(self, instances):
        """
        Runs inference on a batch of input instances.

        Args:
            instances: list of feature vectors, each containing
                       97 numeric values in training feature order

        Returns:
            list of predicted sell-out volumes (log1p scale —
            apply np.expm1() to recover unit volumes)
        """
        inputs = np.array(instances)
        preds  = self._model.predict(inputs)
        return preds.tolist()
