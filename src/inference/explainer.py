"""Model explainability: SHAP attributions, attention maps and prose."""

import base64
import io
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_TORCH_TYPES = {"lstm", "tcn", "transformer"}
_TREE_TYPES = {"xgboost", "rf"}


class SentinelExplainer:
    """Produces SHAP and attention-based explanations for RUL models.

    Deep models are explained with :class:`shap.DeepExplainer` and tree
    models with :class:`shap.TreeExplainer`. Feature attributions are
    summarized into ranked lists, plots and rule-based natural-language
    text.
    """

    def __init__(
        self, models: Dict[str, Any], feature_names: List[str]
    ) -> None:
        """Initialize the explainer.

        Args:
            models: Mapping of model name to model instance.
            feature_names: Ordered names of the input features.
        """
        self.models = models
        self.feature_names = feature_names

    def shap_explain(
        self, model: Any, X: np.ndarray, model_type: str
    ) -> Dict[str, Any]:
        """Compute SHAP values for a model on a batch of inputs.

        Args:
            model: The model whose predictions are explained.
            X: Input array; windowed for deep models, 2-D for tree models.
            model_type: One of ``lstm``, ``tcn``, ``transformer``,
                ``xgboost`` or ``rf``.

        Returns:
            A dict with the ``shap_values`` array and the ``model_type``.

        Raises:
            ValueError: If ``model_type`` is not recognized.
        """
        if model_type in _TORCH_TYPES:
            background = torch.as_tensor(X, dtype=torch.float32)
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(background)
        elif model_type in _TREE_TYPES:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            raise ValueError(f"Unsupported model_type: {model_type!r}")

        values = np.asarray(
            shap_values[0] if isinstance(shap_values, list) else shap_values
        )
        logger.info(
            "Computed SHAP values for %s, shape %s", model_type, values.shape
        )
        return {"shap_values": values, "model_type": model_type}

    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rank features by mean absolute SHAP contribution.

        Args:
            shap_values: SHAP value array; reduced over all axes but the
                last (feature) axis.
            feature_names: Names aligned to the feature axis.
            n: Number of top features to return.

        Returns:
            A list of dicts with ``name``, ``contribution``, ``direction``
            and ``value`` keys, ordered by descending importance.
        """
        values = np.asarray(shap_values)
        flat = values.reshape(-1, values.shape[-1])
        mean_signed = flat.mean(axis=0)
        mean_abs = np.abs(flat).mean(axis=0)
        total = float(mean_abs.sum()) or 1.0

        order = np.argsort(mean_abs)[::-1][:n]
        top: List[Dict[str, Any]] = []
        for index in order:
            top.append(
                {
                    "name": feature_names[index],
                    "contribution": float(mean_abs[index] / total),
                    "direction": "up" if mean_signed[index] >= 0 else "down",
                    "value": float(mean_signed[index]),
                }
            )
        return top

    @staticmethod
    def _fig_to_base64(fig: "plt.Figure") -> str:
        """Render a Matplotlib figure to a base64-encoded PNG string.

        Args:
            fig: The figure to encode.

        Returns:
            The base64-encoded PNG payload.
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")

    def shap_summary_plot(
        self, shap_values: np.ndarray, feature_names: List[str]
    ) -> str:
        """Render a SHAP feature-importance bar chart as a PNG.

        Args:
            shap_values: SHAP value array.
            feature_names: Names aligned to the feature axis.

        Returns:
            A base64-encoded PNG string of the summary plot.
        """
        values = np.asarray(shap_values)
        flat = values.reshape(-1, values.shape[-1])
        mean_abs = np.abs(flat).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]

        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(order))))
        ax.barh(
            [feature_names[i] for i in order][::-1],
            mean_abs[order][::-1],
            color="#2c7fb8",
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance")
        return self._fig_to_base64(fig)

    def attention_heatmap(
        self, attention_weights: np.ndarray, feature_names: List[str]
    ) -> str:
        """Render an attention-weight heatmap as a PNG.

        Args:
            attention_weights: 2-D array of shape ``(timesteps, sensors)``.
            feature_names: Names aligned to the sensor axis.

        Returns:
            A base64-encoded PNG string of the heatmap.
        """
        weights = np.asarray(attention_weights)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)

        fig, ax = plt.subplots(figsize=(10, 6))
        image = ax.imshow(weights, aspect="auto", cmap="viridis")
        ax.set_xlabel("Sensor")
        ax.set_ylabel("Timestep")
        ax.set_title("Attention Weights")
        ax.set_xticks(range(min(len(feature_names), weights.shape[1])))
        ax.set_xticklabels(
            feature_names[: weights.shape[1]], rotation=90, fontsize=7
        )
        fig.colorbar(image, ax=ax, label="Attention")
        return self._fig_to_base64(fig)

    def natural_language_explanation(
        self, top_features: List[Dict[str, Any]], rul: float
    ) -> str:
        """Build a rule-based prose explanation of a prediction.

        Args:
            top_features: Ranked features from :meth:`get_top_features`.
            rul: The predicted remaining useful life in cycles.

        Returns:
            A paragraph describing the top drivers and the risk outlook.
        """
        sentences: List[str] = []
        for index, feature in enumerate(top_features[:3]):
            movement = (
                "increased"
                if feature["direction"] == "up"
                else "decreased"
            )
            magnitude = abs(feature["contribution"]) * 100.0
            role = "primary driver" if index == 0 else "contributing factor"
            sentences.append(
                f"Sensor {feature['name']} has {movement} "
                f"{magnitude:.0f}% from baseline — {role} of the failure "
                f"prediction"
            )

        if rul < 30.0:
            risk = (
                f"With an estimated {rul:.0f} cycles of life remaining, "
                f"this unit is in a CRITICAL state and requires immediate "
                f"maintenance."
            )
        elif rul <= 100.0:
            risk = (
                f"With an estimated {rul:.0f} cycles of life remaining, "
                f"this unit is in a WARNING state and should be scheduled "
                f"for inspection."
            )
        else:
            risk = (
                f"With an estimated {rul:.0f} cycles of life remaining, "
                f"this unit is HEALTHY and operating within expected "
                f"degradation bounds."
            )

        explanation = ". ".join(sentences)
        if explanation:
            explanation += ". "
        explanation += risk
        return explanation
