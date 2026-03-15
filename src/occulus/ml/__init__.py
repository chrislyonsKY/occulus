"""Deep learning semantic segmentation for point clouds.

Provides inference wrappers for pre-trained models (ONNX Runtime or
PyTorch backends) to classify points into semantic categories such as
ground, vegetation, building, water, and powerline.

This module does **not** train models — it loads pre-trained weights
and runs inference only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusMLError

if TYPE_CHECKING:
    from occulus.types import PointCloud

logger = logging.getLogger(__name__)

# Standard ASPRS-like class names
DEFAULT_CLASS_NAMES: dict[int, str] = {
    0: "unclassified",
    1: "ground",
    2: "low_vegetation",
    3: "medium_vegetation",
    4: "high_vegetation",
    5: "building",
    6: "water",
    7: "powerline",
    8: "vehicle",
}


@dataclass
class SegmentationPrediction:
    """Result of semantic segmentation inference.

    Attributes
    ----------
    labels : NDArray[np.int32]
        Per-point predicted class label.
    probabilities : NDArray[np.float32]
        Per-point class probabilities, shape (N, num_classes).
    class_names : dict[int, str]
        Mapping from class ID to human-readable name.
    """

    labels: NDArray[np.int32]
    probabilities: NDArray[np.float32]
    class_names: dict[int, str]


def prepare_features(
    cloud: PointCloud,
    *,
    use_rgb: bool = True,
    use_intensity: bool = True,
    use_normals: bool = False,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Prepare input features for ML inference.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    use_rgb : bool
        Include RGB channels if available.
    use_intensity : bool
        Include intensity channel if available.
    use_normals : bool
        Include surface normals if available.
    normalize : bool
        Center and scale XYZ to unit sphere.

    Returns
    -------
    NDArray[np.float32]
        Feature matrix of shape (N, D) where D depends on available attributes.
    """
    xyz = cloud.xyz.astype(np.float32)

    if normalize:
        centroid = xyz.mean(axis=0)
        xyz = xyz - centroid
        max_dist = np.linalg.norm(xyz, axis=1).max()
        if max_dist > 0:
            xyz = xyz / max_dist

    features = [xyz]

    if use_rgb and hasattr(cloud, "rgb") and cloud.rgb is not None:
        rgb = cloud.rgb.astype(np.float32) / 255.0
        features.append(rgb)

    if use_intensity and hasattr(cloud, "intensity") and cloud.intensity is not None:
        intensity = cloud.intensity.astype(np.float32).reshape(-1, 1)
        if normalize and intensity.max() > 0:
            intensity = intensity / intensity.max()
        features.append(intensity)

    if use_normals and hasattr(cloud, "normals") and cloud.normals is not None:
        features.append(cloud.normals.astype(np.float32))

    result = np.hstack(features)
    logger.debug("Prepared feature matrix: shape %s", result.shape)
    return result


def predict_semantic(
    cloud: PointCloud,
    model_path: str | Path,
    *,
    backend: str = "onnx",
    batch_size: int = 4096,
    device: str = "cpu",
    num_classes: int = 9,
    class_names: dict[int, str] | None = None,
) -> SegmentationPrediction:
    """Run semantic segmentation inference on a point cloud.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    model_path : str or Path
        Path to a pre-trained model file (.onnx or .pt/.pth).
    backend : str
        Inference backend: 'onnx' (ONNX Runtime) or 'torch' (PyTorch).
    batch_size : int
        Number of points per inference batch.
    device : str
        Device for inference ('cpu' or 'cuda').
    num_classes : int
        Number of output classes.
    class_names : dict[int, str], optional
        Custom class name mapping.  Defaults to ASPRS-like names.

    Returns
    -------
    SegmentationPrediction
        Per-point labels and class probabilities.

    Raises
    ------
    OcculusMLError
        If the model cannot be loaded or inference fails.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise OcculusMLError(f"Model file not found: {model_path}")

    if class_names is None:
        class_names = {k: v for k, v in DEFAULT_CLASS_NAMES.items() if k < num_classes}

    features = prepare_features(cloud)
    n = len(features)

    logger.info(
        "Running %s inference on %d points (batch_size=%d, device=%s)",
        backend,
        n,
        batch_size,
        device,
    )

    if backend == "onnx":
        all_probs = _predict_onnx(features, model_path, batch_size)
    elif backend == "torch":
        all_probs = _predict_torch(features, model_path, batch_size, device)
    else:
        raise OcculusMLError(f"Unknown backend: {backend}. Use 'onnx' or 'torch'.")

    # Ensure correct shape
    if all_probs.shape[1] != num_classes:
        logger.warning(
            "Model output %d classes, expected %d",
            all_probs.shape[1],
            num_classes,
        )

    labels = all_probs.argmax(axis=1).astype(np.int32)
    logger.info(
        "Inference complete. Class distribution: %s",
        dict(zip(*np.unique(labels, return_counts=True), strict=False)),
    )

    return SegmentationPrediction(
        labels=labels,
        probabilities=all_probs,
        class_names=class_names,
    )


def _predict_onnx(
    features: NDArray[np.float32],
    model_path: Path,
    batch_size: int,
) -> NDArray[np.float32]:
    """Run inference with ONNX Runtime."""
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OcculusMLError(
            "ONNX inference requires onnxruntime: pip install occulus[ml]"
        ) from exc

    try:
        session = ort.InferenceSession(str(model_path))
    except Exception as exc:
        raise OcculusMLError(f"Failed to load ONNX model: {exc}") from exc

    input_name = session.get_inputs()[0].name
    n = len(features)
    all_probs = []

    for i in range(0, n, batch_size):
        batch = features[i : i + batch_size]
        # Add batch dimension if needed
        if batch.ndim == 2:
            batch = batch[np.newaxis, ...]
        outputs = session.run(None, {input_name: batch})
        probs = outputs[0]
        if probs.ndim == 3:
            probs = probs.squeeze(0)
        all_probs.append(probs)

    return np.vstack(all_probs)


def _predict_torch(
    features: NDArray[np.float32],
    model_path: Path,
    batch_size: int,
    device: str,
) -> NDArray[np.float32]:
    """Run inference with PyTorch."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OcculusMLError(
            "PyTorch inference requires torch: pip install occulus[ml-torch]"
        ) from exc

    try:
        model = torch.load(str(model_path), map_location=device, weights_only=True)
        model.eval()
    except Exception as exc:
        raise OcculusMLError(f"Failed to load PyTorch model: {exc}") from exc

    n = len(features)
    all_probs = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(features[i : i + batch_size]).to(device)
            if batch.ndim == 2:
                batch = batch.unsqueeze(0)
            output = model(batch)
            probs = torch.softmax(output, dim=-1).squeeze(0).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)
