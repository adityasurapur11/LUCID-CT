"""
Adaptive Self-Verifying Deep Learning Framework for Early Lung Cancer Detection
"""

from src.model import HybridLungCancerModel
from src.scan_quality import ScanQualityAssessor
from src.trait_verifier import TraitSpecificRegionalVerifier
from src.xai_verifier import ExplanationFidelityVerifier
from src.adaptive_controller import AdaptiveVerificationController
from src.longitudinal import LongitudinalNoduleTracker
from src.evaluation import BenchmarkAblationSuite

__version__ = "2.0.0"

__all__ = [
    "HybridLungCancerModel",
    "ScanQualityAssessor",
    "TraitSpecificRegionalVerifier",
    "ExplanationFidelityVerifier",
    "AdaptiveVerificationController",
    "LongitudinalNoduleTracker",
    "BenchmarkAblationSuite",
]
