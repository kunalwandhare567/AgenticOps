"""
backend/nodes/reliability/__init__.py
======================================
Reliability Node — Stage 8 of the AIOps pipeline.

Submodules
----------
extractor     : Life-data extraction (TTF + right-censoring) from pipeline CSVs.
weibull_fitter: 2P Weibull MLE fitting (reliability + lifelines libraries).
metrics       : MTTF, MTTR, Availability calculations.
plotter       : Diagnostic plot generation (R(t), Hazard, KM overlay, Per-mode).
"""
from .extractor import LifeDataExtractor, extract_life_data

__all__ = ["LifeDataExtractor", "extract_life_data"]
