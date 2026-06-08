""" Tender: A Spatiotemporal Emotion Dynamics Framework for Group Chat Analysis.

This framework provides a complete pipeline for modeling, analyzing,
and intervening in group emotion dynamics within chat-based social environments.
It integrates topological data analysis with temporal causal inference
to uncover the shape and flow of collective emotional states.
"""
__version__ = "1.0.0"
__author__ = "Tender Contributors"
__license__ = "Apache 2.0"

from tender.pipeline.orchestrator import TenderPipeline
from tender.pipeline.config_loader import load_config

__all__ = [
    "TenderPipeline",
    "load_config",
]
