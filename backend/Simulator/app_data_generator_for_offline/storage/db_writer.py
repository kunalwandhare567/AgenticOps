"""
app_simulator/storage/db_writer.py
====================================
Re-exports DbWriter from Inference_langgraph.nodes.db_writer for backward compatibility.
"""
from Inference_langgraph.nodes.db_writer.db_writer import DbWriter

__all__ = ["DbWriter"]
