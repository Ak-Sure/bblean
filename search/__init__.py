"""
BitBIRCH IVF Search - Efficient Molecular Fingerprint Search

This package implements an Inverted File (IVF) search index for BitBIRCH clustering
algorithm, designed for efficient similarity search in large chemical fingerprint databases.
"""

from search.ivf_index import IVFIndex
from search.similarity_engines import RDKitEngine, FPSim2Engine, get_engine

__all__ = ["IVFIndex", "RDKitEngine", "FPSim2Engine", "get_engine"]