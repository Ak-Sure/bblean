"""
Similarity engine implementations for chemical fingerprints.

This module provides wrappers for different similarity search engines:
1. RDKit's BulkTanimotoSimilarity
2. FPSim2's optimized similarity search

Each engine implements a common interface for easy interchangeability.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
from abc import ABC, abstractmethod

class SimilarityEngine(ABC):
    """
    Abstract base class for similarity engines.
    
    All similarity engines should implement this interface.
    """
    
    @abstractmethod
    def bulk_similarity(self, query_fp: Any, target_fps: List[Any]) -> List[float]:
        """
        Calculate similarities between query and multiple target fingerprints.
        
        Args:
            query_fp: Query fingerprint
            target_fps: List of target fingerprints
            
        Returns:
            List of similarity scores
        """
        pass
    
    @abstractmethod
    def top_k_similarity(self, query_fp: Any, target_fps: List[Any], k: int, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find top-k most similar fingerprints.
        
        Args:
            query_fp: Query fingerprint
            target_fps: List of target fingerprints
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries containing {index, similarity} pairs
        """
        pass


class RDKitEngine(SimilarityEngine):
    """
    Similarity engine using RDKit's BulkTanimotoSimilarity.
    """
    
    def __init__(self):
        """
        Initialize RDKit engine.
        """
        try:
            from rdkit import DataStructs
            from rdkit.Chem import AllChem
            self.DataStructs = DataStructs
            self.AllChem = AllChem
        except ImportError:
            raise ImportError("RDKit is not installed. Please install it with 'pip install rdkit'.")
    
    def _ensure_bitvect(self, fp: Any) -> Any:
        """
        Ensure fingerprint is in RDKit ExplicitBitVect format.
        
        Args:
            fp: Input fingerprint (array-like or ExplicitBitVect)
            
        Returns:
            ExplicitBitVect fingerprint
        """
        from rdkit import DataStructs
        import numpy as np
        
        if isinstance(fp, DataStructs.ExplicitBitVect):
            return fp
            
        if isinstance(fp, (np.ndarray, list)):
            bitvect = DataStructs.ExplicitBitVect(len(fp))
            for i, val in enumerate(fp):
                if val == 1:
                    bitvect.SetBit(i)
            return bitvect
            
        raise ValueError(f"Unsupported fingerprint type: {type(fp)}")
        
    def _convert_fps_to_bitvects(self, fps: List[Any]) -> List[Any]:
        """
        Convert a list of fingerprints to RDKit ExplicitBitVect format.
        
        Args:
            fps: List of fingerprints
            
        Returns:
            List of ExplicitBitVect fingerprints
        """
        return [self._ensure_bitvect(fp) for fp in fps]
        
    def bulk_similarity(self, query_fp: Any, target_fps: List[Any]) -> List[float]:
        """
        Calculate Tanimoto similarities using RDKit's BulkTanimotoSimilarity.
        
        Args:
            query_fp: Query fingerprint (preferably RDKit ExplicitBitVect)
            target_fps: List of target fingerprints (preferably RDKit ExplicitBitVect)
            
        Returns:
            List of Tanimoto similarity scores
        """
        # Check if inputs are already RDKit ExplicitBitVect objects
        from rdkit import DataStructs
        if isinstance(query_fp, DataStructs.ExplicitBitVect) and all(isinstance(fp, DataStructs.ExplicitBitVect) for fp in target_fps):
            # Use directly without conversion for best performance
            return list(self.DataStructs.BulkTanimotoSimilarity(query_fp, target_fps))
        else:
            # Fall back to conversion if needed
            query_bitvect = self._ensure_bitvect(query_fp)
            target_bitvects = self._convert_fps_to_bitvects(target_fps)
            return list(self.DataStructs.BulkTanimotoSimilarity(query_bitvect, target_bitvects))
        
    def top_k_similarity(self, query_fp: Any, target_fps: List[Any], k: int, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find top-k most similar fingerprints using RDKit.
        
        Args:
            query_fp: Query fingerprint
            target_fps: List of target fingerprints
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with {index, similarity} pairs
        """
        similarities = self.bulk_similarity(query_fp, target_fps)
        
        # Filter by threshold
        if threshold > 0.0:
            valid_results = [(i, sim) for i, sim in enumerate(similarities) if sim >= threshold]
        else:
            valid_results = [(i, sim) for i, sim in enumerate(similarities)]
            
        # Sort by similarity (descending)
        valid_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k
        top_results = valid_results[:k]
        
        # Format results
        return [{'index': idx, 'similarity': sim} for idx, sim in top_results]


class FPSim2Engine(SimilarityEngine):
    """
    Similarity engine using FPSim2 for high-performance similarity search.
    """
    
    def __init__(self, db_file: Optional[str] = None):
        """
        Initialize FPSim2 engine.
        
        Args:
            db_file: Optional path to FPSim2 database file
        """
        try:
            import FPSim2
            from FPSim2 import FPSim2Engine as FPS2
            from FPSim2.io import create_db_file
            self.FPSim2 = FPSim2
            self.FPS2 = FPS2
            self.create_db_file = create_db_file
        except ImportError:
            raise ImportError("FPSim2 is not installed. Please install it with 'pip install FPSim2'.")
            
        self.db_file = db_file
        self.engine = None
        
        # Initialize engine if db_file is provided
        if db_file is not None and os.path.exists(db_file):
            self.engine = FPS2(db_file)
            
    def create_database(self, 
                       smiles_file: str, 
                       output_file: str, 
                       fp_type: str = 'Morgan',
                       fp_params: Dict[str, Any] = {'radius': 2, 'fpSize': 2048}) -> str:
        """
        Create a FPSim2 database file from a SMILES file.
        
        Args:
            smiles_file: Path to SMILES file
            output_file: Path to output database file
            fp_type: Fingerprint type ('Morgan', 'RDKit', etc.)
            fp_params: Fingerprint parameters
            
        Returns:
            Path to created database file
        """
        self.create_db_file(
            mols_source=smiles_file,
            filename=output_file,
            mol_format=None,  # Auto-detect
            fp_type=fp_type,
            fp_params=fp_params
        )
        
        # Update engine with new database
        self.db_file = output_file
        self.engine = self.FPS2(output_file)
        
        return output_file
        
    def set_db_file(self, db_file: str) -> None:
        """
        Set the database file for the engine.
        
        Args:
            db_file: Path to FPSim2 database file
        """
        if not os.path.exists(db_file):
            raise FileNotFoundError(f"FPSim2 database file not found: {db_file}")
            
        self.db_file = db_file
        self.engine = self.FPS2(db_file)
        
    def bulk_similarity(self, query_fp: Any, target_fps: List[Any]) -> List[float]:
        """
        Calculate similarities using FPSim2.
        
        Note: This is a simplified implementation that doesn't use FPSim2's optimized functions.
        In practice, you should use top_k_smiles or top_k_similarity methods for better performance.
        
        Args:
            query_fp: Query fingerprint
            target_fps: List of target fingerprints
            
        Returns:
            List of similarity scores
        """
        # FPSim2 doesn't have a direct bulk_similarity function for in-memory fingerprints,
        # so we use RDKit as a fallback for this case
        rdkit_engine = RDKitEngine()
        return rdkit_engine.bulk_similarity(query_fp, target_fps)
        
    def top_k_similarity(self, query_fp: Any, target_fps: List[Any], k: int, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find top-k most similar fingerprints using FPSim2.
        
        Note: This doesn't use FPSim2's optimized functions, which require a database file.
        In practice, use top_k_smiles for better performance.
        
        Args:
            query_fp: Query fingerprint
            target_fps: List of target fingerprints
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dictionaries with {index, similarity} pairs
        """
        # FPSim2 doesn't have a direct top_k function for in-memory fingerprints,
        # so we use RDKit as a fallback for this case
        rdkit_engine = RDKitEngine()
        return rdkit_engine.top_k_similarity(query_fp, target_fps, k, threshold)
        
    def top_k_smiles(self, query_smiles: str, k: int = 100, threshold: float = 0.7, n_workers: int = 1) -> List[Dict[str, Any]]:
        """
        Find top-k most similar molecules using FPSim2's optimized search.
        
        This method requires a database file to be set.
        
        Args:
            query_smiles: Query SMILES string
            k: Number of results to return
            threshold: Minimum similarity threshold
            n_workers: Number of worker threads
            
        Returns:
            List of dictionaries with search results
        """
        if self.engine is None:
            raise RuntimeError("FPSim2 engine not initialized. Set a database file first.")
            
        results = self.engine.similarity(query_smiles, threshold, n_workers)
        
        # Sort by similarity (descending) and take top-k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:k]
        
        # Format results
        return [{'index': int(idx), 'similarity': float(sim)} for idx, sim in top_results]


def get_engine(engine_type: str, **kwargs) -> SimilarityEngine:
    """
    Factory function to get a similarity engine.
    
    Args:
        engine_type: Type of engine ('rdkit' or 'fpsim2')
        **kwargs: Additional arguments for the engine
        
    Returns:
        Initialized similarity engine
    """
    engine_type = engine_type.lower()
    
    if engine_type == 'rdkit':
        return RDKitEngine()
    elif engine_type == 'fpsim2':
        db_file = kwargs.get('db_file')
        return FPSim2Engine(db_file)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}. Use 'rdkit' or 'fpsim2'.")