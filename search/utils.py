"""
Utility functions for the IVF search implementation.

This module provides helper functions for loading data, converting
between different fingerprint formats, and other common operations.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any

def load_smiles_file(file_path: str) -> List[str]:
    """
    Load SMILES from a file.
    
    Supports both .smi and .csv formats:
    - .smi: Space-separated format (SMILES in first column)
    - .csv: Comma-separated format (SMILES in first column)
    
    Args:
        file_path: Path to SMILES file
        
    Returns:
        List of SMILES strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r') as f:
        # Handle different SMILES file formats
        lines = f.readlines()
        smiles = []
        
        # Detect file format based on extension
        is_csv = file_path.lower().endswith('.csv')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract SMILES from the line based on format
            if is_csv:
                # CSV format: split by comma and take first column
                parts = line.split(',')
                if parts:
                    smiles.append(parts[0])  # First column is SMILES
            else:
                # SMI format: split by whitespace and take first column
                parts = line.split()
                if parts:
                    smiles.append(parts[0])  # First column is SMILES
                
    return smiles

def generate_fingerprints(
    smiles: List[str], 
    fp_type: str = 'morgan', 
    fp_params: Dict[str, Any] = None,
    sparse: bool = False
) -> Union[np.ndarray, Any]:
    """
    Generate fingerprints from SMILES.
    
    Args:
        smiles: List of SMILES strings
        fp_type: Type of fingerprint ('morgan', 'rdkit', etc.)
        fp_params: Parameters for fingerprint generation
        sparse: Whether to return sparse matrix (memory efficient) or dense numpy array (default: False)
        
    Returns:
        Sparse matrix (if sparse=True) or NumPy array of fingerprints
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from scipy import sparse as sp
    
    if fp_params is None:
        if fp_type.lower() == 'morgan':
            fp_params = {'radius': 2, 'nBits': 2048}
        elif fp_type.lower() == 'rdkit':
            fp_params = {'fpSize': 2048}
        else:
            fp_params = {}
    
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]  # Filter out None values
    
    # Determine fingerprint size
    fp_size = fp_params.get('nBits', fp_params.get('fpSize', 2048))
    
    if sparse:
        # Build sparse matrix directly without creating dense arrays
        rows, cols = [], []
        
        if fp_type.lower() == 'morgan':
            # Use modern MorganGenerator
            from rdkit.Chem import rdFingerprintGenerator
            mgen = rdFingerprintGenerator.GetMorganGenerator(
                radius=fp_params.get('radius', 2),
                fpSize=fp_params.get('nBits', 2048)
            )
            for mol_idx, mol in enumerate(mols):
                fp = mgen.GetFingerprint(mol)
                # Get the indices of set bits directly from RDKit
                on_bits = fp.GetOnBits()
                rows.extend([mol_idx] * len(on_bits))
                cols.extend(on_bits)
                
        elif fp_type.lower() == 'rdkit':
            # Use modern RDKitFPGenerator
            from rdkit.Chem import rdFingerprintGenerator
            rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=fp_params.get('fpSize', 2048)
            )
            for mol_idx, mol in enumerate(mols):
                fp = rdgen.GetFingerprint(mol)
                # Get the indices of set bits directly from RDKit
                on_bits = fp.GetOnBits()
                rows.extend([mol_idx] * len(on_bits))
                cols.extend(on_bits)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")
        
        # Create sparse matrix - much more memory efficient!
        data = np.ones(len(rows), dtype=np.uint8)  # All set bits are 1
        fingerprints = sp.csr_matrix((data, (rows, cols)), shape=(len(mols), fp_size), dtype=np.uint8)
    else:
        # Original dense array approach for backward compatibility
        fingerprints = []
        
        if fp_type.lower() == 'morgan':
            # Use modern MorganGenerator
            from rdkit.Chem import rdFingerprintGenerator
            mgen = rdFingerprintGenerator.GetMorganGenerator(
                radius=fp_params.get('radius', 2),
                fpSize=fp_params.get('nBits', 2048)
            )
            fingerprints = np.array([
                np.array(mgen.GetFingerprint(mol)) 
                for mol in mols
            ])
        elif fp_type.lower() == 'rdkit':
            # Use modern RDKitFPGenerator
            from rdkit.Chem import rdFingerprintGenerator
            rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(
                fpSize=fp_params.get('fpSize', 2048)
            )
            fingerprints = np.array([
                np.array(rdgen.GetFingerprint(mol))
                for mol in mols
            ])
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")
        
    return fingerprints

def load_fingerprints(
    file_path: str,
    fp_type: str = 'morgan',
    fp_params: Dict[str, Any] = None,
    sparse: bool = False
) -> Tuple[Union[np.ndarray, Any], List[str]]:
    """
    Load SMILES and generate fingerprints.
    
    Args:
        file_path: Path to SMILES file
        fp_type: Type of fingerprint ('morgan', 'rdkit', etc.)
        fp_params: Parameters for fingerprint generation
        sparse: Whether to return sparse matrix (memory efficient) or dense numpy array (default: False)
        
    Returns:
        Tuple of (fingerprints, smiles)
    """
    smiles = load_smiles_file(file_path)
    fingerprints = generate_fingerprints(smiles, fp_type, fp_params, sparse=sparse)
    
    return fingerprints, smiles

def create_fpsim2_db(
    smiles_file: str,
    output_file: str,
    fp_type: str = 'Morgan',
    fp_params: Dict[str, Any] = None
) -> str:
    """
    Create FPSim2 database from SMILES file.
    
    Args:
        smiles_file: Path to SMILES file
        output_file: Path to output database file
        fp_type: Fingerprint type ('Morgan', 'RDKit', etc.)
        fp_params: Parameters for fingerprint generation
        
    Returns:
        Path to created database file
    """
    try:
        from FPSim2.io import create_db_file
    except ImportError:
        raise ImportError("FPSim2 not installed. Install with 'pip install FPSim2'")
    
    if fp_params is None:
        if fp_type == 'Morgan':
            fp_params = {'radius': 2, 'fpSize': 2048}
        else:
            fp_params = {'fpSize': 2048}
    
    create_db_file(
        mols_source=smiles_file,
        filename=output_file,
        mol_format=None,  # Auto-detect
        fp_type=fp_type,
        fp_params=fp_params
    )
    
    return output_file