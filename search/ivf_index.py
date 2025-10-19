"""
IVF (Inverted File) search index implementation using BitBIRCH clustering.

This module provides an efficient search index for chemical fingerprints
by utilizing BitBIRCH clustering to partition the search space and enable
fast approximate nearest neighbor search.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Union, Optional

# Import BitBIRCH for clustering
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bblean.bitbirch import BitBirch, set_merge, calc_centroid
from bblean.cluster_control import calculate_medoid, calculate_comp_sim
from sklearn.cluster import KMeans, AgglomerativeClustering


class IVFIndex:
    """
    Inverted File (IVF) index for efficient similarity search of chemical fingerprints.
    
    The index uses BitBIRCH clustering to partition fingerprints into clusters,
    then at query time, only the most relevant clusters are searched, providing
    a significant speedup over exhaustive search.
    
    Attributes:
        n_clusters (int): Number of clusters to use. If None, uses sqrt(n_samples)
        similarity_method (str): Method to use for similarity calculations ('rdkit' or 'fpsim2')
        threshold (float): Similarity threshold for BitBIRCH clustering
        branching_factor (int): Branching factor for BitBIRCH clustering
        cluster_centroids (np.ndarray): Centroids of each cluster
        cluster_members (Dict[int, List[int]]): Mapping of cluster IDs to member fingerprint indices
        fingerprints (np.ndarray): Stored fingerprints for similarity search
        smiles (List[str]): Optional SMILES strings corresponding to fingerprints
        built (bool): Whether the index has been built
    """
    
    def __init__(
        self, 
        n_clusters: int, 
        similarity_method: str = 'rdkit',
        threshold: float = 0.7, 
        branching_factor: int = 50
    ):
        """
        Initialize the IVF index.
        
        Args:
            n_clusters: Number of clusters to use (required)
            similarity_method: Method for similarity calculations ('rdkit' or 'fpsim2')
            threshold: Similarity threshold for BitBIRCH clustering (used only for tree building)
            branching_factor: Branching factor for BitBIRCH clustering
        """
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        
        self.n_clusters = n_clusters
        self.similarity_method = similarity_method.lower()
        self.threshold = threshold
        self.branching_factor = branching_factor
        
        # Will be populated during build_index
        self.cluster_centroids = None
        self.cluster_centroids_rdkit = None  # RDKit format for performance
        self.cluster_members = {}
        self.fingerprints = None
        self.fingerprints_rdkit = None  # RDKit format for performance
        self.smiles = None
        self.built = False
        
        # Validate similarity method
        if self.similarity_method not in ['rdkit', 'fpsim2']:
            raise ValueError("similarity_method must be 'rdkit' or 'fpsim2'")
        
    def build_index(self, fingerprints: np.ndarray, smiles: Optional[List[str]] = None, fingerprints_rdkit: Optional[List] = None) -> None:
        """
        Build the IVF index by clustering fingerprints using BitBIRCH.
        
        Args:
            fingerprints: Binary fingerprints of shape (n_samples, n_features)
            smiles: Optional list of SMILES strings corresponding to fingerprints
            fingerprints_rdkit: Optional list of RDKit ExplicitBitVect objects for performance
        """
        n_samples = fingerprints.shape[0]
        
        # Store fingerprints and smiles for later use
        self.fingerprints = fingerprints
        self.fingerprints_rdkit = fingerprints_rdkit
        self.smiles = smiles
        
        # Always use k-clusters functionality since n_clusters is required
        print(f"Clustering {n_samples} fingerprints into exactly {self.n_clusters} clusters...")
        
        # Initialize BitBIRCH for clustering
        set_merge('diameter')  # Using diameter merge criterion
        birch = BitBirch(threshold=self.threshold, branching_factor=self.branching_factor)
        
        # Use the k-clusters functionality (n_clusters is always specified)
        birch.fit_with_k_clusters(fingerprints, n_clusters=self.n_clusters, global_clustering='kmeans', random_state=42)
        cluster_ids = birch.get_final_labels()
        
        # Extract cluster information
        unique_clusters = np.unique(cluster_ids)
        
        # Validate cluster assignments
        if len(cluster_ids) != n_samples:
            raise ValueError(f"Cluster assignment length {len(cluster_ids)} doesn't match sample count {n_samples}")
        
        # Ensure all molecules are assigned to clusters
        unassigned = np.where(cluster_ids == -1)[0]
        if len(unassigned) > 0:
            print(f"Warning: {len(unassigned)} molecules were not assigned to clusters, assigning to new clusters")
            # Assign unassigned molecules to their own clusters
            next_cluster_id = max(unique_clusters) + 1 if len(unique_clusters) > 0 else 0
            for idx in unassigned:
                cluster_ids[idx] = next_cluster_id
                next_cluster_id += 1
            unique_clusters = np.unique(cluster_ids)
        
        # Calculate centroids and organize members
        print(f"Found {len(unique_clusters)} unique clusters")
        self.cluster_centroids = []
        self.cluster_members = {}
        
        # Get centroids and member indices for each cluster
        self.cluster_centroids_rdkit = []
        for cluster_id in unique_clusters:
            # Get indices of fingerprints in this cluster
            member_indices = np.where(cluster_ids == cluster_id)[0]
            self.cluster_members[cluster_id] = member_indices
            
            # Get fingerprints in this cluster
            cluster_fps = fingerprints[member_indices]
            
            # Calculate centroid using BitBIRCH's method
            cluster_size = cluster_fps.shape[0] if hasattr(cluster_fps, 'shape') else len(cluster_fps)
            if cluster_size > 1:
                # Use BitBIRCH's calc_centroid function: threshold at 0.5 for binary centroids
                linear_sum = np.sum(cluster_fps, axis=0)
                # Handle sparse matrix sum result
                if hasattr(linear_sum, 'A1'):  # sparse matrix result
                    linear_sum = linear_sum.A1  # convert to 1D array
                centroid_binary = calc_centroid(linear_sum, cluster_size)
                self.cluster_centroids.append(centroid_binary)
            else:
                # Single member cluster - use the fingerprint itself
                if hasattr(cluster_fps, 'toarray'):  # sparse matrix
                    self.cluster_centroids.append(cluster_fps[0].toarray().flatten())
                else:
                    self.cluster_centroids.append(cluster_fps[0])
            
            # For RDKit format, use medoid (most representative fingerprint) using BitBIRCH's method
            if self.fingerprints_rdkit is not None:
                if len(member_indices) > 1:
                    # Convert sparse to dense for medoid calculation if needed
                    if hasattr(cluster_fps, 'toarray'):
                        cluster_fps_dense = cluster_fps.toarray()
                    else:
                        cluster_fps_dense = cluster_fps
                    # Use BitBIRCH's calculate_medoid function
                    medoid_local_idx = calculate_medoid(cluster_fps_dense)
                    medoid_idx = member_indices[medoid_local_idx]
                    self.cluster_centroids_rdkit.append(self.fingerprints_rdkit[medoid_idx])
                else:
                    # Single member cluster
                    self.cluster_centroids_rdkit.append(self.fingerprints_rdkit[member_indices[0]])
            
        # Convert centroids to numpy array
        self.cluster_centroids = np.array(self.cluster_centroids)
        
        self.built = True
        print(f"IVF index built with {len(self.cluster_centroids)} clusters")

    def build_index_with_global_clustering(
        self, 
        fingerprints: np.ndarray,
        smiles: Optional[List[str]] = None, 
        fingerprints_rdkit: Optional[List] = None,
        global_clustering: str = 'kmeans',
        **gc_kwargs
    ) -> None:
        """
        Build the IVF index using specified global clustering algorithm.
        
        Args:
            fingerprints: Binary fingerprints of shape (n_samples, n_features)
            smiles: Optional list of SMILES strings corresponding to fingerprints
            fingerprints_rdkit: Optional list of RDKit ExplicitBitVect objects for performance
            global_clustering: Algorithm for Phase 3 clustering ('kmeans' or 'agglomerative')
            gc_kwargs: Additional arguments for the global clustering algorithm
        """
        n_samples = fingerprints.shape[0]
        
        # Store fingerprints and smiles for later use
        self.fingerprints = fingerprints
        self.fingerprints_rdkit = fingerprints_rdkit
        self.smiles = smiles
        
        # Always use k-clusters functionality with custom global clustering
        print(f"Clustering {n_samples} fingerprints into exactly {self.n_clusters} clusters using {global_clustering}...")
        
        # Initialize BitBIRCH for clustering
        set_merge('diameter')  # Using diameter merge criterion
        birch = BitBirch(threshold=self.threshold, branching_factor=self.branching_factor)
        
        # Use the k-clusters functionality with specified algorithm
        birch.fit_with_k_clusters(fingerprints, n_clusters=self.n_clusters, global_clustering=global_clustering, **gc_kwargs)
        cluster_ids = birch.get_final_labels()
        
        # Extract cluster information
        unique_clusters = np.unique(cluster_ids)
        
        # Validate cluster assignments
        if len(cluster_ids) != n_samples:
            raise ValueError(f"Cluster assignment length {len(cluster_ids)} doesn't match sample count {n_samples}")
        
        # Ensure all molecules are assigned to clusters
        unassigned = np.where(cluster_ids == -1)[0]
        if len(unassigned) > 0:
            raise RuntimeError(f"Fatal: {len(unassigned)} molecules were not assigned to clusters in k-clusters mode")
        
        # Calculate centroids and organize members
        print(f"Found {len(unique_clusters)} unique clusters")
        self.cluster_centroids = []
        self.cluster_members = {}
        
        # Get centroids and member indices for each cluster
        self.cluster_centroids_rdkit = []
        for cluster_id in unique_clusters:
            # Get indices of fingerprints in this cluster
            member_indices = np.where(cluster_ids == cluster_id)[0]
            self.cluster_members[cluster_id] = member_indices
            
            # Get fingerprints in this cluster
            cluster_fps = fingerprints[member_indices]
            
            # Calculate centroid using BitBIRCH's method
            cluster_size = cluster_fps.shape[0] if hasattr(cluster_fps, 'shape') else len(cluster_fps)
            if cluster_size > 1:
                # Use BitBIRCH's calc_centroid function: threshold at 0.5 for binary centroids
                linear_sum = np.sum(cluster_fps, axis=0)
                # Handle sparse matrix sum result
                if hasattr(linear_sum, 'A1'):  # sparse matrix result
                    linear_sum = linear_sum.A1  # convert to 1D array
                centroid_binary = calc_centroid(linear_sum, cluster_size)
                self.cluster_centroids.append(centroid_binary)
            else:
                # Single member cluster - use the fingerprint itself
                if hasattr(cluster_fps, 'toarray'):  # sparse matrix
                    self.cluster_centroids.append(cluster_fps[0].toarray().flatten())
                else:
                    self.cluster_centroids.append(cluster_fps[0])
            
            # For RDKit format, use medoid (most representative fingerprint) using BitBIRCH's method
            if self.fingerprints_rdkit is not None:
                if len(member_indices) > 1:
                    # Convert sparse to dense for medoid calculation if needed
                    if hasattr(cluster_fps, 'toarray'):
                        cluster_fps_dense = cluster_fps.toarray()
                    else:
                        cluster_fps_dense = cluster_fps
                    # Use BitBIRCH's calculate_medoid function
                    medoid_local_idx = calculate_medoid(cluster_fps_dense)
                    medoid_idx = member_indices[medoid_local_idx]
                    self.cluster_centroids_rdkit.append(self.fingerprints_rdkit[medoid_idx])
                else:
                    # Single member cluster
                    self.cluster_centroids_rdkit.append(self.fingerprints_rdkit[member_indices[0]])
            
        # Convert centroids to numpy array
        self.cluster_centroids = np.array(self.cluster_centroids)
        
        self.built = True
        print(f"IVF index built with {len(self.cluster_centroids)} clusters")
        
    def _find_nearest_clusters(self, query_fp, n_probe: int) -> List[int]:
        """
        Find the n_probe nearest clusters to the query fingerprint.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            n_probe: Number of clusters to return
            
        Returns:
            List of cluster IDs, sorted by similarity to query
        """
        from rdkit import DataStructs
        import numpy as np
        import time
        
        # Limit n_probe to available clusters
        n_probe = min(n_probe, len(self.cluster_centroids))
        
        t1 = time.time()
        # Use RDKit fingerprints directly if available
        if (hasattr(self, 'cluster_centroids_rdkit') and 
            self.cluster_centroids_rdkit and 
            isinstance(query_fp, DataStructs.ExplicitBitVect)):
            # Direct RDKit calculation - no conversion needed!
            similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, self.cluster_centroids_rdkit))
        else:
            # Fall back to conversion if needed
            if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                for i, val in enumerate(query_fp):
                    if val == 1:
                        query_bitvect.SetBit(i)
            else:
                query_bitvect = query_fp
                
            # Convert centroids to ExplicitBitVect list if needed
            centroid_bitvects = []
            for centroid in self.cluster_centroids:
                if not isinstance(centroid, DataStructs.ExplicitBitVect):
                    bitvect = DataStructs.ExplicitBitVect(len(centroid))
                    for i, val in enumerate(centroid):
                        if val == 1:
                            bitvect.SetBit(i)
                    centroid_bitvects.append(bitvect)
                else:
                    centroid_bitvects.append(centroid)
            
            # Calculate similarities to all centroids
            similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, centroid_bitvects))
        
        centroid_sim_time = time.time() - t1
        
        t2 = time.time()
        # Get indices of top n_probe most similar centroids
        top_indices = np.argsort(similarities)[-n_probe:][::-1]  # Sort descending
        
        # Map index to cluster ID - fix the mapping!
        cluster_ids = list(self.cluster_members.keys())
        nearest_clusters = [cluster_ids[idx] for idx in top_indices]
        
        sort_time = time.time() - t2
        
        print(f"  Cluster finding details:")
        print(f"    Centroid similarities: {centroid_sim_time*1000:.2f}ms (vs {len(self.cluster_centroids)} centroids)")
        print(f"    Sorting/mapping: {sort_time*1000:.2f}ms")
        
        return nearest_clusters
    
    def search(
        self, 
        query_fp, 
        k: int = 10, 
        n_probe: int = 1,
        threshold: float = 0.0
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        Search for the k most similar fingerprints to the query.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            k: Number of results to return
            n_probe: Number of clusters to search
            threshold: Minimum similarity threshold (0.0 means no threshold)
            
        Returns:
            List of dictionaries containing search results, each with:
                - 'index': Index of the fingerprint
                - 'similarity': Tanimoto similarity to query
                - 'smiles': SMILES string (if available)
        """
        import time
        
        if not self.built:
            raise RuntimeError("Index has not been built. Call build_index first.")
            
        # Find nearest clusters
        t1 = time.time()
        nearest_clusters = self._find_nearest_clusters(query_fp, n_probe)
        cluster_time = time.time() - t1
        
        # Get indices of fingerprints in the selected clusters
        t2 = time.time()
        candidate_indices = []
        for cluster_id in nearest_clusters:
            candidate_indices.extend(self.cluster_members[cluster_id])
        gather_time = time.time() - t2
            
        # Calculate similarities based on method and available formats
        t3 = time.time()
        if self.similarity_method == 'rdkit':
            from rdkit import DataStructs
            
            # Use RDKit fingerprints directly if available
            if (hasattr(self, 'fingerprints_rdkit') and 
                self.fingerprints_rdkit and 
                isinstance(query_fp, DataStructs.ExplicitBitVect)):
                # Direct RDKit calculation - no conversion needed!
                candidate_rdkit_fps = [self.fingerprints_rdkit[i] for i in candidate_indices]
                similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, candidate_rdkit_fps))
            else:
                # Fall back to conversion if needed
                if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                    query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                    for i, val in enumerate(query_fp):
                        if val == 1:
                            query_bitvect.SetBit(i)
                else:
                    query_bitvect = query_fp
                    
                # Get candidate fingerprints and convert if needed
                candidate_fps = self.fingerprints[candidate_indices]
                candidate_bitvects = []
                for fp in candidate_fps:
                    if not isinstance(fp, DataStructs.ExplicitBitVect):
                        bitvect = DataStructs.ExplicitBitVect(len(fp))
                        for i, val in enumerate(fp):
                            if val == 1:
                                bitvect.SetBit(i)
                        candidate_bitvects.append(bitvect)
                    else:
                        candidate_bitvects.append(fp)
                        
                # Calculate similarities
                similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, candidate_bitvects))
            
        elif self.similarity_method == 'fpsim2':
            # For now, use a placeholder implementation; will be replaced with actual FPSim2
            # In the full implementation, this would use FPSim2's optimized similarity calculation
            from rdkit import DataStructs
            
            # Similar optimization can be applied here for FPSim2
            if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                for i, val in enumerate(query_fp):
                    if val == 1:
                        query_bitvect.SetBit(i)
            else:
                query_bitvect = query_fp
                
            # Get candidate fingerprints and convert if needed
            candidate_fps = self.fingerprints[candidate_indices]
            candidate_bitvects = []
            for fp in candidate_fps:
                if not isinstance(fp, DataStructs.ExplicitBitVect):
                    bitvect = DataStructs.ExplicitBitVect(len(fp))
                    for i, val in enumerate(fp):
                        if val == 1:
                            bitvect.SetBit(i)
                    candidate_bitvects.append(bitvect)
                else:
                    candidate_bitvects.append(fp)
                    
            # Calculate similarities
            similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, candidate_bitvects))
        
        sim_time = time.time() - t3
        
        # Apply threshold filter
        t4 = time.time()
        if threshold > 0.0:
            valid_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
            similarities = [similarities[i] for i in valid_indices]
            candidate_indices = [candidate_indices[i] for i in valid_indices]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1][:k]
        
        # Prepare results
        results = []
        for idx in sorted_indices:
            result = {
                'index': candidate_indices[idx],
                'similarity': similarities[idx],
            }
            
            # Add SMILES if available
            if self.smiles is not None:
                result['smiles'] = self.smiles[candidate_indices[idx]]
                
            results.append(result)
        
        post_time = time.time() - t4
        
        # Print timing breakdown (guard against zero total_time)
        total_time = cluster_time + gather_time + sim_time + post_time
        print(f"IVF Search timing breakdown:")
        if total_time > 0:
            pct_cluster = cluster_time / total_time * 100.0
            pct_gather = gather_time / total_time * 100.0
            pct_sim = sim_time / total_time * 100.0
            pct_post = post_time / total_time * 100.0
        else:
            pct_cluster = pct_gather = pct_sim = pct_post = 0.0

        print(f"  Find clusters: {cluster_time*1000:.2f}ms ({pct_cluster:.1f}%)")
        print(f"  Gather candidates: {gather_time*1000:.2f}ms ({pct_gather:.1f}%)")
        print(f"  Similarity calc: {sim_time*1000:.2f}ms ({pct_sim:.1f}%)")
        print(f"  Post-processing: {post_time*1000:.2f}ms ({pct_post:.1f}%)")
        print(f"  Total: {total_time*1000:.2f}ms, candidates: {len(candidate_indices)}")
            
        return results