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
from bblean import BitBirch
#from bblean.cluster_control import calculate_medoid
from bblean.similarity import (
    _jt_sim_arr_vec_packed,
    jt_most_dissimilar_packed,
    centroid_from_sum,
    jt_isim_medoid
)
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
        n_clusters: Optional[int] = None, 
        similarity_method: str = 'rdkit',
        threshold: float = 0.65, 
        branching_factor: int = 50
    ):
        """
        Initialize the IVF index.
        
        Args:
            n_clusters: Number of clusters to use (None = sqrt(n_samples), calculated during build)
            similarity_method: Method for similarity calculations ('rdkit' or 'fpsim2')
            threshold: Similarity threshold for BitBIRCH clustering (used only for tree building)
            branching_factor: Branching factor for BitBIRCH clustering
        """
        if n_clusters is not None and n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer or None")
        
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
        self.default_max_probe = None  # Will be set to sqrt(n_samples) during build
        
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
        
        # Calculate n_clusters if not provided (sqrt heuristic)
        if self.n_clusters is None:
            self.n_clusters = int(np.sqrt(n_samples))
            print(f"Using sqrt heuristic: n_clusters = {self.n_clusters}")
        
        # Set default max_probe to sqrt(n_samples)
        self.default_max_probe = int(np.sqrt(n_samples))
        
        # Always use k-clusters functionality since n_clusters is required
        print(f"Clustering {n_samples} fingerprints into exactly {self.n_clusters} clusters...")
        
        # Initialize BitBIRCH for clustering
        birch = BitBirch(threshold=self.threshold, branching_factor=self.branching_factor)
        birch.fit(fingerprints, input_is_packed=False)
        birch.global_clustering(method="kmeans-normalized", n_clusters=self.n_clusters, random_state=42)
        cluster_ids = birch.get_assignments()
        
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
                centroid_binary = centroid_from_sum(linear_sum, n_samples=cluster_size, pack=False)
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
                    idx, _ = jt_isim_medoid(cluster_fps_dense, input_is_packed=False, pack=False)
                    medoid_idx = member_indices[idx]
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
        global_clustering: str = 'kmeans-normalized',
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
        
        # Calculate n_clusters if not provided (sqrt heuristic)
        if self.n_clusters is None:
            self.n_clusters = int(np.sqrt(n_samples))
            print(f"Using sqrt heuristic: n_clusters = {self.n_clusters}")
        
        # Set default max_probe to sqrt(n_samples)
        self.default_max_probe = int(np.sqrt(n_samples))
        
        # Always use k-clusters functionality with custom global clustering
        print(f"Clustering {n_samples} fingerprints into exactly {self.n_clusters} clusters using {global_clustering}...")
        
        # Initialize BitBIRCH for clustering
        birch = BitBirch(threshold=self.threshold, branching_factor=self.branching_factor)
        birch.fit(fingerprints, input_is_packed=False)
        birch.global_clustering(method=global_clustering, n_clusters=self.n_clusters, **gc_kwargs)
        cluster_ids = birch.get_assignments()
        
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
                centroid_binary = centroid_from_sum(linear_sum, cluster_size)
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
                    idx, _ = jt_isim_medoid(cluster_fps_dense, input_is_packed=False, pack=False)
                    medoid_idx = member_indices[idx]
                    self.cluster_centroids_rdkit.append(self.fingerprints_rdkit[medoid_idx])
                else:
                    # Single member cluster
                    self.cluster_centroids_rdkit.append(self.fingerprints_rdkit[member_indices[0]])
            
        # Convert centroids to numpy array
        self.cluster_centroids = np.array(self.cluster_centroids)
        
        self.built = True
        print(f"IVF index built with {len(self.cluster_centroids)} clusters")
        

    # Dynamic Probe Selection (DPS)
    ''' The following idea is implemented:

1. Calculate similarity of query to all cluster centroids (using RDKit or FPSim2 as available)
2. Convert to probabilities using softmax (with temperature scaling to control sharpness)
3. Keep on taking top clusters until cumulative probability exceeds a threshold (e.g., 0.9) or we reach a max number of probes (n_probe)
4. Also ensure that we have a minimum number of 10*k molecules where k is the number of closest neighbors we want to return, to ensure good recall. If not, keep adding clusters until we have enough candidates.
(k can be found at the search function)
    '''
    def _select_clusters_dps(self, query_fp, k: int, max_probe: Optional[int] = None, prob_threshold: float = 0.6, temperature: float = 0.1, verbose: bool = False) -> List[int]:
        """
        Dynamically select clusters using probability-based approach.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            k: Number of results to return (used to determine minimum candidates)
            max_probe: Maximum number of clusters to probe (None = sqrt(n_samples))
            prob_threshold: Cumulative probability threshold for cluster selection
            temperature: Temperature for softmax (lower = sharper distribution)
            verbose: Print detailed selection info (default: False for performance)
            
        Returns:
            List of cluster IDs to probe
        """
        from rdkit import DataStructs
        
        # Use default max_probe if not provided
        if max_probe is None:
            max_probe = self.default_max_probe if self.default_max_probe is not None else 30
        import numpy as np
        
        # Step 1: Calculate similarity to all cluster centroids
        if (hasattr(self, 'cluster_centroids_rdkit') and 
            self.cluster_centroids_rdkit and 
            isinstance(query_fp, DataStructs.ExplicitBitVect)):
            # Direct RDKit calculation
            similarities = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, self.cluster_centroids_rdkit))
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
            
            similarities = np.array(DataStructs.BulkTanimotoSimilarity(query_bitvect, centroid_bitvects))
        
        # Step 2: Convert to probabilities using softmax with temperature scaling
        # Apply temperature scaling
        scaled_similarities = similarities / temperature
        # Softmax: exp(x) / sum(exp(x))
        exp_sims = np.exp(scaled_similarities)  # for numerical stability
        probabilities = exp_sims / np.sum(exp_sims)
        
        # Step 3: Sort clusters by probability (descending)
        sorted_indices = np.argsort(probabilities)[::-1]
        cluster_ids = list(self.cluster_members.keys())
        
        # Step 4: Select clusters based on cumulative probability threshold
        selected_clusters = []
        cumulative_prob = 0.0
        min_candidates = 5 * k
        total_candidates = 0
        
        for idx in sorted_indices:
            if len(selected_clusters) >= max_probe:
                break
                
            cluster_id = cluster_ids[idx]
            selected_clusters.append(cluster_id)
            cumulative_prob += probabilities[idx]
            if verbose:
                print(f'Cumulative probability after adding cluster {cluster_id}: {cumulative_prob:.4f} (prob of this cluster: {probabilities[idx]:.4f})')
            total_candidates += len(self.cluster_members[cluster_id])
            
            # Check if we've met both conditions:
            # 1. Cumulative probability exceeds threshold
            # 2. Have enough candidate molecules (10*k minimum)
            if cumulative_prob >= prob_threshold and total_candidates >= min_candidates:
                break
        
        # Ensure we always have at least enough candidates
        # Continue adding clusters if needed
        if total_candidates < min_candidates:
            if verbose:
                print(f"Total candidates after initial selection: {total_candidates}, which is less than minimum required {min_candidates}. Adding more clusters...")
            for idx in sorted_indices[len(selected_clusters):]:
                if len(selected_clusters) >= max_probe:
                    break
                    
                cluster_id = cluster_ids[idx]
                if cluster_id not in selected_clusters:
                    selected_clusters.append(cluster_id)
                    total_candidates += len(self.cluster_members[cluster_id])
                    if verbose:
                        print(f'Cumulative probability after adding cluster {cluster_id}: {cumulative_prob:.4f} (prob of this cluster: {probabilities[idx]:.4f})')
                    if total_candidates >= min_candidates:
                        break
        
        return selected_clusters

    
    def _find_nearest_clusters(self, query_fp, n_probe: int, verbose: bool = False) -> List[int]:
        """
        Find the n_probe nearest clusters to the query fingerprint.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            n_probe: Number of clusters to return
            verbose: Print timing details (default: False for performance)
            
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
        
        if verbose:
            print(f"  Cluster finding details:")
            print(f"    Centroid similarities: {centroid_sim_time*1000:.2f}ms (vs {len(self.cluster_centroids)} centroids)")
            print(f"    Sorting/mapping: {sort_time*1000:.2f}ms")
        
        return nearest_clusters
    
    def search(
        self, 
        query_fp, 
        k: int = 10, 
        n_probe: int = 1,
        threshold: float = 0.0,
        verbose: bool = False
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        Search for the k most similar fingerprints to the query.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            k: Number of results to return
            n_probe: Number of clusters to search
            threshold: Minimum similarity threshold (0.0 means no threshold)
            verbose: Print timing breakdown (default: False for performance)
            
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
        nearest_clusters = self._find_nearest_clusters(query_fp, n_probe, verbose=verbose)
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
        
        # Print timing breakdown
        if verbose:
            total_time = cluster_time + gather_time + sim_time + post_time
            print(f"IVF Search timing breakdown:")
            print(f"  Find clusters: {cluster_time*1000:.2f}ms ({cluster_time/total_time*100:.1f}%)")
            print(f"  Gather candidates: {gather_time*1000:.2f}ms ({gather_time/total_time*100:.1f}%)")
            print(f"  Similarity calc: {sim_time*1000:.2f}ms ({sim_time/total_time*100:.1f}%)")
            print(f"  Post-processing: {post_time*1000:.2f}ms ({post_time/total_time*100:.1f}%)")
            print(f"  Total: {total_time*1000:.2f}ms, candidates: {len(candidate_indices)}")
            
        return results
    
    def search_dps(
        self,
        query_fp,
        k: int = 10,
        max_probe: Optional[int] = None,
        prob_threshold: float = 0.1,
        temperature: float = 0.1,
        threshold: float = 0.0,
        verbose: bool = False
    ) -> List[Dict[str, Union[int, float, str]]]:
        """
        Search using Dynamic Probe Selection (DPS).
        
        The number of clusters probed varies per query based on the probability
        distribution over clusters, making it adaptive to query characteristics.
        
        Args:
            query_fp: Query fingerprint (numpy array or RDKit ExplicitBitVect)
            k: Number of results to return
            max_probe: Maximum number of clusters to probe (None = sqrt(n_samples))
            prob_threshold: Cumulative probability threshold for cluster selection
            temperature: Softmax temperature (lower = sharper distribution)
            threshold: Minimum similarity threshold (0.0 means no threshold)
            verbose: Print timing breakdown (default: False for performance)
            
        Returns:
            List of dictionaries containing search results, each with:
                - 'index': Index of the fingerprint
                - 'similarity': Tanimoto similarity to query
                - 'smiles': SMILES string (if available)
                - 'n_probes': Number of clusters probed for this query
        """
        import time
        
        if not self.built:
            raise RuntimeError("Index has not been built. Call build_index first.")
        
        # Select clusters using DPS (query-adaptive)
        t1 = time.time()
        selected_clusters = self._select_clusters_dps(
            query_fp,
            k=k,
            max_probe=max_probe,
            prob_threshold=prob_threshold,
            temperature=temperature,
            verbose=verbose
        )
        cluster_time = time.time() - t1
        n_probes = len(selected_clusters)
        
        # Get indices of fingerprints in the selected clusters
        t2 = time.time()
        candidate_indices = []
        for cluster_id in selected_clusters:
            candidate_indices.extend(self.cluster_members[cluster_id])
        gather_time = time.time() - t2
        
        # Calculate similarities
        t3 = time.time()
        if self.similarity_method == 'rdkit':
            from rdkit import DataStructs
            
            # Use RDKit fingerprints directly if available
            if (hasattr(self, 'fingerprints_rdkit') and 
                self.fingerprints_rdkit and 
                isinstance(query_fp, DataStructs.ExplicitBitVect)):
                candidate_rdkit_fps = [self.fingerprints_rdkit[i] for i in candidate_indices]
                similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, candidate_rdkit_fps))
            else:
                # Fallback conversion
                if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                    query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                    for i, val in enumerate(query_fp):
                        if val == 1:
                            query_bitvect.SetBit(i)
                else:
                    query_bitvect = query_fp
                
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
                
                similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, candidate_bitvects))
        
        elif self.similarity_method == 'fpsim2':
            from rdkit import DataStructs
            
            if not isinstance(query_fp, DataStructs.ExplicitBitVect):
                query_bitvect = DataStructs.ExplicitBitVect(len(query_fp))
                for i, val in enumerate(query_fp):
                    if val == 1:
                        query_bitvect.SetBit(i)
            else:
                query_bitvect = query_fp
            
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
            
            similarities = list(DataStructs.BulkTanimotoSimilarity(query_bitvect, candidate_bitvects))
        
        sim_time = time.time() - t3
        
        # Apply threshold and sort
        t4 = time.time()
        if threshold > 0.0:
            valid_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
            similarities = [similarities[i] for i in valid_indices]
            candidate_indices = [candidate_indices[i] for i in valid_indices]
        
        sorted_indices = np.argsort(similarities)[::-1][:k]
        
        # Prepare results
        results = []
        for idx in sorted_indices:
            result = {
                'index': candidate_indices[idx],
                'similarity': similarities[idx],
                'n_probes': n_probes  # Include number of probes used for this query
            }
            
            if self.smiles is not None:
                result['smiles'] = self.smiles[candidate_indices[idx]]
            
            results.append(result)
        
        post_time = time.time() - t4
        
        # Print timing breakdown
        if verbose:
            total_time = cluster_time + gather_time + sim_time + post_time
            print(f"DPS Search timing breakdown:")
            print(f"  DPS cluster selection: {cluster_time*1000:.2f}ms ({cluster_time/total_time*100:.1f}%)")
            print(f"  Gather candidates: {gather_time*1000:.2f}ms ({gather_time/total_time*100:.1f}%)")
            print(f"  Similarity calc: {sim_time*1000:.2f}ms ({sim_time/total_time*100:.1f}%)")
            print(f"  Post-processing: {post_time*1000:.2f}ms ({post_time/total_time*100:.1f}%)")
            print(f"  Total: {total_time*1000:.2f}ms, probes: {n_probes}, candidates: {len(candidate_indices)}")
        
        return results