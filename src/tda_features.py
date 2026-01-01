import numpy as np
import pandas as pd
import gudhi
from typing import List, Optional, Tuple, Union

class TakensEmbedding:
    """
    Implements Time Delay Embedding (Takens' Theorem) to reconstruct 
    phase space from a single time series.
    """
    def __init__(self, dimension: int, delay: int):
        self.dimension = dimension
        self.delay = delay

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms a 1D time series into a (N - (dim-1)*delay) x dim point cloud.
        """
        n = len(data)
        if n < self.dimension * self.delay:
            raise ValueError("Data length is too short for the specified dimension and delay.")
        
        point_cloud = []
        # Number of points we can construct
        num_points = n - (self.dimension - 1) * self.delay
        
        for i in range(num_points):
            # Extract vector: [x[i], x[i+tau], ..., x[i+(m-1)*tau]]
            vector = [data[i + k * self.delay] for k in range(self.dimension)]
            point_cloud.append(vector)
            
        return np.array(point_cloud)

class TDAExtractor:
    """
    Extracts topological features from point clouds using Persistent Homology.
    """
    def __init__(self, max_dimension: int = 1, max_edge_length: float = float('inf')):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length

    def compute_persistence(self, point_cloud: np.ndarray) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Computes persistence features using Rips Complex.
        Returns a list of (dimension, (birth, death)) tuples.
        """
        if len(point_cloud) == 0:
            return []
            
        # RipsComplex construction
        # max_edge_length can be tuned or set to auto (inf)
        rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=self.max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension + 1)
        
        # Compute persistent homology
        diag = simplex_tree.persistence(min_persistence=0.0)
        return diag

    def get_persistence_entropy(self, diag: List[Tuple[int, Tuple[float, float]]]) -> np.ndarray:
        """
        Computes Persistent Entropy for each dimension up to max_dimension.
        Returns an array of shape (max_dimension + 1,).
        """
        entropies = np.zeros(self.max_dimension + 1)
        
        # Group by dimension
        by_dim = {d: [] for d in range(self.max_dimension + 1)}
        for dim, (birth, death) in diag:
            if dim <= self.max_dimension:
                if death == float('inf'):
                    continue # Skip infinite death for entropy or handle specifically
                by_dim[dim].append(death - birth)
        
        for d in range(self.max_dimension + 1):
            lifetimes = np.array(by_dim[d])
            if len(lifetimes) == 0:
                entropies[d] = 0.0
                continue
            
            total_lifetime = np.sum(lifetimes)
            if total_lifetime == 0:
                entropies[d] = 0.0
                continue
                
            probs = lifetimes / total_lifetime
            # Shannon entropy: -sum(p * log(p))
            entropies[d] = -np.sum(probs * np.log(probs + 1e-10))
            
        return entropies
    
    def get_betti_curves(self, diag, num_points=20) -> np.ndarray:
        """
        Simple discretization of Betti numbers along the filtration.
        (Simplified version of Landscapes/Images for vectorization)
        """
        # This is a placeholder for a more complex vectorization if needed
        # We generally want a fixed size vector for the RL agent.
        # Here we just return entropy which is 1 value per dimension.
        # A more robust one might be stats: [mean_lifetime, max_lifetime, entropy] per dim.
        pass

class FeatureProcessor:
    """
    Orchestrates the TDA pipeline: Data -> Embedding -> Persistence -> Features
    """
    def __init__(self, embedding_dim=3, embedding_delay=1, max_homology_dim=1):
        self.embedding = TakensEmbedding(embedding_dim, embedding_delay)
        self.tda = TDAExtractor(max_dimension=max_homology_dim)
        self.embedding_dim = embedding_dim
        self.embedding_delay = embedding_delay

    def process(self, time_series: np.ndarray) -> np.ndarray:
        """
        Returns a flat feature vector.
        """
        # 1. Embed
        # We need enough history. If time_series is the current window.
        try:
            point_cloud = self.embedding.transform(time_series)
        except ValueError:
            # Not enough data
            return np.zeros((self.tda.max_dimension + 1) * 3) # Return zero vector on failure

        # 2. Compute Persistence
        diag = self.tda.compute_persistence(point_cloud)
        
        # 3. Extract Features (Vectorization)
        # Feature 1: Persistent Entropy
        entropy = self.tda.get_persistence_entropy(diag)
        
        # Feature 2: Statistical summaries of lifetimes (Mean, Max)
        # For H0 and H1
        stats = []
        for d in range(self.tda.max_dimension + 1):
            lifetimes = [death - birth for dim, (birth, death) in diag if dim == d and death != float('inf')]
            if not lifetimes:
                stats.extend([0.0, 0.0])
            else:
                stats.extend([np.mean(lifetimes), np.max(lifetimes)])
                
        # Combine: Entropy + Mean + Max for each dim
        # Structure: [H0_Ent, H0_Mean, H0_Max, H1_Ent, H1_Mean, H1_Max]
        output_features = []
        for d in range(self.tda.max_dimension + 1):
            output_features.append(entropy[d])
            output_features.append(stats[d*2])     # Mean
            output_features.append(stats[d*2 + 1]) # Max
            
        return np.array(output_features, dtype=np.float32)

if __name__ == "__main__":
    # Test
    # Generate random walk
    data = np.cumsum(np.random.randn(100))
    processor = FeatureProcessor(embedding_dim=3, embedding_delay=2)
    feats = processor.process(data)
    print("Input shape:", data.shape)
    print("Features:", feats)
