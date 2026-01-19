"""
FAISS Index Manager for Fast Face Recognition
Uses HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search
for efficient face feature matching using cosine similarity.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

# Configuration
INDEX_FILE = "faiss_visitor_index.bin"
METADATA_FILE = "faiss_visitor_metadata.pkl"
DEFAULT_DIMENSION = 512  # Sface feature dimension
DEFAULT_M = 32  # HNSW parameter: number of bi-directional links
DEFAULT_EF_CONSTRUCTION = 200  # HNSW parameter: size of dynamic candidate list
DEFAULT_EF_SEARCH = 50  # HNSW parameter: number of nearest neighbors to explore


class FAISSIndexManager:
    """
    Manages FAISS HNSW index for fast face recognition using cosine similarity.
    """
    
    def __init__(self, 
                 dimension: int = DEFAULT_DIMENSION,
                 m: int = DEFAULT_M,
                 ef_construction: int = DEFAULT_EF_CONSTRUCTION,
                 ef_search: int = DEFAULT_EF_SEARCH,
                 index_dir: str = "models"):
        """
        Initialize FAISS index manager.
        
        Args:
            dimension: Feature vector dimension (default: 512 for Sface)
            m: HNSW parameter - number of bi-directional links (default: 32)
            ef_construction: HNSW parameter - size of dynamic candidate list during construction (default: 200)
            ef_search: HNSW parameter - number of nearest neighbors to explore during search (default: 50)
            index_dir: Directory to store index files
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[int, Dict] = {}  # Maps index position to visitor info
        self.visitor_id_to_index: Dict[str, int] = {}  # Maps visitor_id to index position
        self.next_index = 0
        
        # Load existing index if available
        self._load_index()
    
    def _create_index(self) -> faiss.Index:
        """Create a new HNSW index for inner product (cosine similarity)."""
        # Use IndexHNSWFlat with inner product metric for cosine similarity
        # For normalized vectors, inner product = cosine similarity
        # We'll use IndexFlatIP wrapped or IndexHNSWFlat with proper metric
        # IndexHNSWFlat uses L2 by default, but we can use inner product with normalized vectors
        # Actually, for cosine similarity with normalized vectors, we use IndexHNSWFlat
        # and convert L2 distance to cosine similarity in search
        index = faiss.IndexHNSWFlat(self.dimension, self.m)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index
    
    def _load_index(self) -> bool:
        """Load index and metadata from disk."""
        index_path = self.index_dir / INDEX_FILE
        metadata_path = self.index_dir / METADATA_FILE
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                self.index.hnsw.efSearch = self.ef_search
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', {})
                    self.visitor_id_to_index = data.get('visitor_id_to_index', {})
                    self.next_index = data.get('next_index', 0)
                
                print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
                return True
            except Exception as e:
                print(f"⚠ Error loading FAISS index: {e}. Creating new index.")
                self.index = self._create_index()
                return False
        else:
            # Create new index
            self.index = self._create_index()
            return False
    
    def _save_index(self) -> bool:
        """Save index and metadata to disk."""
        try:
            index_path = self.index_dir / INDEX_FILE
            metadata_path = self.index_dir / METADATA_FILE
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            data = {
                'metadata': self.metadata,
                'visitor_id_to_index': self.visitor_id_to_index,
                'next_index': self.next_index
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
        except Exception as e:
            print(f"⚠ Error saving FAISS index: {e}")
            return False
    
    def add_visitor(self, visitor_id: str, feature: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Add a visitor's face feature to the index.
        
        Args:
            visitor_id: Unique visitor identifier
            feature: Face feature vector (512-dim numpy array)
            metadata: Optional metadata (e.g., name, image_path)
        
        Returns:
            True if successful, False otherwise
        """
        if not FAISS_AVAILABLE:
            return False
        
        if visitor_id in self.visitor_id_to_index:
            # Visitor already exists, skip or update
            print(f"Visitor {visitor_id} already in index, skipping")
            return False
        
        try:
            # Normalize feature vector for cosine similarity
            feature_norm = feature / np.linalg.norm(feature)
            feature_norm = feature_norm.astype('float32').reshape(1, -1)
            
            # Add to index
            self.index.add(feature_norm)
            
            # Store metadata
            idx = self.next_index
            self.metadata[idx] = {
                'visitor_id': visitor_id,
                **(metadata or {})
            }
            self.visitor_id_to_index[visitor_id] = idx
            self.next_index += 1
            
            return True
        except Exception as e:
            print(f"Error adding visitor to FAISS index: {e}")
            return False
    
    def add_visitors_batch(self, visitors: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        """
        Add multiple visitors to the index in batch.
        
        Args:
            visitors: List of (visitor_id, feature, metadata) tuples
        
        Returns:
            Number of visitors successfully added
        """
        if not FAISS_AVAILABLE or not visitors:
            return 0
        
        features_list = []
        metadata_list = []
        
        for visitor_id, feature, metadata in visitors:
            if feature is None or feature.shape[0] != self.dimension:
                continue
            
            # Skip if already exists
            if visitor_id in self.visitor_id_to_index:
                continue
            
            # Normalize feature
            feature_norm = feature / np.linalg.norm(feature)
            features_list.append(feature_norm.astype('float32'))
            
            metadata_list.append({
                'visitor_id': visitor_id,
                'index': self.next_index,
                **(metadata or {})
            })
            self.visitor_id_to_index[visitor_id] = self.next_index
            self.next_index += 1
        
        if not features_list:
            return 0
        
        try:
            # Batch add to index
            features_array = np.vstack(features_list)
            self.index.add(features_array)
            
            # Store metadata
            for meta in metadata_list:
                idx = meta['index']
                del meta['index']
                self.metadata[idx] = meta
            
            return len(features_list)
        except Exception as e:
            print(f"Error batch adding visitors to FAISS index: {e}")
            return 0
    
    def search(self, query_feature: np.ndarray, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Search for nearest neighbors using ANN.
        
        Args:
            query_feature: Query face feature vector (512-dim numpy array)
            k: Number of nearest neighbors to return
        
        Returns:
            List of (visitor_id, cosine_similarity, metadata) tuples, sorted by similarity (descending)
            Note: cosine_similarity ranges from -1 to 1, higher is better
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query feature
            query_norm = query_feature / np.linalg.norm(query_feature)
            query_norm = query_norm.astype('float32').reshape(1, -1)
            
            # Search - IndexHNSWFlat returns L2 distances
            # For normalized vectors: L2^2 = 2(1 - cosine_similarity)
            # So: cosine_similarity = 1 - (L2^2 / 2)
            distances, indices = self.index.search(query_norm, min(k, self.index.ntotal))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:  # Invalid index
                    continue
                
                if idx in self.metadata:
                    visitor_id = self.metadata[idx].get('visitor_id', f'unknown_{idx}')
                    # Convert L2 distance to cosine similarity
                    # For normalized vectors: ||a-b||^2 = 2(1 - a·b) where a·b is cosine similarity
                    # So: cosine_similarity = 1 - (L2^2 / 2)
                    l2_squared = float(dist) ** 2
                    cosine_similarity = 1.0 - (l2_squared / 2.0)
                    # Clamp to valid range [-1, 1]
                    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
                    results.append((visitor_id, cosine_similarity, self.metadata[idx]))
            
            # Sort by similarity descending (higher is better)
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            print(f"Error searching FAISS index: {e}")
            return []
    
    def remove_visitor(self, visitor_id: str) -> bool:
        """Remove a visitor from the index."""
        if visitor_id not in self.visitor_id_to_index:
            return False
        
        try:
            idx = self.visitor_id_to_index[visitor_id]
            # FAISS doesn't support efficient removal, so we mark as removed in metadata
            # For production, consider rebuilding index periodically
            del self.metadata[idx]
            del self.visitor_id_to_index[visitor_id]
            return True
        except Exception as e:
            print(f"Error removing visitor from FAISS index: {e}")
            return False
    
    def rebuild_from_database(self, get_visitors_func, extract_feature_func) -> int:
        """
        Rebuild index from database visitors.
        
        Args:
            get_visitors_func: Function that returns list of visitor data
            extract_feature_func: Function(visitor_data) -> feature vector
        
        Returns:
            Number of visitors indexed
        """
        if not FAISS_AVAILABLE:
            return 0
        
        # Create new index
        self.index = self._create_index()
        self.metadata = {}
        self.visitor_id_to_index = {}
        self.next_index = 0
        
        visitors = get_visitors_func()
        batch_data = []
        
        for visitor_data in visitors:
            try:
                feature = extract_feature_func(visitor_data)
                if feature is not None:
                    visitor_id = str(visitor_data.get('id', visitor_data.get('visitor_id', 'unknown')))
                    batch_data.append((visitor_id, feature, visitor_data))
            except Exception as e:
                print(f"Error extracting feature for visitor: {e}")
                continue
        
        count = self.add_visitors_batch(batch_data)
        
        if count > 0:
            self._save_index()
            print(f"✓ Rebuilt FAISS index with {count} visitors")
        
        return count
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': 'HNSW',
            'm': self.m,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'visitors_indexed': len(self.visitor_id_to_index)
        }
    
    def save(self) -> bool:
        """Save index to disk."""
        return self._save_index()
    
    def clear(self):
        """Clear the index."""
        self.index = self._create_index()
        self.metadata = {}
        self.visitor_id_to_index = {}
        self.next_index = 0
