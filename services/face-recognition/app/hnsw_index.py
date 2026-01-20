"""
HNSW Index Manager for Fast Face Recognition
Uses HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search
for efficient face feature matching using cosine similarity.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING, Any
from pathlib import Path

if TYPE_CHECKING:
    import hnswlib

try:
    import hnswlib  # type: ignore[import-untyped]
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    hnswlib = None  # type: ignore[assignment]
    print("Warning: HNSW not available. Install with: pip install hnswlib")

# Configuration
INDEX_FILE = "hnsw_visitor_index.bin"
METADATA_FILE = "hnsw_visitor_metadata.pkl"
DEFAULT_DIMENSION = 512  # Sface feature dimension
DEFAULT_M = 32  # HNSW parameter: number of bi-directional links
DEFAULT_EF_CONSTRUCTION = 200  # HNSW parameter: size of dynamic candidate list
DEFAULT_EF_SEARCH = 50  # HNSW parameter: number of nearest neighbors to explore


class HNSWIndexManager:
    """
    Manages HNSW index for fast face recognition using cosine similarity.
    """
    
    def __init__(self, 
                 dimension: int = DEFAULT_DIMENSION,
                 m: int = DEFAULT_M,
                 ef_construction: int = DEFAULT_EF_CONSTRUCTION,
                 ef_search: int = DEFAULT_EF_SEARCH,
                 index_dir: str = "models"):
        """
        Initialize HNSW index manager.
        
        Args:
            dimension: Feature vector dimension (default: 512 for Sface)
            m: HNSW parameter - number of bi-directional links (default: 32)
            ef_construction: HNSW parameter - size of dynamic candidate list during construction (default: 200)
            ef_search: HNSW parameter - number of nearest neighbors to explore during search (default: 50)
            index_dir: Directory to store index files
        """
        if not HNSW_AVAILABLE:
            raise ImportError("HNSW is not available. Install with: pip install hnswlib")
        
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index: Optional[Any] = None  # hnswlib.Index when HNSW is available
        self.metadata: Dict[int, Dict] = {}  # Maps index position to visitor info
        self.visitor_id_to_index: Dict[str, int] = {}  # Maps visitor_id to index position
        self.next_index = 0
        
        # Load existing index if available
        self._load_index()
    
    def _create_index(self) -> Any:  # Returns hnswlib.Index when HNSW is available
        """Create a new HNSW index for cosine similarity."""
        if not HNSW_AVAILABLE or hnswlib is None:
            raise RuntimeError("HNSW is not available")
        
        # Create HNSW index with cosine similarity (inner product for normalized vectors)
        # hnswlib uses 'cosine' space for cosine similarity
        index = hnswlib.Index(space='cosine', dim=self.dimension)
        index.init_index(max_elements=10000, ef_construction=self.ef_construction, M=self.m)
        index.set_ef(self.ef_search)
        return index
    
    def _load_index(self) -> bool:
        """Load index and metadata from disk."""
        index_path = self.index_dir / INDEX_FILE
        metadata_path = self.index_dir / METADATA_FILE
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load HNSW index
                if not HNSW_AVAILABLE or hnswlib is None:
                    raise RuntimeError("HNSW is not available")
                
                index = hnswlib.Index(space='cosine', dim=self.dimension)
                index.load_index(str(index_path))
                index.set_ef(self.ef_search)
                self.index = index
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', {})
                    self.visitor_id_to_index = data.get('visitor_id_to_index', {})
                    self.next_index = data.get('next_index', 0)
                
                print(f"✓ Loaded HNSW index with {self.index.get_current_count()} vectors")
                return True
            except Exception as e:
                print(f"⚠ Error loading HNSW index: {e}. Creating new index.")
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
            
            # Save HNSW index
            if self.index is None:
                return False
            self.index.save_index(str(index_path))
            
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
            print(f"⚠ Error saving HNSW index: {e}")
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
        if not HNSW_AVAILABLE or self.index is None:
            return False
        
        if visitor_id in self.visitor_id_to_index:
            # Visitor already exists, skip or update
            print(f"Visitor {visitor_id} already in index, skipping")
            return False
        
        try:
            # Normalize feature vector for cosine similarity
            feature_norm = feature / np.linalg.norm(feature)
            feature_norm = feature_norm.astype('float32')
            
            # Add to index
            self.index.add_items(feature_norm.reshape(1, -1), np.array([self.next_index]))
            
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
            print(f"Error adding visitor to HNSW index: {e}")
            return False
    
    def add_visitors_batch(self, visitors: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        """
        Add multiple visitors to the index in batch.
        
        Args:
            visitors: List of (visitor_id, feature, metadata) tuples
        
        Returns:
            Number of visitors successfully added
        """
        if not HNSW_AVAILABLE or not visitors or self.index is None:
            return 0
        
        features_list = []
        indices_list = []
        metadata_list = []
        
        for visitor_id, feature, metadata in visitors:
            if feature is None:
                continue
            
            # Ensure feature is 1D array with correct dimension
            feature = np.asarray(feature).flatten()
            if feature.shape[0] != self.dimension:
                print(f"⚠ Skipping visitor {visitor_id}: feature dimension {feature.shape[0]} != {self.dimension}")
                continue
            
            # Skip if already exists
            if visitor_id in self.visitor_id_to_index:
                continue
            
            # Normalize feature
            feature_norm = feature / np.linalg.norm(feature)
            features_list.append(feature_norm.astype('float32'))
            indices_list.append(self.next_index)
            
            metadata_list.append({
                'visitor_id': visitor_id,
                'index': self.next_index,
                **(metadata or {})
            })
            self.visitor_id_to_index[visitor_id] = self.next_index
            self.next_index += 1
        
        if not features_list:
            print(f"⚠ No valid features to add to HNSW index (processed {len(visitors)} visitors)")
            if visitors:
                # Debug: show first visitor's feature shape
                first_id, first_feature, _ = visitors[0]
                if first_feature is not None:
                    print(f"  Debug: First visitor '{first_id}' feature shape: {np.asarray(first_feature).shape}, expected: ({self.dimension},)")
            return 0
        
        try:
            # Batch add to index
            features_array = np.vstack(features_list)
            indices_array = np.array(indices_list)
            print(f"Adding {len(features_list)} features to HNSW index...")
            self.index.add_items(features_array, indices_array)
            
            # Store metadata
            for meta in metadata_list:
                idx = meta['index']
                del meta['index']
                self.metadata[idx] = meta
            
            print(f"✓ Successfully added {len(features_list)} visitors to HNSW index")
            return len(features_list)
        except Exception as e:
            print(f"⚠ Error batch adding visitors to HNSW index: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def search(self, query_feature: np.ndarray, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Search for nearest neighbors using HNSW ANN.
        
        Args:
            query_feature: Query face feature vector (512-dim numpy array)
            k: Number of nearest neighbors to return
        
        Returns:
            List of (visitor_id, cosine_similarity, metadata) tuples, sorted by similarity (descending)
            Note: cosine_similarity ranges from -1 to 1, higher is better
        """
        if not HNSW_AVAILABLE or self.index is None:
            return []
        
        try:
            current_count = self.index.get_current_count()
            if current_count == 0:
                return []
            
            # Normalize query feature
            query_norm = query_feature / np.linalg.norm(query_feature)
            query_norm = query_norm.astype('float32').reshape(1, -1)
            
            # Search - hnswlib returns labels (indices) and distances
            # For cosine space, distances are already cosine distances
            # Cosine similarity = 1 - cosine_distance
            labels, distances = self.index.knn_query(query_norm, k=min(k, current_count))
            
            results = []
            for label, dist in zip(labels[0], distances[0]):
                idx = int(label)
                if idx < 0 or idx not in self.metadata:
                    continue
                
                visitor_id = self.metadata[idx].get('visitor_id', f'unknown_{idx}')
                # Convert cosine distance to cosine similarity
                # hnswlib cosine space: distance = 1 - cosine_similarity
                # So: cosine_similarity = 1 - distance
                cosine_distance = float(dist)
                cosine_similarity = 1.0 - cosine_distance
                # Clamp to valid range [-1, 1]
                cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
                results.append((visitor_id, cosine_similarity, self.metadata[idx]))
            
            # Sort by similarity descending (higher is better)
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        except Exception as e:
            print(f"Error searching HNSW index: {e}")
            return []
    
    def remove_visitor(self, visitor_id: str) -> bool:
        """Remove a visitor from the index."""
        if visitor_id not in self.visitor_id_to_index:
            return False
        
        try:
            idx = self.visitor_id_to_index[visitor_id]
            # HNSW doesn't support efficient removal, so we mark as removed in metadata
            # For production, consider rebuilding index periodically
            del self.metadata[idx]
            del self.visitor_id_to_index[visitor_id]
            return True
        except Exception as e:
            print(f"Error removing visitor from HNSW index: {e}")
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
        if not HNSW_AVAILABLE:
            return 0
        
        # Create new index
        self.index = self._create_index()
        self.metadata = {}
        self.visitor_id_to_index = {}
        self.next_index = 0

        # Try to get visitors from database
        try:
            visitors = get_visitors_func()
        except Exception as e:
            print("⚠ Falling back to using test_images instead of database because getting visitors from the database failed.")
            print(f"Reason: {e}")
            visitors = []
        
        batch_data = []
        features_extracted = 0
        features_failed = 0
        
        print(f"Processing {len(visitors)} visitors for HNSW index...")
        for visitor_data in visitors:
            try:
                feature = extract_feature_func(visitor_data)
                if feature is not None:
                    visitor_id = str(visitor_data.get('id', visitor_data.get('visitor_id', 'unknown')))
                    batch_data.append((visitor_id, feature, visitor_data))
                    features_extracted += 1
                else:
                    features_failed += 1
            except Exception as e:
                print(f"Error extracting feature for visitor: {e}")
                features_failed += 1
                continue
        
        print(f"Extracted features: {features_extracted} successful, {features_failed} failed")
        if len(batch_data) == 0:
            print("⚠ No features extracted from visitors. Cannot build HNSW index.")
            return 0
        
        count = self.add_visitors_batch(batch_data)
        
        if count > 0:
            self._save_index()
            print(f"✓ Rebuilt HNSW index with {count} visitors")
        
        return count
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        current_count = self.index.get_current_count() if self.index else 0
        return {
            'total_vectors': current_count,
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
    
    @property
    def ntotal(self) -> int:
        """Get total number of vectors in index (for compatibility)."""
        if self.index is None:
            return 0
        return self.index.get_current_count()
