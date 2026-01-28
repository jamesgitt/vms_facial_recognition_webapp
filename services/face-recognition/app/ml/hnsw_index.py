"""
HNSW Index Manager for Fast Face Recognition

Uses HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search
for efficient face feature matching using cosine similarity.
"""

import os
import pickle
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable

from core.logger import get_logger
logger = get_logger(__name__)

import numpy as np

try:
    import hnswlib  # type: ignore[import-untyped]
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    hnswlib = None  # type: ignore[assignment]
    logger.warning("HNSW not available. Install with: pip install hnswlib")

# Configuration
INDEX_FILE = os.environ.get("HNSW_INDEX_FILE", "hnsw_visitor_index.bin")
METADATA_FILE = os.environ.get("HNSW_METADATA_FILE", "hnsw_visitor_metadata.pkl")
DEFAULT_DIMENSION = 128
DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 400
DEFAULT_EF_SEARCH = 400
DEFAULT_MAX_ELEMENTS = int(os.environ.get("HNSW_MAX_ELEMENTS", "100000"))


class HNSWIndexManager:
    """Manages HNSW index for fast face recognition using cosine similarity."""
    
    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        index_dir: str = "models",
        max_elements: int = DEFAULT_MAX_ELEMENTS
    ):
        """
        Initialize HNSW index manager.
        
        Args:
            dimension: Feature vector dimension (default: 128 for SFace)
            m: Number of bi-directional links (higher = better recall)
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Number of nearest neighbors to explore during search
            index_dir: Directory to store index files
            max_elements: Maximum number of vectors in index
        """
        if not HNSW_AVAILABLE:
            raise ImportError("HNSW is not available. Install with: pip install hnswlib")
        
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index: Optional[Any] = None
        self.metadata: Dict[int, Dict] = {}
        self.visitor_id_to_index: Dict[str, int] = {}
        self.next_index = 0
        
        self._load_index()
    
    @property
    def _index_path(self) -> Path:
        return self.index_dir / INDEX_FILE
    
    @property
    def _metadata_path(self) -> Path:
        return self.index_dir / METADATA_FILE
    
    def _create_index(self) -> Any:
        """Create a new HNSW index for cosine similarity."""
        if hnswlib is None:
            raise RuntimeError("HNSW is not available")
        
        index = hnswlib.Index(space='cosine', dim=self.dimension)
        index.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.m)
        index.set_ef(self.ef_search)
        return index
    
    def _load_index(self) -> bool:
        """Load index and metadata from disk."""
        if not (self._index_path.exists() and self._metadata_path.exists()):
            self.index = self._create_index()
            return False
        
        try:
            if hnswlib is None:
                raise RuntimeError("HNSW is not available")
            
            index = hnswlib.Index(space='cosine', dim=self.dimension)
            index.load_index(str(self._index_path))
            index.set_ef(self.ef_search)
            self.index = index
            
            with open(self._metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', {})
                self.visitor_id_to_index = data.get('visitor_id_to_index', {})
                self.next_index = data.get('next_index', 0)
            
            logger.info(f"Loaded HNSW index with {self.index.get_current_count()} vectors")
            return True
            
        except Exception as e:
            logger.warning(f"Error loading HNSW index: {e}. Creating new index.")
            self.index = self._create_index()
            return False
    
    def _save_index(self) -> bool:
        """Save index and metadata to disk."""
        if self.index is None:
            return False
        
        try:
            self.index.save_index(str(self._index_path))
            
            data = {
                'metadata': self.metadata,
                'visitor_id_to_index': self.visitor_id_to_index,
                'next_index': self.next_index
            }
            with open(self._metadata_path, 'wb') as f:
                pickle.dump(data, f)
            
            return True
        except Exception as e:
            logger.warning(f"Error saving HNSW index: {e}")
            return False
    
    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """Normalize feature vector for cosine similarity."""
        return (feature / np.linalg.norm(feature)).astype('float32')
    
    def add_visitor(self, visitor_id: str, feature: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Add a visitor's face feature to the index.
        
        Args:
            visitor_id: Unique visitor identifier
            feature: Face feature vector (128-dim numpy array)
            metadata: Optional metadata (e.g., name)
        
        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            return False
        
        if visitor_id in self.visitor_id_to_index:
            logger.info(f"Visitor {visitor_id} already in index, skipping")
            return False
        
        try:
            feature_norm = self._normalize_feature(feature)
            self.index.add_items(feature_norm.reshape(1, -1), np.array([self.next_index]))
            
            self.metadata[self.next_index] = {'visitor_id': visitor_id, **(metadata or {})}
            self.visitor_id_to_index[visitor_id] = self.next_index
            self.next_index += 1
            
            return True
        except Exception as e:
            logger.error(f"Error adding visitor to HNSW index: {e}")
            return False
    
    def add_visitors_batch(self, visitors: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> int:
        """
        Add multiple visitors to the index in batch.
        
        Args:
            visitors: List of (visitor_id, feature, metadata) tuples
        
        Returns:
            Number of visitors successfully added
        """
        if not visitors or self.index is None:
            return 0
        
        features_list = []
        indices_list = []
        pending_metadata = []
        
        for visitor_id, feature, meta in visitors:
            if feature is None or visitor_id in self.visitor_id_to_index:
                continue
            
            feature = np.asarray(feature).flatten()
            if feature.shape[0] != self.dimension:
                logger.warning(f"Skipping visitor {visitor_id}: dimension {feature.shape[0]} != {self.dimension}")
                continue
            
            feature_norm = self._normalize_feature(feature)
            features_list.append(feature_norm)
            indices_list.append(self.next_index)
            
            pending_metadata.append({
                'idx': self.next_index,
                'visitor_id': visitor_id,
                'firstName': meta.get('firstName') if meta else None,
                'lastName': meta.get('lastName') if meta else None,
            })
            self.visitor_id_to_index[visitor_id] = self.next_index
            self.next_index += 1
        
        if not features_list:
            logger.warning(f"No valid features to add (processed {len(visitors)} visitors)")
            return 0
        
        try:
            features_array = np.vstack(features_list)
            indices_array = np.array(indices_list)
            logger.info(f"Adding {len(features_list)} features to HNSW index...")
            self.index.add_items(features_array, indices_array)
            
            for meta in pending_metadata:
                idx = meta.pop('idx')  # pyrefly: ignore
                self.metadata[idx] = meta  # pyrefly: ignore
            
            logger.info(f"Added {len(features_list)} visitors to HNSW index")
            return len(features_list)
            
        except Exception as e:
            logger.warning(f"Error batch adding to HNSW index: {e}")
            traceback.print_exc()
            return 0
    
    def search(self, query_feature: np.ndarray, k: int = 100) -> List[Tuple[str, float, Dict]]:
        """
        Search for nearest neighbors using HNSW ANN.
        
        Args:
            query_feature: Query face feature vector (128-dim numpy array)
            k: Number of nearest neighbors to return
        
        Returns:
            List of (visitor_id, cosine_similarity, metadata) tuples, sorted by similarity descending
        """
        if self.index is None:
            return []
        
        try:
            current_count = self.index.get_current_count()
            if current_count == 0:
                return []
            
            query_norm = self._normalize_feature(query_feature).reshape(1, -1)
            labels, distances = self.index.knn_query(query_norm, k=min(k, current_count))
            
            results = []
            for label, dist in zip(labels[0], distances[0]):
                idx = int(label)
                if idx < 0 or idx not in self.metadata:
                    continue
                
                visitor_id = self.metadata[idx].get('visitor_id', f'unknown_{idx}')
                # hnswlib cosine space: distance = 1 - cosine_similarity
                similarity = max(-1.0, min(1.0, 1.0 - float(dist)))
                results.append((visitor_id, similarity, self.metadata[idx]))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error searching HNSW index: {e}")
            return []
    
    def remove_visitor(self, visitor_id: str) -> bool:
        """Remove a visitor from the index (marks as removed in metadata)."""
        if visitor_id not in self.visitor_id_to_index:
            return False
        
        try:
            idx = self.visitor_id_to_index[visitor_id]
            del self.metadata[idx]
            del self.visitor_id_to_index[visitor_id]
            return True
        except Exception as e:
            logger.error(f"Error removing visitor from HNSW index: {e}")
            return False
    
    def rebuild_from_database(
        self,
        get_visitors_func: Callable[[], List[Dict]],
        extract_feature_func: Callable[[Dict], Optional[np.ndarray]]
    ) -> int:
        """
        Rebuild index from database visitors.
        
        Args:
            get_visitors_func: Function that returns list of visitor data
            extract_feature_func: Function(visitor_data) -> feature vector
        
        Returns:
            Number of visitors indexed
        """
        self.clear()
        
        try:
            visitors = get_visitors_func()
        except Exception as e:
            logger.warning(f"Failed to get visitors from database: {e}")
            visitors = []
        
        if not visitors:
            logger.warning("No visitors to index")
            return 0
        
        logger.info(f"Processing {len(visitors)} visitors for HNSW index...")
        batch_data = []
        success_count = 0
        fail_count = 0
        
        for visitor_data in visitors:
            visitor_id = str(visitor_data.get('id', visitor_data.get('visitor_id', 'unknown')))
            
            try:
                feature = extract_feature_func(visitor_data)
                if feature is None:
                    fail_count += 1
                    continue
                
                feature = np.asarray(feature).flatten()
                if feature.shape[0] != self.dimension:
                    fail_count += 1
                    continue
                
                minimal_metadata = {
                    'firstName': visitor_data.get('firstName', ''),
                    'lastName': visitor_data.get('lastName', ''),
                }
                batch_data.append((visitor_id, feature, minimal_metadata))
                success_count += 1
                
            except Exception as e:
                fail_count += 1
                if fail_count <= 3:
                    logger.error(f"Visitor {visitor_id}: {e}")
        
        logger.info(f"Feature extraction: {success_count} successful, {fail_count} failed")
        
        if not batch_data:
            logger.warning("No features extracted. Cannot build HNSW index.")
            return 0
        
        count = self.add_visitors_batch(batch_data)
        
        if count > 0:
            self._save_index()
            logger.info(f"Rebuilt HNSW index with {count} visitors")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_vectors': self.index.get_current_count() if self.index else 0,
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
    
    def clear(self) -> None:
        """Clear the index."""
        self.index = self._create_index()
        self.metadata = {}
        self.visitor_id_to_index = {}
        self.next_index = 0
    
    @property
    def ntotal(self) -> int:
        """Get total number of vectors in index (for compatibility)."""
        return self.index.get_current_count() if self.index else 0
