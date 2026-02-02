"""
TargetRegistry: Maintains persistent state of processed targets.

This module tracks which TIC ID / Sector combinations have been processed,
preventing duplicate pipeline runs and enabling checkpoint recovery.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, List
from threading import Lock

logger = logging.getLogger(__name__)


class TargetRegistry:
    """
    Persistent registry of processed targets.
    
    Stores state in a JSON file (processed_targets.json) with entries:
    {
        "targets": [
            {
                "tic_id": 12345,
                "sector": 15,
                "status": "ACCEPTED",
                "timestamp": "2025-02-02T10:30:45",
                "rationale": "ML Conf > 0.9 AND CACL Disagreement < 0.1..."
            }
        ],
        "stats": { "accepted": 5, "rejected": 12, "processing": 0 }
    }
    """
    
    def __init__(self, registry_path: str = "processed_targets.json"):
        """
        Initialize the registry, creating the backing JSON file if needed.
        
        Args:
            registry_path: Path to the JSON persistence file.
        """
        self.registry_path = registry_path
        self.lock = Lock()  # Thread-safe access
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.registry_path) if os.path.dirname(self.registry_path) else ".", exist_ok=True)
        
        # Initialize or load the registry
        if not os.path.exists(self.registry_path):
            self._initialize_registry()
        
        logger.info(f"TargetRegistry initialized. Backing file: {self.registry_path}")
    
    def _initialize_registry(self):
        """Create a new empty registry file."""
        with self.lock:
            initial_data = {
                "targets": [],
                "stats": {
                    "accepted": 0,
                    "rejected": 0,
                    "processing": 0
                },
                "initialized_at": datetime.now().isoformat()
            }
            try:
                with open(self.registry_path, "w") as f:
                    json.dump(initial_data, f, indent=2)
                logger.debug(f"Created new registry file: {self.registry_path}")
            except Exception as e:
                logger.error(f"Failed to initialize registry: {e}", exc_info=True)
                raise
    
    def _load_registry(self) -> Dict:
        """Load the current registry from disk."""
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Registry file corrupted. Reinitializing.")
            self._initialize_registry()
            return {"targets": [], "stats": {"accepted": 0, "rejected": 0, "processing": 0}}
        except Exception as e:
            logger.error(f"Failed to load registry: {e}", exc_info=True)
            return {"targets": [], "stats": {"accepted": 0, "rejected": 0, "processing": 0}}
    
    def _save_registry(self, data: Dict):
        """Atomically save registry to disk."""
        try:
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}", exc_info=True)
            raise
    
    def is_processed(self, tic_id: int, sector: int) -> bool:
        """
        Check if a TIC ID / Sector combination has already been processed.
        
        Args:
            tic_id: The TESS Input Catalog ID.
            sector: The observation sector.
        
        Returns:
            True if processed, False otherwise.
        """
        with self.lock:
            data = self._load_registry()
            for entry in data.get("targets", []):
                if entry["tic_id"] == tic_id and entry["sector"] == sector:
                    return True
        return False
    
    def mark_processed(
        self, 
        tic_id: int, 
        sector: int, 
        status: str, 
        rationale: str = "",
        timestamp: Optional[str] = None
    ):
        """
        Mark a target as processed and record the outcome.
        
        Args:
            tic_id: The TESS Input Catalog ID.
            sector: The observation sector.
            status: One of "ACCEPTED", "REJECTED", "PHYSICS_CLEARED".
            rationale: Human-readable reason for the decision.
            timestamp: ISO timestamp (auto-generated if None).
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        with self.lock:
            data = self._load_registry()
            
            # Check if already exists (idempotent)
            for entry in data.get("targets", []):
                if entry["tic_id"] == tic_id and entry["sector"] == sector:
                    logger.debug(f"TIC {tic_id} Sector {sector} already in registry. Skipping.")
                    return
            
            # Add new entry
            entry = {
                "tic_id": tic_id,
                "sector": sector,
                "status": status,
                "timestamp": timestamp,
                "rationale": rationale
            }
            data["targets"].append(entry)
            
            # Update stats
            status_key = status.lower()
            if status_key in data["stats"]:
                data["stats"][status_key] = data["stats"].get(status_key, 0) + 1
            
            self._save_registry(data)
            logger.info(
                f"Registered TIC {tic_id} Sector {sector}: {status} "
                f"({rationale[:50]}...)" if len(rationale) > 50 else f"({rationale})"
            )
    
    def get_stats(self) -> Dict:
        """
        Retrieve summary statistics of the registry.
        
        Returns:
            Dictionary with counts: {"accepted": X, "rejected": Y, "processing": Z, "total": T}
        """
        with self.lock:
            data = self._load_registry()
            stats = data.get("stats", {})
            total = len(data.get("targets", []))
            stats["total"] = total
            return stats
    
    def get_all_targets(self) -> List[Dict]:
        """
        Retrieve all registered targets.
        
        Returns:
            List of target entry dictionaries.
        """
        with self.lock:
            data = self._load_registry()
            return data.get("targets", [])
    
    def get_target_history(self, tic_id: int) -> List[Dict]:
        """
        Retrieve all processing history for a specific TIC ID.
        
        Args:
            tic_id: The TESS Input Catalog ID.
        
        Returns:
            List of entries for that TIC ID across all sectors.
        """
        with self.lock:
            data = self._load_registry()
            return [e for e in data.get("targets", []) if e["tic_id"] == tic_id]
