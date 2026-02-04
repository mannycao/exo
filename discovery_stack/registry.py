import json
import os
import time
from typing import List, Dict, Any

class TargetRegistry:
    def __init__(self, file_path: str = "processed_targets.json"):
        self.file_path = file_path
        self.registry: Dict[str, Dict[str, Any]] = self.load_registry()

    def load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Loads the registry from a JSON file. Creates the file if it does not exist."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {self.file_path} is corrupted. Creating a new registry.")
                return {}
            except Exception as e:
                print(f"Error loading registry from {self.file_path}: {e}. Creating a new registry.")
                return {}
        else:
            return {}

    def save_registry(self):
        """Saves the current registry to the JSON file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            print(f"Error saving registry to {self.file_path}: {e}")

    def is_processed(self, tic_id: str, sector: int) -> bool:
        """Checks if a target (TIC ID and Sector) has already been processed."""
        target_key = f"{tic_id}-{sector}"
        return target_key in self.registry

    def mark_processed(self, tic_id: str, sector: int, status: str, timestamp: str):
        """Marks a target as processed with a given status and timestamp.

        Args:
            tic_id: The Target Identification Catalog (TIC) ID.
            sector: The observation sector.
            status: The processing status (e.g., "Candidate", "Rejected", "Error").
            timestamp: The time of processing in ISO format.
        """
        target_key = f"{tic_id}-{sector}"
        self.registry[target_key] = {
            "status": status,
            "timestamp": timestamp
        }
        self.save_registry()

    def get_stats(self) -> Dict[str, int]:
        """Returns statistics on processed targets (Candidates vs Rejected)."""
        candidates = 0
        rejected = 0
        errors = 0
        for data in self.registry.values():
            if data.get("status") == "Candidate":
                candidates += 1
            elif data.get("status") == "Rejected":
                rejected += 1
            elif data.get("status") == "Error":
                errors += 1
        return {"Candidates": candidates, "Rejected": rejected, "Errors": errors}

# Example of how to use it (for testing purposes, not part of the final class definition in the file):
# if __name__ == "__main__":
#     # Ensure a dummy file exists for testing the load_registry part if needed
#     if not os.path.exists("processed_targets.json"):
#         with open("processed_targets.json", "w") as f:
#             json.dump({"TIC123-1": {"status": "Candidate", "timestamp": "2026-02-02T10:00:00Z"}}, f)
# 
#     registry = TargetRegistry("processed_targets.json")
#     print(f"Initial stats: {registry.get_stats()}")
# 
#     # Mark a new target
#     tic_id_new = "TIC456"
#     sector_new = 20
#     if not registry.is_processed(tic_id_new, sector_new):
#         registry.mark_processed(tic_id_new, sector_new, "Candidate", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
#         print(f"Marked {tic_id_new}-{sector_new} as processed.")
#     else:
#         print(f"{tic_id_new}-{sector_new} already processed.")
# 
#     print(f"Updated stats: {registry.get_stats()}")
# 
#     # Mark another target as rejected
#     tic_id_rejected = "TIC789"
#     sector_rejected = 21
#     registry.mark_processed(tic_id_rejected, sector_rejected, "Rejected", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
#     print(f"Marked {tic_id_rejected}-{sector_rejected} as processed.")
# 
#     print(f"Final stats: {registry.get_stats()}")
# 
#     # Clean up dummy file
#     # if os.path.exists("processed_targets.json"):
#     #     os.remove("processed_targets.json")