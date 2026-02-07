from pathlib import Path
import sys
import os

script_path = Path(__file__).resolve()
parent_dir = script_path.parents[0]
grandparent_dir = script_path.parents[1]
great_grandparent_dir = script_path.parents[2]

print(f"script_path: {script_path}")
print(f"parent_dir: {parent_dir}")
print(f"grandparent_dir: {grandparent_dir}")
print(f"great_grandparent_dir: {great_grandparent_dir}")

sys.path.insert(0, str(great_grandparent_dir))
print(f"sys.path after insert: {sys.path}")

try:
    import config
    print("config imported successfully!")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("config not imported.")
