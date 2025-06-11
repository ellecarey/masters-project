import os
from .utils import find_project_root

PROJECT_ROOT = find_project_root()

# --- Output Settings ---
# paths are derived at runtime and are not static settings
OUTPUT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
FIGURES_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
