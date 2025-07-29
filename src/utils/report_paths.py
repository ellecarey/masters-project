from pathlib import Path
from datetime import datetime
from typing import Literal, Optional
import os
import re

_ART_TYPES = {
    "figure": "figures",
    "spreadsheet": "spreadsheets",
    "comparison": "comparisons", 
    "tuning": "tuning",
    "log": "logs",
    "misc": "misc",
}

def reports_root() -> Path:
    return Path(
        os.getenv("REPORTS_DIR", Path(__file__).resolve().parents[2] / "reports")
    )

def artefact_path(
    experiment: str,
    art_type: Literal[*_ART_TYPES.keys()],
    filename: str,
    dated_subfolder: bool = False,
) -> Path:
    """
    Construct a full path such as
    reports/figures/<experiment>[/YYYY-MM-DD]/<filename>
    """
    if art_type not in _ART_TYPES:
        raise ValueError(f"Unknown art_type '{art_type}'")

    base = reports_root() / _ART_TYPES[art_type] / experiment
    if dated_subfolder:
        base = base / datetime.today().strftime("%Y-%m-%d")
    base.mkdir(parents=True, exist_ok=True)
    return base / filename

def experiment_family_path(
    full_experiment_name: str,
    art_type: Literal[*_ART_TYPES.keys()],
    subfolder: str,
    filename: str,
) -> Path:
    """
    Create nested experiment family structure:
    reports/figures/<family_base>/<subfolder>/<filename>
    """
    if art_type not in _ART_TYPES:
        raise ValueError(f"Unknown art_type '{art_type}'")
    
    # Extract family base (everything before first _seed or _pert)
    family_base = extract_family_base(full_experiment_name)
    
    base = reports_root() / _ART_TYPES[art_type] / family_base / subfolder
    base.mkdir(parents=True, exist_ok=True)
    return base / filename

def extract_family_base(experiment_name: str) -> str:
    """Extract the base family name from a full experiment name."""
    import re
    print(f"DEBUG: Input experiment_name = {experiment_name}")
    
    # Remove everything from _seed onwards, OR everything from _pert onwards, OR everything from _mlp onwards
    base = re.sub(r'_(?:seed\d+|pert_|mlp_).*$', '', experiment_name)
    
    print(f"DEBUG: Extracted family_base = {base}")
    return base