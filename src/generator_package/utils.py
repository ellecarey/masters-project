import os
from pathlib import Path


def find_project_root():
    """Find project root by looking for a marker file/directory."""
    current_path = Path(__file__).resolve()

    # Look for common project root markers
    markers = [".git", "pyproject.toml", "requirements.txt", "README.md"]

    for parent in current_path.parents:
        if any((parent / marker).exists() for marker in markers):
            return str(parent)

    # Fallback: assume project root is 2 levels up from this utils file
    return str(current_path.parent.parent)


def get_project_paths():
    """Get commonly used project paths."""
    project_root = find_project_root()
    return {
        "project_root": project_root,
        "data_dir": os.path.join(project_root, "data"),
        "figures_dir": os.path.join(project_root, "reports", "figures"),
        "src_dir": os.path.join(project_root, "src"),
    }
