""" I dont love this solution but seems good enough for now might look into this more later"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # Python < 3.8
    try:
        from importlib_metadata import PackageNotFoundError, version
    except ImportError:
        PackageNotFoundError = Exception
        version = None

from pathlib import Path
import re
import warnings


def _version_from_setup_py():
    """Read package version from setup.py as a source-tree fallback."""
    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    try:
        content = setup_py.read_text(encoding="utf-8")
    except OSError:
        return "0+unknown"
    match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
    return match.group(1) if match else "0+unknown"


if version is None:
    __version__ = _version_from_setup_py()
else:
    try:
        __version__ = version("bmtool")
    except PackageNotFoundError:
        __version__ = _version_from_setup_py()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
