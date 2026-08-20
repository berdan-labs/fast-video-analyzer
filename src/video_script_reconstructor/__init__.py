"""Fast Video Analyzer implementation package.

The historical ``video_script_reconstructor`` import path remains supported so
existing projects and serialized evidence can be read without migration.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

try:
    __version__ = distribution_version("fast-video-analyzer")
except PackageNotFoundError:
    # Direct source execution is not a released distribution. Keep that state
    # explicit instead of manufacturing a version that could enter evidence.
    __version__ = "0+unknown"
