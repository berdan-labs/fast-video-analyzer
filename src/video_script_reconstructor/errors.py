from __future__ import annotations


class ReconstructorError(Exception):
    """Base class for expected production failures."""


class InputError(ReconstructorError):
    """The requested input or configuration is invalid."""


class SecurityError(ReconstructorError):
    """An input violates an offline, containment, or trust boundary."""


class BlockedError(ReconstructorError):
    """A fidelity prerequisite prevents safe completion."""


class ReviewRequired(ReconstructorError):
    """A usable result exists but consequential uncertainty remains."""


class ValidationFailure(ReconstructorError):
    """A deterministic contract or integrity check failed."""


class StaleRevisionError(ReconstructorError):
    """An observation targets an obsolete metadata base revision."""
