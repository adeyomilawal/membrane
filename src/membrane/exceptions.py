"""Membrane public exceptions."""


class MembraneError(Exception):
    """Base class for Membrane errors."""

    pass


class FeatureNotAvailable(MembraneError, ImportError):
    """Raised when a feature is not included in the open-source edition."""

    def __init__(self, feature: str) -> None:
        super().__init__(f"{feature} is not included in the open-source edition of Membrane.")
        self.feature = feature


class MemoryCollisionError(MembraneError):
    """Native framework memory is active alongside Membrane-managed context (adapters)."""

    def __init__(
        self,
        message: str = (
            "Membrane detected dual memory management: disable the framework's native "
            "conversation memory or Membrane's ConversationManager — not both."
        ),
    ) -> None:
        super().__init__(message)
