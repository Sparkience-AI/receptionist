"""Provider errors with messages safe to display in the chat."""


class ProviderError(RuntimeError):
    """A provider could not complete a response."""

    def __init__(self, message, *, partial=""):
        super().__init__(message)
        self.partial = partial
