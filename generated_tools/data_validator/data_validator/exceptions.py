"""data_validator exceptions."""


class DataValidatorError(Exception):
    """Base exception for data_validator."""
    pass


class ValidationError(DataValidatorError):
    """Raised when validation fails."""
    pass


class ConfigurationError(DataValidatorError):
    """Raised when configuration is invalid."""
    pass


class ProcessingError(DataValidatorError):
    """Raised when processing fails."""
    pass
