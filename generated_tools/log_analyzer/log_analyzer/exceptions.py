"""log_analyzer exceptions."""


class LogAnalyzerError(Exception):
    """Base exception for log_analyzer."""
    pass


class ValidationError(LogAnalyzerError):
    """Raised when validation fails."""
    pass


class ConfigurationError(LogAnalyzerError):
    """Raised when configuration is invalid."""
    pass


class ProcessingError(LogAnalyzerError):
    """Raised when processing fails."""
    pass
