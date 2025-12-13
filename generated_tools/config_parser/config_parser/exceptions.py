"""config_parser exceptions."""


class ConfigParserError(Exception):
    """Base exception for config_parser."""
    pass


class ValidationError(ConfigParserError):
    """Raised when validation fails."""
    pass


class ConfigurationError(ConfigParserError):
    """Raised when configuration is invalid."""
    pass


class ProcessingError(ConfigParserError):
    """Raised when processing fails."""
    pass
