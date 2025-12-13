"""log_analyzer command-line interface."""

import argparse
import sys
import logging

from .core import LogAnalyzer
from .exceptions import LogAnalyzerError

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="log_analyzer",
        description="Analyze and summarize log files with pattern matching"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Configuration file path"
    )
    return parser


def main(argv: list = None) -> int:
    """Main CLI entry point.
    
    Args:
        argv: Command line arguments.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    
    try:
        instance = LogAnalyzer()
        result = instance.run()
        print(f"Success: {result}")
        return 0
    except LogAnalyzerError as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
