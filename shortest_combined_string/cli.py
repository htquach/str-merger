#!/usr/bin/env python
"""
Command-line interface for the Shortest Combined String algorithm.

This module provides a CLI that accepts two strings as arguments and displays
the result, metrics, and validation status of the shortest combined string algorithm.
"""

import argparse
import sys
from typing import List, Optional
from colorama import init, Fore, Style

from .shortest_combined_string import ShortestCombinedString
from .models import AlgorithmResult
from .exceptions import ShortestCombinedStringError


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Find the shortest combined string containing two input sentences as subsequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m shortest_combined_string.cli "this is a red vase" "his son freddy love vase"
  python -m shortest_combined_string.cli --no-color "hello world" "world test"
  python -m shortest_combined_string.cli --quote "to be or not to be" "that is the question"
        """
    )
    
    parser.add_argument(
        "string1",
        help="First input string"
    )
    
    parser.add_argument(
        "string2",
        help="Second input string"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    parser.add_argument(
        "--quote",
        action="store_true",
        help="Quote the input and output strings in the display"
    )
    
    return parser.parse_args(args)


def format_result(result: AlgorithmResult, use_color: bool = True, quote: bool = False) -> str:
    """
    Format the algorithm result for display.
    
    Args:
        result: Algorithm result to format
        use_color: Whether to use colored output
        quote: Whether to quote strings in the output
        
    Returns:
        Formatted result string
    """
    # Initialize colorama for Windows support
    if use_color:
        init()
    
    # Format strings with quotes if requested
    def fmt_str(s: str) -> str:
        return f'"{s}"' if quote else s
    
    # Prepare colored output functions
    def green(s: str) -> str:
        return f"{Fore.GREEN}{s}{Style.RESET_ALL}" if use_color else s
    
    def red(s: str) -> str:
        return f"{Fore.RED}{s}{Style.RESET_ALL}" if use_color else s
    
    def yellow(s: str) -> str:
        return f"{Fore.YELLOW}{s}{Style.RESET_ALL}" if use_color else s
    
    def blue(s: str) -> str:
        return f"{Fore.BLUE}{s}{Style.RESET_ALL}" if use_color else s
    
    def bold(s: str) -> str:
        return f"{Style.BRIGHT}{s}{Style.RESET_ALL}" if use_color else s
    
    # Build the output
    lines = []
    
    # Header
    lines.append(bold("=== Shortest Combined String Result ==="))
    lines.append("")
    
    # Input strings
    lines.append(f"{bold('Input String 1:')} {fmt_str(result.metrics.original_s1_length * '*')}")
    lines.append(f"{bold('Input String 2:')} {fmt_str(result.metrics.original_s2_length * '*')}")
    
    # We need to access the original input strings, but they're not directly available in the metrics
    # In a real implementation, we would need to modify the AlgorithmResult to store the original strings
    lines.append("")
    
    # Result
    status = green("VALID") if result.is_valid else red("INVALID")
    lines.append(f"{bold('Result:')} {fmt_str(result.combined_string)} [{status}]")
    lines.append("")
    
    # Metrics
    lines.append(bold("Optimization Metrics:"))
    lines.append(f"  Original lengths: s1={result.metrics.original_s1_length}, s2={result.metrics.original_s2_length}")
    lines.append(f"  Combined length: {result.metrics.combined_length}")
    lines.append(f"  Total savings: {result.metrics.total_savings} characters")
    
    # Format compression ratio with color based on effectiveness
    ratio = result.metrics.compression_ratio
    ratio_str = f"{ratio:.3f}"
    if ratio <= 0.6:
        ratio_colored = green(ratio_str)  # Excellent compression
    elif ratio <= 0.8:
        ratio_colored = blue(ratio_str)   # Good compression
    else:
        ratio_colored = yellow(ratio_str) # Minimal compression
    
    lines.append(f"  Compression ratio: {ratio_colored}")
    lines.append("")
    
    # Warnings
    if result.processing_warnings:
        lines.append(bold("Processing Warnings:"))
        for warning in result.processing_warnings:
            lines.append(f"  {yellow('!')} {warning}")
        lines.append("")
    
    # Validation errors
    if result.validation_errors:
        lines.append(bold("Validation Errors:"))
        for error in result.validation_errors:
            lines.append(f"  {red('✗')} {error}")
        lines.append("")
    
    return "\n".join(lines)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Parse arguments
        parsed_args = parse_args(args)
        
        # Store original input strings for display
        string1 = parsed_args.string1
        string2 = parsed_args.string2
        
        # Create algorithm instance
        algorithm = ShortestCombinedString()
        
        # Process the strings
        result = algorithm.combine(string1, string2)
        
        # Modify the format_result function to include the original strings
        def enhanced_format_result(result, use_color, quote):
            # Get the standard formatted result
            formatted = format_result(result, use_color, quote)
            
            # Format strings with quotes if requested
            def fmt_str(s: str) -> str:
                return f'"{s}"' if quote else s
            
            # Prepare colored output functions
            def bold(s: str) -> str:
                return f"{Style.BRIGHT}{s}{Style.RESET_ALL}" if use_color else s
            
            # Insert the actual input strings after the asterisk representation
            lines = formatted.split('\n')
            insert_position = 4  # After the second input string line and empty line
            
            # Insert the actual input strings
            lines.insert(insert_position, f"{bold('Actual Input 1:')} {fmt_str(string1)}")
            lines.insert(insert_position + 1, f"{bold('Actual Input 2:')} {fmt_str(string2)}")
            
            return '\n'.join(lines)
        
        # Display the enhanced result
        formatted_result = enhanced_format_result(
            result, 
            use_color=not parsed_args.no_color,
            quote=parsed_args.quote
        )
        print(formatted_result)
        
        # Return success if the result is valid, error if invalid
        return 0 if result.is_valid else 1
        
    except ShortestCombinedStringError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())