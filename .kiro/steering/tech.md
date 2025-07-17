# Technical Stack & Build System

## Technology Stack
- **Language**: Python 3.8+
- **Testing Framework**: pytest
- **Data Structures**: Custom data models using Python dataclasses
- **Algorithm Approach**: Dynamic Programming with word-boundary awareness

## Dependencies
- Python 3.8+ (core language)
- pytest (for testing)
- dataclasses (built-in for Python 3.7+)

## Development Tools
- pytest for unit and integration testing
- pytest-cov for test coverage reporting

## Common Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=shortest_combined_string

# Run specific test file
python -m pytest tests/test_models.py -v
```

### Running the CLI
```bash
# Basic usage
python -m shortest_combined_string.cli "this is a red vase" "his son freddy love vase"

# Disable colored output
python -m shortest_combined_string.cli --no-color "hello world" "world test"

# Quote strings in the output for clarity
python -m shortest_combined_string.cli --quote "to be or not to be" "that is the question"
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd shortest-combined-string

# Install dependencies
pip install -r requirements.txt
```

## Code Style & Conventions
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Document all classes and functions with docstrings
- Use dataclasses for structured data models
- Prefer explicit error handling with custom exceptions
- Use descriptive variable names that reflect their purpose
- Write comprehensive unit tests for all components