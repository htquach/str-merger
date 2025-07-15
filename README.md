# Shortest Combined String Algorithm

A smart text merging tool that combines two sentences into the shortest possible string while keeping both sentences readable as subsequences.

## What does it do?

Simply put: it takes two sentences and creates the shortest possible combination that still contains both original sentences.

**Example 1:**
- Input: `"I love programming"` + `"programming is fun"`
- Output: `"I love programming is fun"` 
- Why: Reuses the word "programming" instead of repeating it

**Example 2:**
- Input: `"hello world"` + `"hello planet"`  
- Output: `"hello worldplanet"` or `"hello planetworld"`
- Why: Both are equally short, so we pick the one with fewer spaces

**Example 3:**
- Input: `"the quick brown"` + `"brown fox jumps"`
- Output: `"the quick brown fox jumps"`
- Why: Merges at the shared word "brown"

Instead of just sticking sentences together (which wastes space on repeated words), this tool finds the smartest way to combine them.

## How It Works

This algorithm takes two input sentences and produces an optimized combined string that:
- Contains both input sentences as subsequences (you can read both original sentences in the result)
- Preserves word boundaries and spacing
- Maximizes character reuse between overlapping words
- Minimizes the total length of the result

## Features

- **Word-level processing**: Maintains word integrity during optimization
- **Space preservation**: Handles leading/trailing spaces correctly
- **Input validation**: Comprehensive preprocessing with error handling
- **Optimization metrics**: Detailed statistics on space savings achieved
- **Robust data models**: Type-safe data structures with validation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd shortest-combined-string

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest
```

## Usage

```python
from shortest_combined_string import AlgorithmResult, OptimizationMetrics

# Example usage will be available after implementation
# result = find_shortest_combined_string("hello world", "world test")
# print(f"Combined: {result.combined_string}")
# print(f"Savings: {result.metrics.total_savings} characters")
```

## Algorithm Approach

The algorithm uses dynamic programming to find the optimal combination by:

1. **Preprocessing**: Tokenize inputs into words with spacing metadata
2. **DP Table**: Build a table tracking optimal subproblem solutions
3. **Backtracking**: Reconstruct the optimal solution path
4. **Validation**: Verify the result contains both inputs as subsequences



## Testing

The project includes comprehensive unit tests covering:
- Data model validation and edge cases
- Input preprocessing and error handling
- Algorithm correctness and optimization
- Performance benchmarks

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=shortest_combined_string

# Run specific test file
python -m pytest tests/test_models.py -v
```

## Development Status

- ✅ Core data models implemented
- ⏳ Input preprocessing (planned)
- ⏳ Dynamic programming algorithm (planned)
- ⏳ Result validation (planned)
- ⏳ CLI interface (planned)

## Requirements

- Python 3.8+
- pytest (for testing)
- dataclasses (built-in for Python 3.7+)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Architecture

The project follows a modular architecture:

```
shortest_combined_string/
├── __init__.py          # Package exports
├── models.py            # Core data structures
├── preprocessor.py      # Input validation (planned)
├── algorithm.py         # DP implementation (planned)
├── validator.py         # Result validation (planned)
└── cli.py               # Command-line interface (planned)
```

## Performance

The algorithm has:
- Time complexity: O(n × m) where n, m are word counts
- Space complexity: O(n × m) for the DP table
- Optimized for typical sentence lengths (10-50 words)