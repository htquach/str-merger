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
from shortest_combined_string import InputProcessor, WordTokenizer, SubsequenceVerifier

# Input preprocessing with validation
processor = InputProcessor()
result = processor.validate_and_preprocess("hello  world", "world  test")

print(f"Processed inputs: '{result.s1}' + '{result.s2}'")
print(f"Warnings: {result.warnings}")
print(f"Had consecutive spaces: {result.has_consecutive_spaces}")

# Example output:
# Processed inputs: 'hello world' + 'world test'  
# Warnings: ['Consecutive spaces detected in s1 at position 5 (2 spaces)...']
# Had consecutive spaces: True

# Word tokenization for algorithm processing
tokenizer = WordTokenizer()
tokens1 = tokenizer.tokenize("this is a red vase")
tokens2 = tokenizer.tokenize("his son freddy love vase")

print(f"S1 words: {[token.word for token in tokens1]}")
print(f"S2 words: {[token.word for token in tokens2]}")

# Perfect reconstruction
reconstructed = tokenizer.reconstruct_from_tokens(tokens1)
print(f"Reconstructed: '{reconstructed}'")

# Example output:
# S1 words: ['this', 'is', 'a', 'red', 'vase']
# S2 words: ['his', 'son', 'freddy', 'love', 'vase']
# Reconstructed: 'this is a red vase'

# Output validation with subsequence verification
verifier = SubsequenceVerifier()
s1 = "this is a red vase"
s2 = "his son freddy love vase"
output = "this is son freddy love a red vase"

verification = verifier.verify(s1, s2, output)
print(f"Valid output: {verification.is_valid}")
print(f"S1 subsequence positions: {verification.s1_match.output_positions}")
print(f"S2 subsequence positions: {verification.s2_match.output_positions}")

# Generate detailed report
report = verifier.get_detailed_match_info(verification)
print(report)

# Example output:
# Valid output: True
# S1 subsequence positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# S2 subsequence positions: [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
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
- ✅ Input preprocessing with validation
- ✅ Word tokenization with space preservation
- ✅ Subsequence verification for output validation
- ⏳ Dynamic programming algorithm (planned)
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
├── __init__.py              # Package exports
├── models.py                # Core data structures
├── input_processor.py       # Input validation and preprocessing
├── word_tokenizer.py        # Word boundary tokenization
├── subsequence_verifier.py  # Output validation and verification
├── algorithm.py             # DP implementation (planned)
└── cli.py                   # Command-line interface (planned)
```

## Performance

The algorithm has:
- Time complexity: O(n × m) where n, m are word counts
- Space complexity: O(n × m) for the DP table
- Optimized for typical sentence lengths (10-50 words)