# Contributing to Fake Review Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain professional communication

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/fake-review-detection/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
   - Error logs if applicable

### Suggesting Features

1. Open an issue with label `enhancement`
2. Describe the feature and its use case
3. Explain why it would be valuable
4. Provide examples if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow coding standards (see below)
   - Add tests for new features
   - Update documentation

4. **Commit your changes**
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Describe what you changed and why
   - Reference any related issues
   - Wait for review

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/fake-review-detection.git
cd fake-review-detection

# Add upstream remote
git remote add upstream https://github.com/yourusername/fake-review-detection.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if exists

# Install pre-commit hooks (optional)
pre-commit install
```

## Coding Standards

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small
- Maximum line length: 100 characters

**Example:**
```python
def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocess review text for model input.
    
    Args:
        text: Raw review text
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Preprocessed text string
    """
    # Implementation
    pass
```

### JavaScript/React Code

- Use ES6+ features
- Follow Airbnb style guide
- Use functional components with hooks
- Maintain component modularity
- Use meaningful component names

**Example:**
```javascript
const ReviewCard = ({ review, prediction, confidence }) => {
  return (
    <div className="card">
      <p>{review}</p>
      <span className={prediction === 'Fake' ? 'text-red-600' : 'text-green-600'}>
        {prediction} ({(confidence * 100).toFixed(1)}%)
      </span>
    </div>
  );
};
```

## Testing

### Running Tests

```bash
# Backend tests
cd tests
python -m pytest -v

# Frontend tests
cd frontend
npm test
```

### Writing Tests

- Add tests for new features
- Maintain test coverage above 80%
- Test edge cases
- Use descriptive test names

**Example:**
```python
def test_preprocessor_removes_urls():
    """Test that URLs are removed from text."""
    preprocessor = TextPreprocessor()
    text = "Check http://example.com for info"
    result = preprocessor.clean_text(text)
    assert "http" not in result
```

## Documentation

- Update README.md for major changes
- Add docstrings to new functions
- Update API documentation
- Include code examples
- Keep documentation clear and concise

## Project Structure

```
fake-review-detection/
├── backend/              # Python backend
│   ├── app.py           # Flask API
│   ├── preprocessing.py # Text preprocessing
│   ├── model_trainer.py # Model training
│   └── predictor.py     # Inference
├── frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # Reusable components
│   │   ├── pages/       # Page components
│   │   └── api/         # API calls
├── models/              # Saved models
├── data/                # Datasets
├── notebooks/           # Jupyter notebooks
└── tests/               # Test files
```

## Areas for Contribution

### High Priority
- [ ] Add more preprocessing techniques
- [ ] Implement deep learning models (BERT, RoBERTa)
- [ ] Add multi-language support
- [ ] Improve model explainability (SHAP, LIME)
- [ ] Add API authentication
- [ ] Implement rate limiting

### Medium Priority
- [ ] Add more evaluation metrics
- [ ] Create model comparison dashboard
- [ ] Add batch prediction optimization
- [ ] Implement A/B testing framework
- [ ] Add user feedback collection

### Nice to Have
- [ ] Mobile app
- [ ] Browser extension
- [ ] Email integration
- [ ] Slack/Discord bot
- [ ] Advanced visualizations

## Commit Message Guidelines

Format: `type(scope): description`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add batch prediction endpoint
fix(preprocessing): handle empty review text
docs(readme): update installation instructions
test(api): add tests for predict endpoint
```

## Review Process

1. **Automated Checks:** CI/CD runs tests and linting
2. **Code Review:** Maintainer reviews code quality
3. **Testing:** Verify functionality works as expected
4. **Merge:** Once approved, PR is merged

## Getting Help

- Open an issue for questions
- Join discussions on GitHub
- Check existing documentation
- Contact maintainers

## Recognition

Contributors are acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to make fake review detection better! 🎉
