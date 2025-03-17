# AI Wetter Project Assistant

## Commands
- Run app: `python -m app.app`
- Lint: `flake8 app tests`
- Type check: `mypy app`
- Run tests: `pytest tests`
- Run single test: `pytest tests/path_to_test.py::TestClass::test_method -v`

## Code Style
- **Imports**: Group by standard lib, third-party, and local modules with blank lines between
- **Formatting**: Use Black with default settings (line length 88)
- **Types**: Use type hints for all function signatures and return values
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- **Error handling**: Use specific exception types, log errors with app.utils.logging
- **Documentation**: Docstrings in Google format for all public functions and classes
- **Structure**: Keep modules focused on single responsibility, use composition over inheritance
- **Storage**: Save user data in data/ directory using utils.storage module