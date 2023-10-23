# Code Contribution

* We expect Type Hints in all methods.

You don't need to type hint every variable or function. Start with critical components, or where types might be confusing.
Type hints are not enforced at runtime. They're purely for the developer and tooling.
For large projects, consider gradual typing. You can add hints incrementally.
Type hints, when used judiciously, can make your Python code more readable and maintainable, and can catch potential bugs before runtime.

* We expect Docstrings in all methods and classes.

For modules, the docstring should list the classes, exceptions, and functions (and any other objects) that are exported by the module. For classes, document its methods and instance variables.

## Type Hints

Type hints, introduced in Python 3.5 via PEP 484, provide a way to specify the expected data types for variables, function arguments, and return values. This can greatly improve the readability of your code, facilitate debugging, and enable better tooling (like type checkers and IDE assistance). Here's a guide to get you started with type hints.

### Function and Method Signatures

Type hints can be added to function arguments and return values:
```python
def greet(name: str) -> None:
    print(f"Hello, {name}!")

def add(a: int, b: int) -> int:
    return a + b
```

### Type Hinting Lists, Dictionaries, and Other Collections

For collections, you can use the `typing` module:

```python
from typing import List, Dict

def get_names() -> List[str]:
    return ["Alice", "Bob", "Charlie"]

def get_age_mapping() -> Dict[str, int]:
    return {"Alice": 30, "Bob": 25, "Charlie": 28}
```

### Optional Types

If a variable might be of a certain type or None, use `Optional`:

```python
from typing import Optional

def find_user(username: str) -> Optional[Dict[str, str]]:
    # Return user dict if found, else None
    pass
```

### Union Types

If a variable can be one of several types, use `Union`:

```python
from typing import Union

def process_data(data: Union[str, bytes]) -> None:
    pass
```

### Classes and Type Hints

You can also use type hints in class definitions:

```python
class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age
```

## Docstrings

Docstrings, or documentation strings, are an essential aspect of Python programming. They provide concise descriptions of how a module, class, method, or function works. The PEP-8 style guide suggests a specific format for these descriptions. We'll walk through the process of adding PEP-8 style docstrings to your Python code. This style is commonly associated with Google's Python style guide.

### What is PEP-8

PEP-8 is the Python Enhancement Proposal that provides coding conventions for the Python code comprising the standard library in the main Python distribution. These conventions help in making the code readable and maintainable.

### Why Docstrings

Docstrings provide a built-in system for associating blocks of documentation with modules, classes, methods, and functions. This documentation can be accessed at runtime using the `help()` function or outside the runtime using tools like Sphinx.

### PEP-8 Docstring Format

Here's the basic structure:

```python
def function_name(arg1, arg2):
    """Brief description.

    More detailed description.

    Args:
        arg1 (type): Description of arg1.
        arg2 (type): Description of arg2.

    Returns:
        type: Description of the return value.

    Raises:
        ExceptionType: Description of the circumstances under which the exception is raised.
    """
    pass

```

