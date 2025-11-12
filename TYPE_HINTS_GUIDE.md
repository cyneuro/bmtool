# Type Hints Implementation Guide for BMTool

## Current Status

### Code Quality Improvements Completed ✅
1. **Formatting**: All 28 Python files formatted with ruff
2. **Linting**: All linting issues resolved (imports sorted, style issues fixed)
3. **Type Hints**: Started adding type hints (currently 18.4% coverage)

### Type Hint Coverage by Module

| Module | Functions | Coverage | Priority |
|--------|-----------|----------|----------|
| **Already Complete (100%)** |
| bmplot/entrainment.py | 9/9 | 100% | ✅ Done |
| bmplot/spikes.py | 5/5 | 100% | ✅ Done |
| analysis/spikes.py | 8/8 | 100% | ✅ Done |
| bmplot/lfp.py | 6/6 | 100% | ✅ Done |
| **Good Progress** |
| bmplot/connections.py | 19/22 | 86.4% | 🟡 Nearly done |
| analysis/entrainment.py | 8/12 | 66.7% | 🟡 Partial |
| analysis/lfp.py | 10/19 | 52.6% | 🟡 Partial |
| synapses.py | 27/69 | 39.1% | 🟡 Partial |
| **Needs Attention** |
| singlecell.py | 7/43 | 16.3% | ⚠️ Low coverage |
| util/util.py | 2/67 | 3.0% | ❌ Critical |
| **No Coverage Yet** |
| connectors.py | 6/133 | 4.5% | ❌ Started |
| util/celltuner.py | 0/110 | 0.0% | ❌ Not started |
| util/commands.py | 0/50 | 0.0% | ❌ Not started |
| plot_commands.py | 0/18 | 0.0% | ❌ Not started |
| SLURM.py | 0/21 | 0.0% | ❌ Not started |

**Overall: 115/591 functions (19.5%)**

## Type Hint Examples

### Example 1: Simple Function with NumPy Arrays

```python
# Before
def num_prop(ratio, N):
    """Calculate numbers of total N in proportion to ratio."""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0))
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)

# After
from typing import Union
from numpy.typing import ArrayLike, NDArray
import numpy as np

def num_prop(ratio: ArrayLike, N: int) -> NDArray[np.integer]:
    """Calculate numbers of total N in proportion to ratio."""
    ratio = np.asarray(ratio)
    p = np.cumsum(np.insert(ratio.ravel(), 0, 0))
    return np.diff(np.round(N / p[-1] * p).astype(int)).reshape(ratio.shape)
```

### Example 2: Function with Optional Parameters

```python
# Before
def decision(prob, size=None):
    """Make random decision(s) based on input probability."""
    return rng.random(size) < prob

# After
from typing import Optional, Union
from numpy.typing import NDArray
import numpy as np

def decision(prob: float, size: Optional[Union[int, tuple]] = None) -> Union[bool, NDArray[np.bool_]]:
    """Make random decision(s) based on input probability."""
    return rng.random(size) < prob
```

### Example 3: Method in a Class

```python
# Before
class DistantDependentProbability(ProbabilityFunction):
    def __init__(self, min_dist=0.0, max_dist=np.inf):
        self.min_dist, self.max_dist = min_dist, max_dist
    
    def decisions(self, dist):
        dist = np.asarray(dist)
        # ... implementation
        return dec

# After
from numpy.typing import ArrayLike, NDArray
import numpy as np

class DistantDependentProbability(ProbabilityFunction):
    def __init__(self, min_dist: float = 0.0, max_dist: float = np.inf) -> None:
        self.min_dist, self.max_dist = min_dist, max_dist
    
    def decisions(self, dist: ArrayLike) -> NDArray[np.bool_]:
        dist = np.asarray(dist)
        # ... implementation
        return dec
```

## Common Type Patterns in BMTool

### NumPy Types
```python
from numpy.typing import ArrayLike, NDArray
import numpy as np

# Input that accepts array-like (lists, tuples, arrays)
def func(data: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(data)
    return arr * 2

# Specific numpy array types
def process_ints(data: NDArray[np.integer]) -> int:
    return int(np.sum(data))

def process_bools(flags: NDArray[np.bool_]) -> bool:
    return bool(np.all(flags))
```

### Pandas Types
```python
import pandas as pd

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()

def get_column(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col]
```

### Optional and Union Types
```python
from typing import Optional, Union

# Optional for parameters that can be None
def func(x: int, y: Optional[int] = None) -> int:
    return x if y is None else x + y

# Union for multiple possible types
def process(data: Union[int, float, list]) -> float:
    if isinstance(data, list):
        return sum(data)
    return float(data)
```

### NEURON Objects
```python
# NEURON doesn't have type stubs, so use Any or object
from typing import Any

def setup_cell(cell: Any) -> None:  # or just object
    """Setup a NEURON cell object."""
    cell.soma[0].insert('pas')
```

## Recommended Implementation Strategy

### Phase 1: High-Impact Public APIs (Recommended to complete)
1. **connectors.py** (133 functions) - Core connection building
2. **synapses.py** (42 functions need hints) - Synaptic dynamics
3. **singlecell.py** (36 functions need hints) - Single cell analysis

### Phase 2: Utilities and Commands
4. **util/util.py** (65 functions) - Shared utilities
5. **util/celltuner.py** (110 functions) - Cell parameter tuning
6. **util/commands.py** (50 functions) - CLI commands
7. **SLURM.py** (21 functions) - SLURM job management

### Phase 3: Polish
8. Fill gaps in partially-typed modules
9. Add type hints to command-line interfaces
10. Add type hints to plotting utilities

## Tools and Validation

### Running Type Checks with MyPy

```bash
# Install mypy
pip install mypy

# Check a single file
mypy bmtool/connectors.py

# Check entire package
mypy bmtool/

# Check with specific configuration
mypy --config-file mypy.ini bmtool/
```

### Configuration (mypy.ini)

See the `mypy.ini` file created in this repository for the recommended configuration.

### Using MonkeyType for Automatic Type Inference

```bash
# Install monkeytype
pip install monkeytype

# Run your code with monkeytype
monkeytype run your_script.py

# Generate stubs
monkeytype stub bmtool.connectors

# Apply stubs to file
monkeytype apply bmtool.connectors
```

## Benefits of Adding Type Hints

1. **Better IDE Support**: Autocomplete, inline documentation, error detection
2. **Catch Bugs Early**: Type checkers can find bugs before runtime
3. **Documentation**: Types serve as inline documentation
4. **Refactoring Safety**: Types make refactoring safer and easier
5. **Better Collaboration**: Clear interfaces help team understanding

## Gradual Adoption Approach

Type hints can be added incrementally:

1. Start with function signatures (parameters and return types)
2. Add hints to new code as it's written
3. Add hints when modifying existing code
4. Don't need to annotate internal/private functions initially
5. Focus on public APIs first

## Resources

- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [Python Typing Module](https://docs.python.org/3/library/typing.html)
- [NumPy Type Hints](https://numpy.org/devdocs/reference/typing.html)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Real Python - Type Checking](https://realpython.com/python-type-checking/)
