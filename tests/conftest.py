import numpy as np
import pytest
import os
from pathlib import Path
import torch
from torch import Tensor


_A = np.ndarray | Tensor

def _canonicalize_array(arr: _A) -> np.ndarray:
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


class NumpySnapshot:
    """Snapshot testing utility for NumPy arrays using .npz format."""
    
    def __init__(
        self, 
        snapshot_dir: str = "tests/_snapshots",
    ):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
    
    def _get_snapshot_path(self, test_name: str) -> Path:
        """Get the path to the snapshot file."""
        return self.snapshot_dir / f"{test_name}.npz"
    
    def assert_match(
        self, 
        actual: _A | dict[str, _A], 
        test_name: str, 
        force_update: bool = False,
        rtol: float = 1e-4, 
        atol: float = 1e-2,
    ):
        """
        Assert that the actual array(s) matches the snapshot.
        
        Args:
            actual: Single NumPy array or dictionary of named arrays
            test_name: The name of the test (used for the snapshot file)
            update: If True, update the snapshot instead of comparing
        """
        snapshot_path = self._get_snapshot_path(test_name)
        
        # Convert single array to dictionary for consistent handling
        arrays_dict = actual if isinstance(actual, dict) else {"array": actual}
        arrays_dict = {
            k: _canonicalize_array(v)
            for k, v in arrays_dict.items()
        }
        
        
        # Load the snapshot
        expected_arrays = dict(np.load(snapshot_path))
        
        # Verify all expected arrays are present
        missing_keys = set(arrays_dict.keys()) - set(expected_arrays.keys())
        if missing_keys:
            raise AssertionError(f"Keys {missing_keys} not found in snapshot for {test_name}")
        
        # Verify all actual arrays are expected
        extra_keys = set(expected_arrays.keys()) - set(arrays_dict.keys())
        if extra_keys:
            raise AssertionError(f"Snapshot contains extra keys {extra_keys} for {test_name}")
        
        # Compare all arrays
        for key in arrays_dict:
            np.testing.assert_allclose(
                _canonicalize_array(arrays_dict[key]),
                expected_arrays[key], 
                rtol=rtol, 
                atol=atol,
                err_msg=f"Array '{key}' does not match snapshot for {test_name}"
            )


# Fixture that can be used in all tests
@pytest.fixture
def numpy_snapshot(request):
    """
    Fixture providing numpy snapshot testing functionality.
    
    Usage:
        def test_my_function(numpy_snapshot):
            result = my_function()
            numpy_snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    match_exact = request.config.getoption("--snapshot-exact", default=False)
    
    # Create the snapshot handler with default settings
    snapshot = NumpySnapshot()
    
    # Patch the assert_match method to include the update flag by default
    original_assert_match = snapshot.assert_match
    
    def patched_assert_match(actual, test_name=None, force_update=force_update, rtol=1e-4, atol=1e-2):
        # If test_name is not provided, use the test function name
        if test_name is None:
            test_name = request.node.name
        if match_exact:
            rtol = atol = 0
        return original_assert_match(actual, test_name=test_name, force_update=force_update, rtol=rtol, atol=atol)
    
    snapshot.assert_match = patched_assert_match
    
    return snapshot


@pytest.fixture
def ts_state_dict(request):
    from .common import FIXTURES_PATH
    import json
    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))
    state_dict = {
        k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
    }
    return state_dict, config



# Model parameters used for model fixture

@pytest.fixture
def n_layers():
    return 3


@pytest.fixture
def vocab_size():
    return 10_000


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def n_queries():
    return 12


@pytest.fixture
def n_keys():
    return 16


@pytest.fixture
def n_heads():
    return 4

@pytest.fixture
def d_head():
    return 16

@pytest.fixture
def d_model(n_heads, d_head):
    return n_heads * d_head

@pytest.fixture
def d_ff():
    return 128

@pytest.fixture
def q(batch_size, n_queries, d_model):
    torch.manual_seed(1)
    return torch.randn(batch_size, n_queries, d_model)

@pytest.fixture
def k(batch_size, n_keys, d_model):
    torch.manual_seed(2)
    return torch.randn(batch_size, n_keys, d_model)

@pytest.fixture
def v(batch_size, n_keys, d_model):
    torch.manual_seed(3)
    return torch.randn(batch_size, n_keys, d_model)

@pytest.fixture
def in_embeddings(batch_size, n_queries, d_model):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)

@pytest.fixture
def mask(batch_size, n_queries, n_keys):
    torch.manual_seed(5)
    return torch.randn(batch_size, n_queries, n_keys) > 0.5

@pytest.fixture
def in_indices(batch_size, n_queries):
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))

@pytest.fixture
def theta():
    return 10000.0

@pytest.fixture
def pos_ids(n_queries):
    return torch.arange(0, n_queries)


# # Example usage:
# def test_single_array(numpy_snapshot):
#     # Sample function that produces a numpy array
#     def my_function():
#         return np.array([[1.0, 2.0], [3.0, 4.0001]])
    
#     result = my_function()
    
#     # Just provide the result - the test name will be inferred
#     numpy_snapshot.assert_match(result)


# def test_multiple_arrays(numpy_snapshot):
#     # Function that produces multiple arrays
#     def my_function():
#         return {
#             "weights": np.array([0.1, 0.2, 0.3]),
#             "biases": np.array([0.01, 0.02]),
#             "gradients": np.array([[0.001, 0.002], [0.003, 0.004]])
#         }
    
#     results = my_function()
    
#     # Test with explicit name and custom tolerances
#     # custom_snapshot = NumpySnapshot()
#     numpy_snapshot.assert_match(
#         results, 
#         "my_special_test",
#         rtol=1e-4,
#         atol=1e-6,
#     )

# def test_state_dict(ts_state_dict):
#     print(ts_state_dict)