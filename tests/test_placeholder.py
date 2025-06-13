# tests/test_placeholder.py
import pytest

# from src.asys_i.orchestration.config_loader import load_config, MasterConfig # Example


def test_sanity():
    """Basic check that pytest is working."""
    assert 1 + 1 == 2


@pytest.mark.hpc
def test_hpc_feature_placeholder():
    """
    Example of a test marked for HPC only.
    Run `pytest -m "not hpc"` to skip this.
    """
    # try:
    # import src.asys_i.hpc.cpp_ringbuffer as cpp_rb
    # except ImportError:
    # pytest.skip("HPC C++ module not available")
    print("Pretending to run HPC test")
    assert True


# @pytest.mark.slow
# def test_integration_placeholder():
#      pass
