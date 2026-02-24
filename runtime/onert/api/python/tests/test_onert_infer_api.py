import os
import pytest
import numpy as np
from onert import infer
from onert.native.libnnfw_api_pybind import tensorinfo


@pytest.fixture(scope="module")
def nnpackage_path():
    path = os.environ.get("ONERT_TEST_NNPACKAGE")
    if not path or not os.path.isdir(path):
        pytest.skip("Set ONERT_TEST_NNPACKAGE env var to a valid nnpackage directory")
    return path


@pytest.fixture(params=["cpu"])
def session(nnpackage_path, request):
    return infer.session(nnpackage_path, backends=request.param)


def test_get_inputs_tensorinfo(session):
    infos = session.get_inputs_tensorinfo()
    assert isinstance(infos, list)
    assert infos, "There should be at least one input tensorinfo"
    for info in infos:
        assert isinstance(info, tensorinfo)
        assert hasattr(info, "dtype") and hasattr(info, "rank") and hasattr(info, "dims")
        assert len(info.dims) >= info.rank


def test_infer_output_shapes(session):
    infos = session.get_inputs_tensorinfo()
    dummy = []
    for info in infos:
        shape = tuple([dim if dim > 0 else 1 for dim in info.dims[:info.rank]])
        dummy.append(np.zeros(shape, dtype=info.dtype))
    result = session.infer(dummy)
    outputs = result if isinstance(result, list) else [result]
    assert all(isinstance(x, np.ndarray) for x in outputs)
    out_infos = session.get_outputs_tensorinfo()
    assert len(outputs) == len(out_infos)
    for arr, info in zip(outputs, out_infos):
        expected = tuple(info.dims[:info.rank])
        assert arr.shape == expected


def test_infer_with_metrics(session):
    infos = session.get_inputs_tensorinfo()
    dummy = []
    for info in infos:
        shape = tuple([dim if dim > 0 else 1 for dim in info.dims[:info.rank]])
        dummy.append(np.zeros(shape, dtype=info.dtype))
    outputs, metrics = session.infer(dummy, measure=True)
    assert "run_time_ms" in metrics and isinstance(metrics["run_time_ms"], float)


def test_random_dynamic_shape(session):
    infos = session.get_inputs_tensorinfo()
    dummy = []
    for info in infos:
        dims = [np.random.randint(1, 5) if d == -1 else d for d in info.dims[:info.rank]]
        dummy.append(np.random.rand(*tuple(dims)).astype(info.dtype))
    # should run without error
    session.infer(dummy)
