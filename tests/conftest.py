import numpy as np
import pytest
import mrcfile


@pytest.fixture
def reference():
    # Nonzero mean and nonunit std discriminate scaling from centering.
    return np.arange(35, dtype=np.float64).reshape(5, 7) / 3 + 2


@pytest.fixture
def write_mrc(tmp_path):
    def write(name, data):
        path = tmp_path / name
        with mrcfile.new(path) as handle:
            handle.set_data(np.asarray(data, dtype=np.float32))
        return path
    return write


def unit_weights(images, reference, std):
    return np.ones(images.shape)


def image_unit_weights(images, reference, std):
    return np.ones(images.shape[0])
