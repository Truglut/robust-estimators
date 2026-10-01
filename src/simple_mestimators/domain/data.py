from dataclasses import dataclass
from typing import Literal

import numpy as np

VarianceMode = Literal["per-pixel", "global"]
ImageSpace = Literal["real", "fourier"]


@dataclass
class ImageBatch:
    real: np.ndarray
    fourier: np.ndarray | None = None
    ctf: np.ndarray | None = None
    norm: Literal["ortho", "forward", "backward"] = "ortho"

    def __post_init__(self):
        if self.real.ndim != 3:
            raise ValueError(
                f"Expected image batch with shape (n, h, w), got {self.real.shape}."
            )

    @property
    def real_shape(self) -> tuple[int, int]:
        return self.real.shape[1:]

    @property
    def rfft_shape(self) -> tuple[int, int]:
        h, w = self.real_shape
        return (h, w // 2 + 1)

    @property
    def n_images(self) -> int:
        return self.real.shape[0]

    def ensure_fourier(self) -> np.ndarray:
        if self.fourier is None:
            self.fourier = np.fft.rfft2(self.real, norm=self.norm)

        return self.fourier

    def real_variance(self, mode: VarianceMode = "per-pixel") -> np.ndarray:
        """Compute the variance of real space images"""
        per_pixel_variance = self.real.var(axis=0, ddof=1)

        if mode == "per-pixel":
            return per_pixel_variance

        return np.asarray(per_pixel_variance.mean())

    def fourier_residual_variance(self, mode: VarianceMode = "per-pixel") -> np.ndarray:
        fourier_images = self.ensure_fourier()
        ddof = 1 if self.n_images > 1 else 0
        per_freq_variance = fourier_images.var(axis=0, ddof=ddof)

        if mode == "per-pixel":
            return per_freq_variance
        if mode == "global":
            return np.asarray(per_freq_variance.mean())

        raise ValueError(f"Unsupported variance mode: {mode}")

    def select_space(self, space: ImageSpace) -> np.ndarray:
        if space == "real":
            return self.real

        if space == "fourier":
            return self.ensure_fourier()

        raise ValueError(f"Unsupported image space: {space}")

    def select_space_data(
        self, space: ImageSpace
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if space == "real":
            return self.real, self.real_variance(), None

        if space == "fourier":
            return self.ensure_fourier(), self.fourier_residual_variance(), self.ctf

        raise ValueError(f"Unsupported image space: {space}")
