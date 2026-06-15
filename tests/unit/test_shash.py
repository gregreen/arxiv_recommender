"""
Unit tests for experiments/shash_distribution.py.
"""

import numpy as np
import pytest
from experiments.shash_distribution import shash_cdf, shash_pdf, shash_logpdf, fit_shash


# ---------------------------------------------------------------------------
# shash_cdf
# ---------------------------------------------------------------------------

class TestShashCdf:
    _PARAMS = {"mu": 0.5, "sigma": 0.05, "delta": 1.0, "epsilon": 0.1}

    @staticmethod
    def _cdf(x: np.ndarray) -> np.ndarray:
        return shash_cdf(x, **TestShashCdf._PARAMS)

    def test_range_zero_to_one(self):
        """CDF values should always fall in [0, 1]."""
        x = np.linspace(-5, 5, 1000)
        cdf = self._cdf(x)
        assert np.all(cdf >= 0.0)
        assert np.all(cdf <= 1.0)

    def test_monotonic(self):
        """CDF should be strictly non-decreasing."""
        x = np.linspace(-5, 5, 2000)
        cdf = self._cdf(x)
        assert np.all(np.diff(cdf) >= 0.0)

    def test_asymptotes(self):
        """CDF should approach 0 for very negative x, 1 for very positive x."""
        assert self._cdf(np.array([-1e6]))[0] < 1e-6
        assert self._cdf(np.array([ 1e6]))[0] > 0.999999

    def test_no_nan_inf(self):
        """No NaN or Inf for a broad input range."""
        x = np.linspace(-50, 50, 10001)
        result = self._cdf(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ---------------------------------------------------------------------------
# shash_pdf
# ---------------------------------------------------------------------------

class TestShashPdf:
    _PARAMS = {"mu": 0.5, "sigma": 0.05, "delta": 1.0, "epsilon": 0.1}

    @staticmethod
    def _pdf(x: np.ndarray) -> np.ndarray:
        return shash_pdf(x, **TestShashPdf._PARAMS)

    def test_nonnegative(self):
        """PDF should be ≥ 0 everywhere."""
        x = np.linspace(-5, 5, 1000)
        pdf = self._pdf(x)
        assert np.all(pdf >= 0.0)

    def test_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        x = np.linspace(-5, 5, 20001)
        pdf = self._pdf(x)
        integral = np.trapz(pdf, x)
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_no_nan_inf(self):
        x = np.linspace(-50, 50, 10001)
        result = self._pdf(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ---------------------------------------------------------------------------
# shash_logpdf
# ---------------------------------------------------------------------------

class TestShashLogpdf:
    _PARAMS = {"mu": 0.5, "sigma": 0.05, "delta": 1.0, "epsilon": 0.1}

    @staticmethod
    def _logpdf(x: np.ndarray) -> np.ndarray:
        return shash_logpdf(x, **TestShashLogpdf._PARAMS)

    def test_no_nan_inf(self):
        x = np.linspace(-50, 50, 10001)
        result = self._logpdf(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_consistent_with_pdf(self):
        """exp(logpdf) should match pdf within numerical tolerance."""
        x = np.linspace(-1, 2, 500)
        logpdf = self._logpdf(x)
        pdf = shash_pdf(x, **self._PARAMS)
        assert not np.any(np.isnan(logpdf))
        # In dense region, relative error should be small
        mask = pdf > 1e-6
        if mask.any():
            ratio = np.exp(logpdf[mask]) / pdf[mask]
            assert np.allclose(ratio, 1.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# fit_shash
# ---------------------------------------------------------------------------

class TestFitShash:
    def test_recovers_known_params_approximately(self):
        """Fitting synthetic data drawn from known SHASH params should recover
        parameters in the right ballpark.  Exact recovery is not expected with
        only 200 samples, but the fit should produce plausible values."""
        rng = np.random.default_rng(42)
        true_params = {"mu": 0.5, "sigma": 0.05, "delta": 1.0, "epsilon": 0.12}
        p = np.linspace(0.001, 0.999, 1000)
        x_true = np.array([_find_quantile(q, true_params) for q in p])
        x = rng.choice(x_true, size=200, replace=True)

        fitted, _ = fit_shash(x)

        # Plausibility checks — not strict equality
        assert 0.4 < fitted["mu"] < 0.6
        assert 0.02 < fitted["sigma"] < 0.15
        assert 0.1 < fitted["delta"] < 3.0
        assert -0.5 < fitted["epsilon"] < 0.5


def _find_quantile(p: float, params: dict) -> float:
    """Inverse CDF via binary search for a SHASH distribution (used for synthetic data)."""
    lo, hi = params["mu"] - 10 * params["sigma"], params["mu"] + 10 * params["sigma"]
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if shash_cdf(np.array([mid]), **params)[0] < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
