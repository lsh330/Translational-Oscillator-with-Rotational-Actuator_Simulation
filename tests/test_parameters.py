"""Tests for the parameters module."""

import numpy as np
import pytest

from parameters.physical import PhysicalParams
from parameters.derived import compute_derived
from parameters.packing import pack_params, unpack_params
from parameters.equilibrium import equilibrium
from parameters.config import SystemConfig


class TestPhysicalParams:
    def test_default_values(self):
        pp = PhysicalParams()
        assert pp.M == 1.3608
        assert pp.m == 0.096
        assert pp.e == 0.0592
        assert pp.k == 186.3
        assert pp.I == 0.0002175

    def test_custom_values(self):
        pp = PhysicalParams(M=2.0, m=0.1, e=0.05, k=200.0, I=0.001)
        assert pp.M == 2.0


class TestDerivedParams:
    def test_derived_computation(self):
        pp = PhysicalParams()
        dp = compute_derived(pp)
        assert dp.Mt == pytest.approx(1.3608 + 0.096, rel=1e-6)
        assert dp.me == pytest.approx(0.096 * 0.0592, rel=1e-6)
        assert dp.I_eff == pytest.approx(0.0002175 + 0.096 * 0.0592 ** 2, rel=1e-6)
        assert dp.k == 186.3


class TestPacking:
    def test_round_trip(self):
        pp = PhysicalParams()
        dp = compute_derived(pp)
        p = pack_params(dp)
        dp2 = unpack_params(p)
        assert dp.Mt == pytest.approx(dp2.Mt, rel=1e-10)
        assert dp.me == pytest.approx(dp2.me, rel=1e-10)
        assert dp.I_eff == pytest.approx(dp2.I_eff, rel=1e-10)
        assert dp.k == pytest.approx(dp2.k, rel=1e-10)

    def test_pack_shape(self):
        pp = PhysicalParams()
        dp = compute_derived(pp)
        p = pack_params(dp)
        assert p.shape == (4,)
        assert p.dtype == np.float64


class TestEquilibrium:
    def test_zero_equilibrium(self):
        eq = equilibrium()
        np.testing.assert_array_equal(eq, np.zeros(4))
        assert eq.dtype == np.float64


class TestSystemConfig:
    def test_default_config(self):
        cfg = SystemConfig()
        assert cfg.physical.M == 1.3608
        assert cfg.derived.Mt == pytest.approx(1.4568, rel=1e-3)

    def test_pack(self):
        cfg = SystemConfig()
        p = cfg.pack()
        assert p.shape == (4,)

    def test_equilibrium(self):
        cfg = SystemConfig()
        np.testing.assert_array_equal(cfg.equilibrium, np.zeros(4))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            SystemConfig(M=-1.0)
        with pytest.raises(ValueError):
            SystemConfig(m=0)
        with pytest.raises(ValueError):
            SystemConfig(k="abc")

    def test_custom_config(self):
        cfg = SystemConfig(M=2.0, m=0.2, e=0.1, k=100.0, I=0.001)
        assert cfg.derived.Mt == pytest.approx(2.2, rel=1e-6)
        assert cfg.derived.me == pytest.approx(0.02, rel=1e-6)
