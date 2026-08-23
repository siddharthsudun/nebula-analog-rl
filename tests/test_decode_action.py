"""The action decoder must be a coordinate change, not a branch.

The previous implementation inferred the domain with
`x = (x + 1) / 2 if x < 0 else x`, which put a cliff at exactly zero — the point a
tanh-squashed policy visits most often. These tests pin the properties that make the
mapping usable by an optimizer: monotonic, continuous, and endpoint-exact.
"""
from __future__ import annotations

import numpy as np
import pytest

from eqrl.circuits.ctle import ACTION_SPACE, DesignVars, decode_action, encode_action

FIELDS = list(ACTION_SPACE)


def vals(a, domain):
    dv = decode_action(a, domain=domain)
    return [getattr(dv, f) for f in FIELDS]


class TestEndpoints:

    def test_unit_endpoints_hit_the_range_exactly(self):
        lo = vals([0.0] * 7, "unit")
        hi = vals([1.0] * 7, "unit")
        for f, l, h in zip(FIELDS, lo, hi):
            assert l == pytest.approx(ACTION_SPACE[f][0], rel=1e-12), f
            assert h == pytest.approx(ACTION_SPACE[f][1], rel=1e-12), f

    def test_pm1_endpoints_hit_the_same_range(self):
        lo = vals([-1.0] * 7, "pm1")
        hi = vals([1.0] * 7, "pm1")
        for f, l, h in zip(FIELDS, lo, hi):
            assert l == pytest.approx(ACTION_SPACE[f][0], rel=1e-12), f
            assert h == pytest.approx(ACTION_SPACE[f][1], rel=1e-12), f

    def test_the_two_domains_are_the_same_map(self):
        """pm1(x) must equal unit((x+1)/2) everywhere, not just at the ends."""
        for x in np.linspace(-1.0, 1.0, 41):
            a = vals([x] * 7, "pm1")
            b = vals([(x + 1) / 2] * 7, "unit")
            for f, u, v in zip(FIELDS, a, b):
                assert u == pytest.approx(v, rel=1e-12), f"{f} at x={x}"

    def test_pm1_zero_is_the_midpoint_not_the_floor(self):
        """The old bug: a=+0.0 collapsed to the range minimum."""
        mid = vals([0.0] * 7, "pm1")
        for f, v in zip(FIELDS, mid):
            lo, hi = ACTION_SPACE[f]
            assert v > lo * 1.000001 or lo == 0, (
                f"{f}={v:g} sits at its range floor {lo:g} for action 0.0"
            )


class TestMonotoneAndContinuous:

    @pytest.mark.parametrize("domain,xs", [
        ("unit", np.linspace(0.0, 1.0, 201)),
        ("pm1", np.linspace(-1.0, 1.0, 201)),
    ])
    def test_every_parameter_is_monotonic(self, domain, xs):
        series = {f: [] for f in FIELDS}
        for x in xs:
            for f, v in zip(FIELDS, vals([x] * 7, domain)):
                series[f].append(v)
        for f, ys in series.items():
            d = np.diff(ys)
            assert np.all(d >= -1e-18), (
                f"{f} is not monotonic in domain {domain}: worst step {d.min():.3g}"
            )

    @pytest.mark.parametrize("domain,around", [("unit", 0.5), ("pm1", 0.0)])
    def test_no_discontinuity_at_the_policy_centre(self, domain, around):
        """A tanh policy concentrates near the centre; a cliff there is fatal."""
        eps = 1e-6
        lo = vals([around - eps] * 7, domain)
        hi = vals([around + eps] * 7, domain)
        for f, a, b in zip(FIELDS, lo, hi):
            span = ACTION_SPACE[f][1] - ACTION_SPACE[f][0]
            assert abs(b - a) < span * 1e-3, (
                f"{f} jumps {abs(b - a):.4g} across {around} in domain {domain} "
                f"({span * 1e-3:.4g} allowed) — the decoder is discontinuous"
            )

    def test_the_old_cliff_is_gone(self):
        """Regression: the exact values from the original bug report."""
        w = {x: decode_action([x] * 7, domain="pm1").w_in * 1e6
             for x in (-0.30, -0.01, 0.0, 0.01, 0.30)}
        assert w[-0.01] < w[0.0] < w[0.01], (
            f"non-monotonic across zero: {w}"
        )
        assert w[-0.30] < w[-0.01], f"non-monotonic below zero: {w}"


class TestDomainIsExplicit:

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError, match="unknown domain"):
            decode_action([0.5] * 7, domain="whatever")

    def test_default_is_unit(self):
        assert vals([0.25] * 7, "unit") == [
            getattr(decode_action([0.25] * 7), f) for f in FIELDS
        ]

    def test_out_of_range_is_clipped_not_extrapolated(self):
        """A policy that overshoots must saturate at the range edge, never beyond —
        an extrapolated W or L is a device that does not exist."""
        for f, v in zip(FIELDS, vals([5.0] * 7, "unit")):
            assert v == pytest.approx(ACTION_SPACE[f][1], rel=1e-12), f
        for f, v in zip(FIELDS, vals([-5.0] * 7, "pm1")):
            assert v == pytest.approx(ACTION_SPACE[f][0], rel=1e-12), f

    def test_encode_is_the_inverse_of_decode(self):
        original = DesignVars(w_in=47.9e-6, l_in=0.246e-6, i_tail=0.624e-3,
                              rs=2448.0, cs=234e-15, r_load=3330.0)
        recovered = decode_action(encode_action(original))
        for f in FIELDS:
            assert getattr(recovered, f) == pytest.approx(getattr(original, f), rel=1e-6)


class TestEnvWiring:
    """The two envs use Box(-1,1) for different reasons; both must decode correctly."""

    def test_equalizer_env_declares_pm1(self):
        import inspect
        from eqrl.envs import equalizer_env
        src = inspect.getsource(equalizer_env.EqualizerEnv.step)
        assert 'domain="pm1"' in src, (
            "EqualizerEnv passes an ABSOLUTE Box(-1,1) action straight to the decoder, "
            "so it must state the pm1 domain"
        )

    def test_sequential_env_keeps_state_in_unit(self):
        import inspect
        from eqrl.envs import sequential_env
        src = inspect.getsource(sequential_env.SequentialEqualizerEnv)
        assert "np.clip(self._x" in src, (
            "SequentialEqualizerEnv's internal state must stay clipped to [0,1] for the "
            "default 'unit' decode to be correct"
        )
