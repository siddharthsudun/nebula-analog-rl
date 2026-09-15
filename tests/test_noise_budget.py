import numpy as np
import pytest

from silq.sim.noise_budget import flat_noise_gain, combined_output_rms, differential_output_noise
from silq.sim.ngspice_runner import NgspiceError


def test_flat_response_refers_noise_by_voltage_gain():
    f = np.array([1e6, 1e7, 1e9, 5e9, 24e9])
    assert flat_noise_gain(f, np.full(f.size, -2 + 0j)) == pytest.approx(2)
    assert combined_output_rms(0.003, 2, 0.008) == pytest.approx(0.01)


def test_noise_referral_integrates_power_instead_of_using_peak_gain():
    f = np.array([1e7, 5e9])
    assert flat_noise_gain(f, np.array([1, 3])) == pytest.approx(np.sqrt(5))


@pytest.mark.parametrize("freq", [[1e8, 5e9], [1e7, 4e9], [5e9, 1e7]])
def test_noise_referral_rejects_incomplete_or_reversed_band(freq):
    with pytest.raises(ValueError):
        flat_noise_gain(freq, [1, 1])


@pytest.mark.parametrize("values", [(-1, 1, 1), (1, float('nan'), 1), (1, 1, -1)])
def test_combining_noise_rejects_unusable_values(values):
    with pytest.raises(ValueError):
        combined_output_rms(*values)


def test_output_noise_is_direct_differential_and_has_no_fallback(tmp_path):
    class Backend:
        def __init__(self):
            self._dir = tmp_path
            self._ng = self
            self.commands = []
        def _prime(self, *args):
            pass
        def _analysis(self, command):
            self.commands.append(command)
        def exec_command(self, command):
            self.commands.append(command)
        def _read(self, name):
            return np.array([[0, 0.002]])
    backend = Backend()
    assert differential_output_noise(backend, None) == 0.002
    assert 'noise v(outp,outn) vinp dec 20 10e6 5e9' in backend.commands
    assert any('onoise_total' in c for c in backend.commands)
    backend._read = lambda name: np.array([[0, float('nan')]])
    with pytest.raises(NgspiceError):
        differential_output_noise(backend, None)
