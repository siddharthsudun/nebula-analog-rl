/* Explicit external-noise inputs. This does not select the search policy. */
function initNoiseInputs() {
  const panel = document.querySelector('#noise-inputs');
  if (!panel) return;
  const update = () => {
    const enabled = document.querySelector('#noise-enabled').checked;
    document.querySelector('#noise-settings').disabled = !enabled;
    document.querySelector('#noise-settings').hidden = !enabled;
    document.querySelector('#noise-state').textContent = enabled ? 'SNR on' : 'SNR off';
    const mode = document.querySelector('#noise-mode').value;
    const form = document.querySelector('#noise-form').value;
    panel.querySelectorAll('[data-noise-modes]').forEach(el => {
      el.hidden = !el.dataset.noiseModes.split(' ').includes(mode);
    });
    document.querySelector('#noise-range-fields').hidden = mode === 'measured' || (mode === 'estimated' && form === 'budget');
    document.querySelector('#noise-value-field').hidden = mode === 'unknown' || (mode === 'estimated' && form === 'range');
    document.querySelector('#noise-value-label').textContent = mode === 'measured'
      ? (document.querySelector('#noise-unit').value === 'snr' ? 'Measured input SNR (dB)' : 'Measured noise (mV RMS)') : 'Maximum noise budget (mV RMS)';
    document.querySelector('#noise-signal-label').textContent = document.querySelector('#noise-signal-reference').value === 'tx_vpp'
      ? 'Differential TX amplitude (Vpp)' : 'Differential CTLE-input signal (V RMS, in stated band)';
    document.querySelector('#noise-assumption-note').textContent = mode === 'unknown'
      ? 'Conditional results. Noise bounds are assumptions. The 1 V TX swing and 10 MHz–5 GHz band are assumptions until edited.'
      : 'Enter your signal and measurement band explicitly. Noise is differential RMS after the channel, before the CTLE.';
  };
  panel.addEventListener('change', update);
  document.querySelector('#noise-enabled').addEventListener('change', e => {
    e.target.dataset.manual = e.target.checked ? 'on' : 'off';
  });
  document.querySelector('#noise-unit').addEventListener('change', () => {
    document.querySelector('#noise-value').value = '';
  });
  window.refreshNoiseInputs = update;
  panel.querySelectorAll('[data-provenance]').forEach(el => el.addEventListener('input', () => { el.dataset.edited = 'true'; }));
  document.querySelector('#noise-signal-reference').addEventListener('change', () => {
    document.querySelector('#noise-signal').value = '';
    document.querySelector('#noise-signal').dataset.edited = 'true';
  });
  document.querySelector('#noise-mode').addEventListener('change', () => {
    const unknown = document.querySelector('#noise-mode').value === 'unknown';
    for (const [id, value] of [['noise-signal', '1'], ['noise-band-low', '10'], ['noise-band-high', '5000'], ['noise-low', '1'], ['noise-high', '50']]) {
      const el = document.querySelector('#' + id);
      if (el.dataset.edited !== 'true') el.value = unknown ? value : '';
    }
  });
  update();
}

function readNoiseRequest() {
  if (!document.querySelector('#noise-enabled')?.checked) return undefined;
  const mode = document.querySelector('#noise-mode').value;
  const value = id => {
    const el = document.querySelector('#' + id);
    if (!el.value.trim() || !Number.isFinite(Number(el.value))) throw new Error('Complete the noise, signal, and bandwidth fields.');
    return Number(el.value);
  };
  const request = {schema: 'eqrl.snr.request.v2', mode,
    signal_reference: document.querySelector('#noise-signal-reference').value,
    signal_value_v: value('noise-signal'),
    bandwidth_hz: [value('noise-band-low')*1e6, value('noise-band-high')*1e6]};
  if (mode === 'measured' && document.querySelector('#noise-unit').value === 'snr') request.input_snr_db = value('noise-value');
  else if (mode === 'measured') request.value_vrms = value('noise-value')/1000;
  else if (mode === 'estimated' && document.querySelector('#noise-form').value === 'budget') request.budget_vrms = value('noise-value')/1000;
  else { request.low_vrms = value('noise-low')/1000; request.high_vrms = value('noise-high')/1000; }
  if (mode === 'unknown') {
    request.assumed_fields = ['noise'];
    if (document.querySelector('#noise-signal').dataset.edited !== 'true') request.assumed_fields.push('signal');
    if (['noise-band-low', 'noise-band-high'].some(id => document.querySelector('#'+id).dataset.edited !== 'true')) request.assumed_fields.push('bandwidth');
  }
  const noises = [request.value_vrms, request.budget_vrms, request.low_vrms, request.high_vrms].filter(v => v !== undefined);
  if (noises.some(v => v < 0) || request.signal_value_v <= 0 || request.low_vrms > request.high_vrms
      || request.bandwidth_hz[0] < 1e7 || request.bandwidth_hz[1] > 5e9 || request.bandwidth_hz[0] >= request.bandwidth_hz[1]) {
    throw new Error('Use nonnegative ordered noise, positive signal, and ordered band edges within 10–5000 MHz.');
  }
  return request;
}

function applyParsedNoise(noise) {
  const toggle = document.querySelector('#noise-enabled');
  if (!toggle || !noise) return;
  const signature = JSON.stringify(noise);
  if (toggle.dataset.lastParsed === signature) return;
  toggle.dataset.lastParsed = signature;
  if (noise.source === 'explicit_opt_out') {
    toggle.checked = false;
    toggle.dataset.manual = 'off';
  } else if (toggle.dataset.manual !== 'off') {
    if (!noise.enabled) {
      if (toggle.dataset.manual !== 'on') toggle.checked = false;
    } else {
      toggle.checked = true;
      document.querySelector('#snr-advanced').open = true;
      const r = noise.request || {};
      document.querySelector('#noise-mode').value = r.mode || 'measured';
      document.querySelector('#noise-unit').value = r.input_snr_db !== undefined ? 'snr' : 'rms';
      document.querySelector('#noise-form').value = r.budget_vrms !== undefined ? 'budget' : 'range';
      document.querySelector('#noise-signal-reference').value = r.signal_reference || 'tx_vpp';
      const unknown = r.mode === 'unknown';
      const entries = {
        'noise-value': r.input_snr_db ?? ((r.value_vrms ?? r.budget_vrms) !== undefined ? (r.value_vrms ?? r.budget_vrms)*1000 : ''),
        'noise-low': r.low_vrms !== undefined ? r.low_vrms*1000 : (unknown ? 1 : ''),
        'noise-high': r.high_vrms !== undefined ? r.high_vrms*1000 : (unknown ? 50 : ''),
        'noise-signal': r.signal_value_v ?? (unknown ? 1 : ''),
        'noise-band-low': r.bandwidth_hz ? r.bandwidth_hz[0]/1e6 : (unknown ? 10 : ''),
        'noise-band-high': r.bandwidth_hz ? r.bandwidth_hz[1]/1e6 : (unknown ? 5000 : ''),
      };
      for (const [id, value] of Object.entries(entries)) {
        const el = document.querySelector('#'+id);
        el.value = String(value);
        el.dataset.edited = (id === 'noise-signal' ? r.signal_value_v !== undefined :
          id.startsWith('noise-band-') ? r.bandwidth_hz !== undefined : false) ? 'true' : 'false';
      }
    }
  }
  window.refreshNoiseInputs?.();
}

function noiseResultHtml(r) {
  const n = r.noise_evaluation;
  if (!n) return '';
  const show = v => v === null || v === undefined ? 'n/a' : Number(v).toFixed(2);
  const rows = (n.points || []).map(p => `<tr><td>${show(p.input_noise_vrms*1000)}</td><td>${p.input_snr_zero_noise ? 'zero-noise control' : show(p.input_snr_db)}</td><td>${show(p.output_snr_db)}</td><td>${show(p.eye_v_mv)}</td><td>${show(p.eye_h_ui)}</td><td>${p.errors}/${p.count}</td><td>${p.passed ? 'Pass' : 'Fail'}</td></tr>`).join('');
  return `<div class="panel noise-result"><div class="panel-head"><span class="panel-title">${n.conditional ? 'Conditional noise assessment' : 'Sampled noise assessment'}</span></div>
    <p>${escapeHtml(n.search_note || '')}</p><p>${escapeHtml(n.reason || n.scope || n.status)}</p>
    <p>Provenance: ${escapeHtml(Object.entries(n.provenance || {}).map(([k,v]) => `${k}: ${v}`).join(', '))}.</p>
    ${n.within_training_coverage === false ? '<p>Outside pilot training coverage. This is a modeled assessment, not a validated policy prediction.</p>' : ''}
    ${rows ? `<div class="noise-table-scroll"><table><thead><tr><th>Noise mV RMS</th><th>Input SNR dB</th><th>Output SNR dB</th><th>Eye mV</th><th>Eye UI</th><th>Errors/bits</th><th>Sample</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p>Modeled signal RMS in the stated band: CTLE input ${show(n.modeled_input_signal_vrms*1000)} mV; output before DFE ${show(n.modeled_output_signal_vrms*1000)} mV. Equivalent TX swing: ${show(n.equivalent_tx_vpp)} Vpp. Worst sample: ${n.worst_point_index+1}.</p>` : ''}
    <p>Flat external noise spectrum; integrated noise approximated as white slicer noise. Linear signal model, nominal TT, no low-BER or PVT sign-off.</p></div>`;
}

initNoiseInputs();
