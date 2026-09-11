# PVT runtime and measured success rate

The dashboard now leads with **Measured success rate: 96.8%**. This is Auto solving 30 of the 31 matched historical specifications in `dashboard/static/data/mode_bench5.json`, spec seed 137, with a +/-1.5 dB target tolerance and the recorded nominal circuit checks. It is a system solve rate, not standalone PPO accuracy, PVT certification, optional SNR certification, or a guarantee for unseen specifications. The headline is calculated from the evidence data, not hardcoded. The newer Fastest row remains 31/31 on that matched subset; its full replay and the separate seed-761 replay each solved 32/32 nominal cases.

## Production changes

- Server startup prewarms the independent search and verification PVT worker banks under the existing simulator lock. Health stays warming until initialization finishes; lifespan shutdown closes the pool.
- Fastest certifies the three stress corners of `corner_grid(spec, "reduced")` -- tt/nominal/27 C, ss/1.71 V/125 C, ff/1.89 V/0 C -- inside its own 5 s request budget, with 1.0 s of that budget reserved for the sweep. Every other mode certifies the full 45-corner grid. The Check PVT button always runs the full grid on the exact sizing it is handed, whatever mode produced it. Exhaustion is explicitly unverified and does not become a nominal pass.
  - **Correction, same day.** This bullet previously read "Fastest now requests full-grid PVT with a ten-second repair budget. Default/Retarget retain 300 seconds; other modes retain 900." That described a request path the resident runtime had already replaced: `realtime.design_realtime` returned from Fastest with `pvt.status = 'not_requested'` before `certify` was ever called, and no ten-second PVT budget existed anywhere in it. Fastest therefore ran **no** corner sweep at all, not a full one. The sentence above describes what the code does as of this commit. Measured warm on the author's machine, delivered sizing plus anchor across both banks: 3 corners 0.50 s, 45 corners 3.12 s.
- Worker startup, pool-lock waits and batch-lock waits share the request deadline. Expired requests cannot dispatch new work. Windows Job Objects own launcher descendants and kill the entire worker tree on timeout, shutdown or owner exit. Search and verification remain separate processes, with no cached measurement results.
- Fastest cost accounting charges one actual measurement per corpus seed. Legacy parallel worker counts are aggregated into enclosing simulation counters.
- The evidence page displays the previously hidden mixed-run disclosure: Fastest timings include a shared simulator improvement absent from the older other-mode timings. They are not a controlled runtime comparison.

## Verification

`work/speed-accuracy-20260910/finish_verification_v2/timings.json` records the historical spec-0 repair through `run_repair()` after real `server._warm_up()`:

| Check | Result |
|---|---|
| Cold request, one-second budget | Unverified at 1.031 seconds including cleanup |
| Server warm-up | 52.313 seconds, ready only afterward |
| Warm full repair, three repeats | 8.656 / 7.109 / 7.031 seconds |
| Work per accepted repair | 135 measurements, 578 analyses |
| Independent acceptance | Exact 45-corner metric/guard/check parity with legacy |
| Legacy recount | 135 measurements, 578 analyses; accepted |

These timings establish the measured case, not a universal ten-second solve guarantee for specifications that require additional search. All ordinary physical checks and the exact 45-corner acceptance grid remain active. Cold model loading is moved to service readiness, not eliminated.

Validation: 63 Python tests passed, including solver regression tests, timeout/lock behavior, Fastest endpoint PVT wiring, nested subprocess counts and a real Windows launcher-descendant cleanup test. Three JavaScript tests passed, including the rendered evidence headline and PVT export verdicts. Browser visual inspection was unavailable in this session. The existing live server reported the new PVT-ready health message; an additional live design request returned busy because another run was active, so it was not interrupted.
