# silQ cinematic intro v2

The requested revision replaces the restrained chip flyover with a five-second PCB signal chase and restores a dramatic delayed silQ reveal. The user's supplied signal-conditioner board image is used as a hardware reference for component density, traces, vias, solder joints and chip packages, reinterpreted in graphite and electric cyan. This is conceptual brand footage, not a representation of the project's fabricated hardware.

## Direction

The shot begins at trace level, follows a cyan differential signal through a populated board, then rises toward the main IC. The website brings in the exact silQ wordmark at 3.15 seconds of video playback, followed by the tagline and the original linked Astera Labs logo. Brand lettering is rendered in HTML so it stays exact. The generated board retains markings derived from the supplied reference; it is illustrative hardware. The overlay uses a short focus-and-position reveal, restrained letterboxing and a delayed left-side shade, allowing the opening chase to remain visible.

The film remains muted. The overlay timeline follows video time. The intro skips on reduced motion, can be skipped with Escape or the button, contains keyboard focus, restores focus after closing, and can be replayed from the footer. An unavailable or blocked video exposes the static title and retains a bounded exit timer. Playback beginning after a slow load resets any fallback title before the chase.

## Provenance and recovery

- Higgsfield Seedance 2.0, 1080p, high bitrate, 16:9, five seconds.
- Generation job: `12595104-93ae-4e5e-8be9-d461de35b74f`.
- Prompt, generation response, frame inspection and behavioral harness: `work/intro-cinematic-v2/`.
- Previous graphite intro, frontend integration files and SHA-256 manifest: `backups/intro-graphite-v1/`.
- Original yellow UI remains in `backups/frontend-original-2026-09-10.zip`.

The browser session key and app/style cache versions are changed so the new version is eligible to play on the next visit. Intro behavior is checked with a JavaScript harness; actual rendered browser review requires an available browser surface.

## Completed checks and media

The integrated film is `dashboard/static/media/silq-cinematic-v2.mp4`: 5.0417 seconds, 17,276,735 bytes. The first-frame poster is `silq-cinematic-v2-poster.jpg`. Six sampled frames at 0.15, 1.0, 2.0, 2.9, 3.6 and 4.6 seconds were inspected in `work/intro-cinematic-v2/contact-sheet.jpg`: the cyan trace chase progresses to IC pins and then a complete graphite board reveal. The high-bitrate original is retained; attempts to obtain a local encoder failed because the package host closed TLS connections. No backend dependencies were changed.

The intro harness passed playback-timed reveal, focus exclusion before the credit appears, keyboard containment, Escape, replay reset, reduced-motion and session bypass, blocked-autoplay fallback, stalled-load fallback and cleanup. App syntax and nine existing SNR/PVT JavaScript checks passed. Local HTTP checks returned 200 for the page, revised scripts/styles and poster. Previous-intro backup hashes were verified. Rendered browser compositing was not visually checked because this session has no browser surface.
