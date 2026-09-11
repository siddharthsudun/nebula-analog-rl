# silQ intro v3: the chip carries the name

This revision responds to the request for silQ to appear on the central square chip surface, powered by traces converging from all directions. The generated final-frame reference has legible lowercase s/i/l and uppercase Q engraved in cyan on the central graphite package. It is a conceptual brand chip, not fabricated-project evidence.

Generation artifacts live in work/intro-converge-v3. The final-frame prompt and video prompt are saved with their job responses. The video uses Seedance 2.0 with a verified GPT Image 2 end frame to anchor the final lettering, composition and PCB style. The previous cinematic intro remains recoverable from backups/intro-cinematic-v2, with the first graphite version and original yellow UI also preserved in their own backups.

Normal playback has no separate large HTML wordmark. The exact Astera Labs image and hyperlink remain a small bottom-center credit. A static fallback wordmark appears only if video playback fails or stalls. The complete chip stays within mobile framing through object-fit: contain. The session key and cache versions change so the new intro is eligible to play after refresh.

The integration harness verifies media references, on-chip presentation, mobile framing and the fallback state before changing the active page. Playback tests cover delayed credit visibility, replay reset, keyboard containment, focus restoration and motion preferences. Actual video frame inspection and encoding metrics are stored alongside the render.

Completed render: video job `0125418c-d6d7-48d4-962b-70f91b92adb8`, end-frame job `5fe425b7-8b4b-43c0-a1e2-80bda84347ee`. The final web film is 5.0417 seconds, 5,867,750 bytes, encoded at 1080p with H.264 CRF 19 and faststart; the 19,607,978-byte source is retained. Six inspected frames show the trace chase, dark silQ engraving at 2.4 seconds and illuminated on-chip lettering from 3.2 seconds through the end. The small Astera credit appears at 3.75 seconds. There is no large floating title during successful playback.

The integration and intro behavior harnesses, JavaScript syntax check and nine SNR/PVT regression checks passed. Preview was restarted on port 8013 after the prior server stopped. Browser compositing has not been visually verified in this session; generated film frames were inspected directly.
