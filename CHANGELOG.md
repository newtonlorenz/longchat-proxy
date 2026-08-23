# Changelog

All notable changes will be documented here.

## 0.1.1 - 2026-08-23

- Accept an optional early-compaction threshold for latency and prompt-cache tuning.
- Add safe, configuration-enforced request body overrides for provider-specific controls.
- Add optional tool removal and volatile system-boilerplate sanitisation.
- Preserve client dictionary values when applying configured request overrides.

## 0.1.0 - 2026-08-23

- Initial public release.
- Transparent OpenAI-compatible streaming proxy.
- Rolling helper-model summarisation with exact-prefix caching.
- Safe context and output budgeting.
- In-memory, SQLite and stateless modes.
- Authentication, remote-listener guard, request limits and optional rate limiting.
- System/developer instruction merging and tool/image-aware accounting.
- Health, readiness, status and doctor commands.
- Docker, launchd and systemd examples.
