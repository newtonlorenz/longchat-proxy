# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's security advisory feature for
this repository. Do not open a public issue containing exploit details or private data.

## Deployment guidance

- LongChat binds to `127.0.0.1` by default.
- A non-loopback listener requires a client bearer key unless the operator explicitly opts out.
- The built-in server is plain HTTP. Use a private VPN or an HTTPS reverse proxy for remote access.
- Never publish the listener directly to the internet without TLS, authentication and network
  controls.
- SQLite state contains generated summaries and may contain sensitive information. Protect it with
  OS permissions and full-disk encryption, or use the default in-memory state.
- Treat both main-model and summariser output as untrusted. Do not automatically execute generated
  tools or commands without a separate authorization layer.

Supported security fixes target the latest release on the default branch.
