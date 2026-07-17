# Security policy

## Reporting

Please report vulnerabilities through GitHub's private security-advisory
facility for this repository. Do not open a public issue containing credentials,
patient data, exploitable details or private infrastructure information.

## Scope

The maintained package and container on the latest release line receive
security fixes. Files under `legacy/` and `coursework/` are retained for
academic provenance and are not supported runtime components.

Never commit clinical data, DVC credentials, MLflow credentials or model-store
tokens. Use provider credential chains and local ignored configuration. The API
must be placed behind authentication, TLS and request-size controls before use
on any non-public data.

This project is a research prototype, not a medical device. A security review
does not make it suitable for diagnosis or clinical decision-making.
