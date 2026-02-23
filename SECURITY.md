# Security Policy

## Supported Versions

The following table lists the versions of MangoMAS Demo currently receiving security updates:

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | Yes                |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in MangoMAS Demo, please report it responsibly.
**Do not open a public GitHub issue for security vulnerabilities.**

### How to Report

Send an email to **security@example.com** with the following information:

- A description of the vulnerability and its potential impact.
- Steps to reproduce the issue, including any relevant configuration or input data.
- The affected version(s) of MangoMAS Demo.
- Any suggested mitigations or fixes, if applicable.

### Response Timeline

- **Acknowledgment**: Within 48 hours of receiving your report.
- **Initial assessment**: Within 5 business days.
- **Resolution or mitigation**: Depending on severity, we aim to release a fix within
  14 business days for critical issues and 30 business days for lower-severity issues.

We will keep you informed of progress throughout the process. If you do not receive an
acknowledgment within 48 hours, please follow up to confirm receipt.

### Disclosure

We follow a coordinated disclosure process. We ask that you refrain from publicly disclosing
the vulnerability until we have released a fix and notified affected users. We will credit
reporters in the release notes unless anonymity is requested.

## What Qualifies as a Security Issue

The following are considered security issues and should be reported through the process above:

- Remote code execution via crafted input to the Gradio interface or any API endpoint.
- Unauthorized access to internal model weights, configuration, or system resources.
- Denial of service vulnerabilities in the MCTS engine, MoE router, or agent orchestration.
- Injection attacks (e.g., prompt injection that bypasses the Ethics cell safeguards).
- Data exfiltration or unintended information leakage through model outputs.
- Dependency vulnerabilities that directly affect MangoMAS Demo functionality.

The following are generally **not** security issues (please open a regular GitHub issue instead):

- Performance regressions or slow response times.
- UI rendering bugs in the Gradio interface.
- Feature requests related to security hardening.

## PII Handling

MangoMAS Demo includes an Ethics cognitive cell that detects personally identifiable
information (PII) in user input as part of its safety evaluation pipeline. This detection
is performed entirely at inference time for the purpose of flagging potentially sensitive
content.

**MangoMAS Demo does not store, log, or transmit user data.** All processing occurs in
memory during a single session and is discarded when the session ends. No user input or
model output is persisted to disk, sent to external services, or retained between sessions.

Operators who deploy MangoMAS Demo in their own environments are responsible for ensuring
that their deployment configuration complies with applicable data protection regulations.

## Dependency Security

MangoMAS Demo depends on the following core libraries:

| Dependency | Purpose                                      |
|------------|----------------------------------------------|
| PyTorch    | Neural network inference (MoE, MCTS, cells)  |
| Gradio     | Web-based interactive demo interface          |
| NumPy      | Numerical computation and feature extraction  |
| Plotly     | Visualization and charting                    |

We monitor these dependencies for known vulnerabilities and update them regularly.
The project pins minimum versions in `pyproject.toml` to ensure compatibility while
allowing patch-level updates that include security fixes.

If you discover a vulnerability in one of our dependencies that affects MangoMAS Demo,
please report it using the process described above.

---

This security policy is reviewed and updated with each major release.
