# ALMA hackathon laptop setup

Use [hackathon-setup/README.md](hackathon-setup/README.md) for the setup-only
handoff prepared for October 22, 2026.

The target is Windows x64 with standard Python 3.13. The folder contains a
dependency lock, a cached embedding model, setup checks and optional ALMA endpoint
configuration tools. It does not include the AuditGuard application for transfer.

**Offline package wheels are not yet available.** A fresh laptop currently needs
internet to install the pinned dependencies. Recreating the environment on the
laptop and testing the actual ALMA endpoint are still required.

The event description says to build application code on the day. Transfer the
setup folder only. Do not copy the repository, credentials, private documents,
existing indexes, built frontend, or this PC's virtual environment.

ALMA provides the inference infrastructure. Confirm its URL, model ID and API
protocol with the organizers. Use the ALMA-only route; a Google API key is not a
prerequisite for this handoff.
