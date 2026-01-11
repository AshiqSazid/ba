"""API package for TheraMuse endpoints."""

# Avoid importing FastAPI objects at package import time so CLI tools can run in
# lightweight environments that do not ship the web stack dependencies.
