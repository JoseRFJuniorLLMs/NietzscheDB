"""Configuration for NietzscheDB Graph Curator Agent."""

import os

# NietzscheDB connection
NIETZSCHE_HOST = os.getenv("NIETZSCHE_HOST", "136.111.0.47:443")
NIETZSCHE_INSECURE = os.getenv("NIETZSCHE_INSECURE", "false").lower() == "true"
NIETZSCHE_CERT_PATH = os.getenv(
    "NIETZSCHE_CERT_PATH",
    os.path.expanduser("~/AppData/Local/Temp/eva-cert.pem"),
)

# Claude API
CLAUDE_MODEL = os.getenv("CURATOR_MODEL", "claude-sonnet-4-6")
MAX_TURNS = int(os.getenv("CURATOR_MAX_TURNS", "30"))

# Agent behavior
DEFAULT_COLLECTIONS = os.getenv(
    "CURATOR_COLLECTIONS",
    "eva_core,tech_galaxies,knowledge_galaxies,culture_galaxies,science_galaxies",
).split(",")
