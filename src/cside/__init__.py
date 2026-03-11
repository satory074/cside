"""CSIDE — Cover Song IDEntification (2026 remake)."""

import os

# Both torch and faiss-cpu ship their own libomp.dylib on macOS, which causes
# a fatal duplicate-runtime error when both are loaded in the same process.
# This must be set before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "2.0.0"
