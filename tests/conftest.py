"""Pytest configuration and shared fixtures."""

import os

# Both torch and faiss-cpu ship their own libomp.dylib on macOS, causing a
# duplicate-runtime conflict.  KMP_DUPLICATE_LIB_OK suppresses the fatal error.
# Must be set before either library is imported (i.e., here in conftest).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
