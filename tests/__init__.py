"""Test suite for DCC Studio.

The suite is split by what it protects:

* ``test_core``            -- synthetic unit tests of every numerical helper;
* ``test_io``              -- sample discovery, atlas parsing, result files;
* ``test_pipeline``        -- full workflow, cache, cancellation, optional analyses;
* ``test_theme``           -- appearance switching and parameter help markers;
* ``test_reference_parity``-- element-by-element comparison with the committed
  reference fixtures, including a real-data end-to-end run.

Run everything with ``python run_tests.py``, which also produces statement,
branch and function coverage reports. All tests are
self-contained: no external data folder, display server or network access is
required; GUI tests use an offscreen Qt platform.
"""
