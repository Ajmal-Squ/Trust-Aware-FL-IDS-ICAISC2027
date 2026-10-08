"""Compatibility loader for the shared METRA-FL experiment utilities.

The conference operating-point script reuses the public preprocessing,
model, attack, metric, and aggregation helpers maintained in the METRA-FL
repository. This loader fetches that exact public source when imported.
"""
from urllib.request import urlopen

_SOURCE = "https://raw.githubusercontent.com/Ajmal-Squ/METRA-FL/main/metra_fl_binary_rerun.py"
_code = urlopen(_SOURCE, timeout=30).read().decode("utf-8")
exec(compile(_code, _SOURCE, "exec"), globals(), globals())
