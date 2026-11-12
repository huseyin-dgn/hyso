"""Identify the installed source profile without environment or JSON switches.

Commercial distributions omit the development marker and the excluded modules.
Adding trusted Python files to an installation changes that installation's code.
"""
from importlib.util import find_spec

DEVELOPMENT = find_spec(__package__ + '._development') is not None
PROFILE = 'development' if DEVELOPMENT else 'commercial'
