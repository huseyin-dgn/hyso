"""Importable trusted factories for failure-only fresh-worker protocol tests."""
import os


def explode():
    raise LookupError('controlled factory startup exception')


def crash():
    os._exit(9)
