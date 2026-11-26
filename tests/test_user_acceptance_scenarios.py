"""Only selected by -UserAcceptance; never executed during preparation."""
import os
from pathlib import Path
import pytest


def journey(name):
    # Fail closed under an unrelated general pytest invocation.
    root=os.environ.get('HYSO_USER_ATTEMPT')
    if not root: pytest.skip('explicit owned UserAcceptance payload required')
    from tools.user_acceptance.scenarios import Journey,execute,MissingEvidence
    import torch
    torch.set_num_threads(1)
    try: execute(Journey(Path(root),'cpu'),name)
    except MissingEvidence as exc: pytest.skip(str(exc))


def test_U01(): journey('U01')
def test_U02(): journey('U02')
def test_U03(): journey('U03')
def test_U04(): journey('U04')
def test_U05(): journey('U05')
def test_U06(): journey('U06')
def test_U07(): journey('U07')
def test_U08(): journey('U08')
def test_U09(): journey('U09')
def test_U10(): journey('U10')
def test_U11(): journey('U11')
def test_U12(): journey('U12')
def test_U13(): journey('U13')
def test_U14(): journey('U14')
