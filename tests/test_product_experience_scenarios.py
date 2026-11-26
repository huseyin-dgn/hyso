"""U15-U18 acceptance only through the owned CPU controller; not default pytest."""
import os
import pytest
from tools.product_acceptance.scenarios import execute

pytestmark=pytest.mark.skipif(not os.environ.get('HYSO_USER_ATTEMPT'),reason='Requires owned product acceptance controller')

def test_U15():execute(os.environ['HYSO_USER_ATTEMPT'],'U15')
def test_U16():execute(os.environ['HYSO_USER_ATTEMPT'],'U16')
def test_U17():execute(os.environ['HYSO_USER_ATTEMPT'],'U17')
def test_U18():execute(os.environ['HYSO_USER_ATTEMPT'],'U18')
