"""Executed only by the export controller; never part of ordinary regressions."""
import os
import pytest
from tools.export_acceptance.scenarios import execute
pytestmark=pytest.mark.skipif(not os.environ.get('HYSO_EXPORT_ATTEMPT'),reason='Requires owned ExportAcceptance admission')
def test_E01():execute('E01')
def test_E02_CPU():execute('E02_CPU')
def test_E05_CPU():execute('E05_CPU')
@pytest.mark.skip(reason='Native C++ deferred to the next release by user instruction')
def test_E05_CPP():execute('E05_CPP')
def test_E06_EP():execute('E06_EP')
@pytest.mark.skip(reason='AOTInductor deferred to the next release by user instruction')
def test_E06_AOT():execute('E06_AOT')
