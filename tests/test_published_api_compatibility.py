"""Frozen export contract from the verified PyPI 0.1.6 wheel."""

import hyso.models as models
from hyso.models import EBO, ELO, CEMO, EMA, CNNSENormalWithFit
from hyso.models.bcam import EBO as CanonicalEBO, ELO as CanonicalELO
from hyso.models.WithSE import CEMO as CanonicalCEMO, EMA as CanonicalEMA
from hyso.models.fit import CNNSENormalWithFit as CanonicalCNNS


def test_published_016_models_exports_and_import_identity():
    expected = ["EBO", "ELO", "CEMO", "EMA", "CNNSENormalWithFit"]
    direct = [EBO, ELO, CEMO, EMA, CNNSENormalWithFit]
    canonical = [CanonicalEBO, CanonicalELO, CanonicalCEMO, CanonicalEMA, CanonicalCNNS]
    assert isinstance(models.__all__, list)
    assert models.__all__ == expected
    wildcard = {}
    exec("from hyso.models import *", wildcard)
    assert [name for name in wildcard if name != "__builtins__"] == expected
    for name, imported, defined in zip(expected, direct, canonical):
        assert getattr(models, name) is imported is defined is wildcard[name]
        assert isinstance(imported, type)
