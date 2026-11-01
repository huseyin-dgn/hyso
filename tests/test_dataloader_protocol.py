"""Deterministic P21 clock arithmetic, quantiles, lifecycle and bounded storage."""
from types import SimpleNamespace
import weakref
import gc
import pytest
import torch
from hyso.performance import DataLoaderAnalysisProtocol, TrainingTimingProtocol, TrainingTimingError
from hyso.performance.dataloader import _LoaderRecorder, _distribution


@pytest.mark.parametrize('value',[True,0,-1,1000001,1.5])
def test_protocol_rejects_invalid_cap(value):
    with pytest.raises(ValueError): DataLoaderAnalysisProtocol(max_records=value)


def test_quantile_policy_known_array():
    d=_distribution(list(range(1,1001)))
    assert d['mean_ms']==500.5 and d['p50_ms']==500.5 and d['p95_ms']==950.05
    assert d['p99']==dict(value=990.01,unit='ms',status='derived',reason=None)
    assert _distribution(list(range(999)))['p99']['status']=='insufficient_samples'
    assert _distribution([])['mean_ms'] is None


def test_exact_iter_next_calls_and_fake_clock(monkeypatch):
    import hyso.performance.dataloader as module
    ticks=iter(range(1000,100000,100))
    monkeypatch.setattr(module.time,'perf_counter_ns',lambda:next(ticks))
    calls=[]
    class Iterator:
        def __init__(self): self.index=0
        def __next__(self):
            calls.append('next'); self.index+=1
            if self.index==3: raise StopIteration
            return self.index
    class Loader:
        def __iter__(self): calls.append('iter'); return Iterator()
    r=_LoaderRecorder(SimpleNamespace(device=torch.device('cpu')),TrainingTimingProtocol(),DataLoaderAnalysisProtocol())
    r.epoch=dict(epoch_id=1)
    observer=r.observe_loader(Loader(),'training')
    assert list(observer)==[1,2]
    assert calls==['iter','next','next','next']
    assert [x['kind'] for x in r.loader_rows]==['iterator_init','batch','batch','exhaustion']
    assert all(x['host_ms']==.0001 for x in r.loader_rows)
    assert not r.live and observer.iterator is None and observer.loader is None
    assert list(observer)==[] and calls==['iter','next','next','next']
    r.close()


def test_cap_failure_releases_underlying_iterator():
    refs=[]
    class Iterator:
        def __next__(self): return 1
    class Loader:
        def __iter__(self):
            it=Iterator(); refs.append(weakref.ref(it)); return it
    r=_LoaderRecorder(SimpleNamespace(device=torch.device('cpu')),TrainingTimingProtocol(),DataLoaderAnalysisProtocol(max_records=1))
    r.epoch=dict(epoch_id=1)
    observer=iter(r.observe_loader(Loader(),'training'))
    with pytest.raises(TrainingTimingError): next(observer)
    r.close(); gc.collect()
    assert refs[0]() is None and not r.live
