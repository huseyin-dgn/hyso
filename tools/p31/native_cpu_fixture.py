"""Finite CPU-only sleep fixture for P31 launcher integration, no inference."""
import time


def make_cpu():
    import torch
    if torch.cuda.is_initialized():
        raise AssertionError('unexpected CUDA initialization')
    def forbidden(name):
        def fail(*args, **kwargs):
            raise AssertionError('native smoke attempted CUDA: ' + name)
        return fail
    for name in ('_lazy_init', 'is_available', 'device_count', 'current_device',
                 'get_device_properties', 'memory_allocated', 'memory_reserved',
                 'synchronize', 'set_device'):
        setattr(torch.cuda, name, forbidden(name))
    class Work:
        identity = dict(kind='native_smoke_sleep', unit='operation', device_id='cpu')
        def step(self):
            time.sleep(.01)
            if torch.cuda.is_initialized():
                raise AssertionError('CUDA initialized during CPU smoke')
            return 1
        def close(self):
            if torch.cuda.is_initialized():
                raise AssertionError('CUDA initialized during CPU smoke')
    return Work()
