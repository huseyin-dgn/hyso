"""Small independent known-live-set references, never allocator-counter inputs."""
import json
import mmap
from pathlib import Path
import torch


class ShortPeak(torch.nn.Module):
    def __init__(self,reference_path):
        super().__init__();self.reference_path=reference_path

    def forward(self,x):
        first=torch.empty(8192);second=torch.empty(12288)
        expected=first.untyped_storage().nbytes()+second.untyped_storage().nbytes()
        del first,second
        Path(self.reference_path).write_text(json.dumps(dict(expected_increment=expected,retained_new_bytes=0)))
        return x.view_as(x)


class NativePeak(ShortPeak):
    def forward(self,x):
        count=2*1024**2
        native=mmap.mmap(-1,count);native.write(b'x'*count);native.close()
        Path(self.reference_path).write_text(json.dumps(dict(expected_increment=0,native_bytes=count)))
        return x.view_as(x)


class Reuse(ShortPeak):
    def forward(self,x):
        addresses=[]
        for _ in range(100):
            value=torch.empty(1024);addresses.append(value.untyped_storage().data_ptr());del value
        Path(self.reference_path).write_text(json.dumps(dict(addresses=addresses,expected_increment=4096)))
        return x.view_as(x)


class Producer(torch.nn.Module):
    def forward(self,x):
        self.saved=x.clone()
        return self.saved.view_as(x)


class Holder(torch.nn.Module):
    def forward(self,x):
        self.borrowed=x
        return x.view_as(x)


class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__();self.producer=Producer();self.holder=Holder();self.alias=self.producer

    def forward(self,x):
        first=self.producer(x);value=self.alias(x);result=self.holder(value);self.saved=value;del first
        return result


class ForeignStorage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import numpy as np
        self.weight=torch.nn.Parameter(torch.from_numpy(np.ones((4,16),dtype=np.float32)))

    def forward(self,x):return x.view_as(x)


class InteriorStorage(Holder):
    def forward(self,x):
        self.borrowed=torch.from_numpy(x.numpy()[...,1:])
        return x.view_as(x)


class SeparateStorageAliases(Shared):
    def __init__(self):
        super().__init__();self.holder=InteriorStorage()
