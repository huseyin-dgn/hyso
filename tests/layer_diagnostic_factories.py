"""Small owned inference recipes for allocation/perturbation tests."""
import torch


class Temporary(torch.nn.Module):
    def forward(self,x):
        temporary=x.clone()
        result=temporary+1
        del temporary
        return result


class Memory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf=Temporary()

    def forward(self,x):
        return self.leaf(x)


class Retaining(torch.nn.Module):
    """A new retained block replaces an old block: retained is not net change."""
    def __init__(self):
        super().__init__()
        self.cache=None

    def forward(self,x):
        self.cache=x.clone()
        temporary=x.clone()
        result=self.cache.view_as(x)
        del temporary
        return result


class NestedRetaining(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf=Retaining()
        self.alias=self.leaf

    def forward(self,x):
        first=self.leaf(x)
        second=self.alias(x)
        del first
        return second


class ReferencePeak(torch.nn.Module):
    """TEST ONLY: independent CUDA allocator counter, resets this owned worker."""
    def __init__(self,reference_path):
        super().__init__();self.reference_path=reference_path

    def forward(self,x):
        import json
        from pathlib import Path
        baseline=torch.cuda.memory_allocated()
        # Test-only reset; production recorder never calls this API.
        torch.cuda.reset_peak_memory_stats()
        kept=x.clone();alias=kept.view_as(x);temporary=torch.empty(x.numel()*4,device=x.device)
        del temporary
        peak=torch.cuda.max_memory_allocated();after=torch.cuda.memory_allocated()
        Path(self.reference_path).write_text(json.dumps(dict(baseline=baseline,peak=peak,after=after,
            storage_bytes=alias.untyped_storage().nbytes(),same_storage=alias.untyped_storage()._cdata==kept.untyped_storage()._cdata)))
        return alias
