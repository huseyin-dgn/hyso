"""Pure JSON/byte numerical evidence validation; no Torch/runtime import."""
import base64
import hashlib
import math
import struct

def verify_comparisons(data):
    refs=data['reference_outputs'];actual=data['actual_outputs'];measures=data['stages']['equivalence']['comparisons']
    if not refs or len(refs)!=len(actual) or len(refs)!=len(measures):raise ValueError('output evidence arity mismatch')
    types={'<f4':'f','<f8':'d','<f2':'e','<i8':'q','<i4':'i','|b1':'?','|u1':'B'}
    for a,b,measure in zip(refs,actual,measures):
        if a['shape']!=b['shape'] or a['dtype']!=b['dtype'] or a['dtype'] not in types:raise ValueError('output shape/dtype evidence mismatch')
        buffers=[]
        for row in (a,b):
            raw=base64.b64decode(row['data'],validate=True)
            if len(raw)>32*1024**2 or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('output bytes integrity')
            item='<'+types[row['dtype']]
            if len(raw)!=math.prod(row['shape'])*struct.calcsize(item):raise ValueError('output bytes/shape mismatch')
            buffers.append(struct.iter_unpack(item,raw))
        atol=data['config']['atol'];rtol=data['config']['rtol']
        if measure.get('atol')!=atol or measure.get('rtol')!=rtol or measure.get('shape')!=a['shape']:raise ValueError('comparison scope/tolerance mismatch')
        if not all(type(t) in (int,float) and math.isfinite(t) and t>=0 for t in (atol,rtol)):raise ValueError('invalid equivalence tolerance')
        absolute=relative=0.
        for (x,),(y,) in zip(*buffers):
            if not math.isfinite(x) or not math.isfinite(y):raise ValueError('nonfinite actual output')
            error=abs(x-y);absolute=max(absolute,error);relative=max(relative,error/max(abs(x),1e-30))
            if error>atol+rtol*abs(x):raise ValueError('actual output exceeds configured tolerance')
        if not math.isclose(absolute,measure['max_absolute_error'],rel_tol=1e-10,abs_tol=1e-15) or not math.isclose(relative,measure['max_relative_error'],rel_tol=1e-10,abs_tol=1e-15):raise ValueError('forged error summary')
