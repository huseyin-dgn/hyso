"""Source-pinned, separately authorized P18 native reference capture. No quotas.
Derived from the preserved reference worker listed in counters/origin.json.
Not invoked by preparation or smoke. The original worker is unchanged.
"""
import argparse,ctypes as C,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_acceptance.core import read,sha,verify
ROOT=Path(__file__).resolve().parent/'counters'
BLOCKS,THREADS,CHAINS,STEPS=240,256,8,4096
ELEMENTS=16_777_216

def write_json(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)
def bind(lib,name,types):
    fun=getattr(lib,name);fun.argtypes,fun.restype=types,C.c_int;return fun
def checked(fun,*args):
    code=fun(*args)
    if code:raise RuntimeError(str(fun)+' returned '+str(code))
def target_case(output):
    import torch
    n=65536;torch.cuda.set_device(0)
    a=torch.ones(n,device='cuda:0');b=torch.full_like(a,2.);out=torch.empty_like(a)
    for _ in range(2):torch.add(a,b,out=out);torch.cuda.synchronize(0)
    samples=[];runtime=torch.cuda.cudart()
    for i in range(3):
        if int(runtime.cudaProfilerStart()):raise RuntimeError('profiler start')
        try:
            torch.cuda.nvtx.range_push('hyso_p18_roof_target')
            try:torch.add(a,b,out=out);torch.cuda.synchronize(0)
            finally:torch.cuda.nvtx.range_pop()
        finally:
            if int(runtime.cudaProfilerStop()):raise RuntimeError('profiler stop')
        if not torch.equal(out.cpu(),torch.full((n,),3.)):raise ValueError('target reference mismatch')
        samples.append(dict(index=i,output_equal=True,terminal_completed=True))
    write_json(output/'capture.json',dict(pid=os.getpid(),device=0,gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),dtype='float32',nvtx_range='hyso_p18_roof_target',
        targeted_launches=3,samples=samples,output_equal=True,terminal_completed=True,script_sha256=sha(__file__),
        scope='new_controlled_vector_add_not_trained_CNN_DRAM',allocator_fraction=None))

def run_case(case, output):
    import numpy as np
    import torch

    if not output.is_dir() or (output / 'capture.json').exists():
        raise RuntimeError('A new output directory is required')
    compilation = dict(source_sha256=sha(ROOT/'reference_kernels.cu'),ptx_sha256=sha(ROOT/'reference_kernels.ptx'),worker_sha256=sha(__file__))
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.init()
    if (torch.cuda.get_device_capability(0) != (8, 6) or
        torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 3060 Laptop GPU'):
        raise RuntimeError('This fixed calibration is limited to the original SM86 GPU')
    driver = C.WinDLL('nvcuda.dll')
    init = bind(driver, 'cuInit', [C.c_uint])
    get_device = bind(driver, 'cuDeviceGet', [C.POINTER(C.c_int), C.c_int])
    attribute = bind(driver, 'cuDeviceGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_int])
    pci_bus = bind(driver, 'cuDeviceGetPCIBusId', [C.c_void_p, C.c_int, C.c_int])
    load = bind(driver, 'cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p])
    function = bind(driver, 'cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])
    unload = bind(driver, 'cuModuleUnload', [C.c_void_p])
    launch = bind(driver, 'cuLaunchKernel', [C.c_void_p] + [C.c_uint]*7 +
                  [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p])
    checked(init, 0)
    device, l2 = C.c_int(), C.c_int()
    checked(get_device, C.byref(device), 0)
    checked(attribute, C.byref(l2), 38, device.value)  # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
    bus = C.create_string_buffer(32)
    checked(pci_bus, bus, len(bus), device.value)
    if l2.value != 3_145_728 or ELEMENTS*4 <= 16*l2.value:
        raise RuntimeError('Expected L2 size or >16x-L2 per-buffer DRAM working set differs')

    if case == 'compute':
        out = torch.empty((BLOCKS*THREADS, CHAINS), device='cuda:0', dtype=torch.float32)
        base = 1.0 + ((torch.arange(BLOCKS*THREADS) & 255).float() / 256.0)
        expected = base[:, None] + torch.arange(CHAINS).float()[None, :]/16.0 + STEPS/1024.0
        pointers = [C.c_uint64(out.data_ptr())]
        tensor_bytes = out.numel()*4
        reference = dict(expected_ffma_thread_instructions=BLOCKS*THREADS*CHAINS*STEPS,
            expected_fma2_flops=2*BLOCKS*THREADS*CHAINS*STEPS,
            definition='native_FP32_SIMT_FMA=2; initialization_FADD_counted_separately_from_report')
    else:
        # Fixed integer hash -> nonuniform finite FP32 [1,2); no GPU RNG or fill kernels.
        bits = np.arange(ELEMENTS, dtype=np.uint32)
        bits ^= bits >> 16; bits *= np.uint32(0x7feb352d)
        bits ^= bits >> 15; bits *= np.uint32(0x846ca68b)
        bits ^= bits >> 16; bits &= np.uint32(0x007fffff); bits |= np.uint32(0x3f800000)
        expected = torch.from_numpy(bits.view(np.float32))
        source = expected.to('cuda:0')
        out = torch.empty_like(source)
        pointers = [C.c_uint64(source.data_ptr()), C.c_uint64(out.data_ptr())]
        tensor_bytes = source.numel()*8
        reference = dict(logical_read_bytes=ELEMENTS*4, logical_write_bytes=ELEMENTS*4,
            expected_fp32_arithmetic_instructions=0,
            definition='payload_is_only_a_coverage_check; roof_numerator_must_use_real_DRAM_counters',
            load_policy='ld.global.cg.v4.f32', store_policy='st.global.cs.v4.f32', sweeps=1)
    parameters = (C.c_void_p * len(pointers))(*(C.cast(C.byref(p), C.c_void_p) for p in pointers))
    ptx = C.create_string_buffer((ROOT / 'reference_kernels.ptx').read_bytes())
    module, kernel = C.c_void_p(), C.c_void_p()
    checked(load, C.byref(module), ptx)
    name = b'p18_compute_f32' if case == 'compute' else b'p18_dram_copy_f32'
    checked(function, C.byref(kernel), module, name)
    stream = C.c_void_p(torch.cuda.current_stream(0).cuda_stream)
    nvtx = f'hyso_p18_roof_{case}'
    runtime = torch.cuda.cudart()
    samples = []

    def execute():
        checked(launch, kernel, BLOCKS, 1, 1, THREADS, 1, 1, 0, stream, parameters, None)
        torch.cuda.synchronize(0)

    def validate():
        actual = out.cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(f'{case} result differs from exact independent CPU reference')
        return hashlib.sha256(actual.numpy().tobytes()).hexdigest()

    try:
        for _ in range(2):
            execute()
        warmup_hash = validate()
        for index in range(3):
            start = time.time_ns()
            if int(runtime.cudaProfilerStart()) != 0:
                raise RuntimeError('cudaProfilerStart failed')
            try:
                torch.cuda.nvtx.range_push(nvtx)
                try:
                    execute()
                finally:
                    torch.cuda.nvtx.range_pop()
            finally:
                if int(runtime.cudaProfilerStop()) != 0:
                    raise RuntimeError('cudaProfilerStop failed')
            digest = validate()  # Outside capture; D2H is not DRAM roof numerator.
            if digest != warmup_hash:
                raise RuntimeError('Output changed between warmup and captured sample')
            samples.append(dict(index=index, output_equal=True, terminal_completed=True,
                output_sha256=digest, host_start_unix_ns=start, host_end_unix_ns=time.time_ns()))
        write_json(output / 'capture.json', dict(
            status='workload_completed_pending_counter_review', case=case, pid=os.getpid(),
            kernel=name.decode(), nvtx_range=nvtx, device=0, pci_bus=bus.value.decode(),
            gpu=torch.cuda.get_device_name(0), capability=[8,6], l2_bytes=l2.value,
            precision='FP32_SIMT_no_Tensor_Core_no_TF32', python=sys.executable,
            torch=str(torch.__version__), cuda_build=torch.version.cuda,
            blocks=BLOCKS, threads_per_block=THREADS, tensor_bytes=tensor_bytes,
            allocator_fraction=None, allocator_policy="no_allocator_fraction_or_RAM_VRAM_quota",
            warmup_launches=2, targeted_launches=3, samples=samples, reference=reference,
            compilation=compilation, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            measurement_scope='NCU_kernel_replay; cold_cache_all; clocks_unchanged',
            timing_source='external_gpu__time_duration.sum; host timestamps are context only',
            roof_status='NOT_VALIDATED; no roof or diagnosis until offline evidence review',
        ))
    finally:
        checked(unload, module)
    print(json.dumps(dict(case=case,status='workload_completed',samples=len(samples))),flush=True)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--case',choices=['target','compute','memory'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();m=verify(a.manifest,expected=os.environ.get('HYSO_FULL_ACCEPTANCE_ADMITTED'),live=True)
    if not next(x for x in m['experiments'] if x['id']=='06.roofline')['authorization']['approved']:
        raise PermissionError('06.roofline NOT_AUTHORIZED')
    sys.path[:0]=[m['source_snapshot']['root'],*m['runtime']['site_paths']]
    if a.case=='target':target_case(a.output)
    else:run_case(a.case,a.output)
