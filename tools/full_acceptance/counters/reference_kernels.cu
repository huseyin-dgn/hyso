// Private P18 references. No Tensor Core instructions, no TF32, no atomics.
// Fixed dimensions: 240 blocks * 256 threads; eight accumulators, 4096 steps.
extern "C" __global__ void p18_compute_f32(float* output) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int bits = 0x3f800000u | ((tid & 255u) << 15);
    float base;
    asm("mov.b32 %0, %1;" : "=f"(base) : "r"(bits));
    float x0 = base, x1 = base + 0.0625f, x2 = base + 0.125f;
    float x3 = base + 0.1875f, x4 = base + 0.25f, x5 = base + 0.3125f;
    float x6 = base + 0.375f, x7 = base + 0.4375f;
    #pragma unroll 1
    for (unsigned int step = 0; step < 4096; ++step) {
        // Exactly rounded FP32 FMA: x = x * 1 + 1/1024.
        // Volatile assembly retains the actual instructions, including alpha=1.
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x0));
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x1));
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x2));
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x3));
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x4));
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x5));
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x6));
        asm volatile("fma.rn.f32 %0, %0, 0f3F800000, 0f3A800000;" : "+f"(x7));
    }
    output[8 * tid + 0] = x0; output[8 * tid + 1] = x1;
    output[8 * tid + 2] = x2; output[8 * tid + 3] = x3;
    output[8 * tid + 4] = x4; output[8 * tid + 5] = x5;
    output[8 * tid + 6] = x6; output[8 * tid + 7] = x7;
}

extern "C" __global__ void p18_dram_copy_f32(const float* input, float* output) {
    // Each buffer is 64 MiB: 16,777,216 FP32 elements, 4,194,304 vectors.
    // Single sweep, no repeated small tile and no arithmetic on copied FP32 bits.
    for (unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
         v < 4194304u; v += gridDim.x * blockDim.x) {
        float x0, x1, x2, x3;
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)
            : "l"(input + 4u * v) : "memory");
        asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};"
            : : "l"(output + 4u * v), "f"(x0), "f"(x1), "f"(x2), "f"(x3)
            : "memory");
    }
}
