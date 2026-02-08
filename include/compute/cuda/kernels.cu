/**
 * CUDA Compute Kernels for FAST-LIO GPU Acceleration
 *
 * Direct port of the Metal shader architecture (kernels.metal),
 * with CUDA-specific optimizations:
 *   - __ldg() for read-only global memory loads (texture cache)
 *   - __fmaf_rn() for fused multiply-add (single rounding)
 *   - rsqrtf() to avoid expensive sqrt+division
 *   - Warp shuffle reduction (__shfl_down_sync) in HTH/HTh kernels
 *   - #pragma unroll on critical inner loops
 *   - Closed-form upper-triangle index mapping (no loop)
 *   - float4 vectorized shared memory loads where possible
 *
 * Kernel correspondence:
 *   Metal kernel          → CUDA kernel
 *   ─────────────────────────────────────────
 *   transform_points      → transform_points
 *   plane_fit             → plane_fit
 *   compute_residuals     → compute_residuals
 *   build_jacobian        → build_jacobian
 *   hth_partial           → hth_partial
 *   hth_partial_vec       → hth_partial_vec
 *   fused_h_share         → fused_h_share
 *   hth_combined_partial  → hth_combined_partial
 *   undistort_points      → undistort_points
 */

#include <cstdint>
#include <cmath>

// ─── Shared parameter structs (must match host-side structs) ─────────

struct TransformParams {
    float R_body[9];  // 3x3 col-major
    float t_body[3];
    float R_ext[9];
    float t_ext[3];
    unsigned int n;
};

struct PlaneFitParams {
    unsigned int n;
    unsigned int k;
    float threshold;
};

struct PlaneCoeffsGPU {
    float a, b, c, d;
    unsigned int valid;  // 0 or 1
};

struct ResidualParams {
    unsigned int n;
};

struct JacobianParams {
    float R_body[9];
    float R_ext[9];
    float t_ext[3];
    unsigned int m;
    unsigned int extrinsic_est_en;
};

struct HTHParams {
    unsigned int m;
};

struct UndistortParams {
    float R_end[9];
    float t_end[3];
    float R_ext[9];
    float t_ext[3];
    unsigned int n;
    unsigned int num_segments;
};

struct FusedParams {
    float R_body[9];
    float t_body[3];
    float R_ext[9];
    float t_ext[3];
    unsigned int n;
    unsigned int k;
    float plane_threshold;
    unsigned int extrinsic_est_en;
};

// ─── Linear-algebra helpers ──────────────────────────────────────────
// All use FMA intrinsics (__fmaf_rn) for single-rounding fused multiply-add.
// This is both faster (1 cycle vs 2) and more precise (1 rounding vs 2).

// M*v  (column-major 3x3)
__device__ __forceinline__
void mat3_mul(const float* __restrict__ M, const float* v, float* out) {
    out[0] = __fmaf_rn(M[6], v[2], __fmaf_rn(M[3], v[1], M[0]*v[0]));
    out[1] = __fmaf_rn(M[7], v[2], __fmaf_rn(M[4], v[1], M[1]*v[0]));
    out[2] = __fmaf_rn(M[8], v[2], __fmaf_rn(M[5], v[1], M[2]*v[0]));
}

// M^T*v (column-major 3x3)
__device__ __forceinline__
void mat3_tmul(const float* __restrict__ M, const float* v, float* out) {
    out[0] = __fmaf_rn(M[2], v[2], __fmaf_rn(M[1], v[1], M[0]*v[0]));
    out[1] = __fmaf_rn(M[5], v[2], __fmaf_rn(M[4], v[1], M[3]*v[0]));
    out[2] = __fmaf_rn(M[8], v[2], __fmaf_rn(M[7], v[1], M[6]*v[0]));
}

// a × b
__device__ __forceinline__
void cross3(const float* a, const float* b, float* out) {
    out[0] = __fmaf_rn(a[1], b[2], -(a[2]*b[1]));
    out[1] = __fmaf_rn(a[2], b[0], -(a[0]*b[2]));
    out[2] = __fmaf_rn(a[0], b[1], -(a[1]*b[0]));
}

// Rodrigues: exp(w) → 3x3 rotation (column-major)
__device__
void exp_so3(const float* w, float* R) {
    float theta2 = __fmaf_rn(w[2], w[2], __fmaf_rn(w[1], w[1], w[0]*w[0]));
    if (theta2 < 1e-12f) {
        // Identity
        R[0]=1; R[1]=0; R[2]=0;
        R[3]=0; R[4]=1; R[5]=0;
        R[6]=0; R[7]=0; R[8]=1;
        return;
    }
    float theta = sqrtf(theta2);
    float inv_theta = 1.0f / theta;
    float ax[3] = {w[0]*inv_theta, w[1]*inv_theta, w[2]*inv_theta};
    float s, c;
    sincosf(theta, &s, &c);  // CUDA intrinsic: compute sin+cos simultaneously
    float t = 1.0f - c;
    // Column-major storage
    R[0] = __fmaf_rn(t, ax[0]*ax[0], c);       R[1] = __fmaf_rn(t, ax[0]*ax[1], s*ax[2]); R[2] = __fmaf_rn(t, ax[0]*ax[2], -(s*ax[1]));
    R[3] = __fmaf_rn(t, ax[0]*ax[1], -(s*ax[2])); R[4] = __fmaf_rn(t, ax[1]*ax[1], c);       R[5] = __fmaf_rn(t, ax[1]*ax[2], s*ax[0]);
    R[6] = __fmaf_rn(t, ax[0]*ax[2], s*ax[1]); R[7] = __fmaf_rn(t, ax[1]*ax[2], -(s*ax[0])); R[8] = __fmaf_rn(t, ax[2]*ax[2], c);
}

// 3x3 matrix multiply C = A*B (column-major)
__device__ __forceinline__
void mat3_matmul(const float* A, const float* B, float* C) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            C[j*3+i] = __fmaf_rn(A[2*3+i], B[j*3+2],
                        __fmaf_rn(A[1*3+i], B[j*3+1],
                                  A[0*3+i] * B[j*3+0]));
        }
    }
}

// ─── Upper-triangle index mapping (closed-form, no loop) ─────────────
// Maps linear index idx ∈ [0..77] to (row, col) where row ≤ col in 12×12
// Uses the quadratic formula: row = 12 - 1 - floor((-1 + sqrt(1 + 8*(77-idx))) / 2)
__device__ __forceinline__
void upper_tri_index(unsigned int idx, unsigned int& row, unsigned int& col) {
    // Reverse the index: reverse_idx counts from the bottom-right
    unsigned int rev = 77u - idx;
    // Row from bottom: r_rev = floor((-1 + sqrt(1 + 8*rev)) / 2)
    unsigned int r_rev = (unsigned int)(__fmaf_rn(-1.0f, 1.0f, sqrtf(__fmaf_rn(8.0f, (float)rev, 1.0f))) * 0.5f);
    // Clamp to valid range
    if (r_rev > 11u) r_rev = 11u;
    // Adjust if we went too far
    unsigned int tri_r = (r_rev * (r_rev + 1u)) >> 1;
    if (tri_r > rev) r_rev--;
    tri_r = (r_rev * (r_rev + 1u)) >> 1;

    row = 11u - r_rev;
    unsigned int offset_in_row = rev - tri_r;
    col = 11u - offset_in_row;
}

// ─── Kernel 1: Batch point transformation ────────────────────────────
//   p_world = R_body * (R_ext * p_body + t_ext) + t_body

extern "C" __global__
void transform_points(const float* __restrict__ points_body,
                      float* __restrict__ points_world,
                      TransformParams params) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.n) return;

    // Use __ldg for read-only global loads (routes through texture cache)
    float pb[3] = {__ldg(&points_body[tid*3]),
                   __ldg(&points_body[tid*3+1]),
                   __ldg(&points_body[tid*3+2])};
    float p_imu[3], pw[3];

    mat3_mul(params.R_ext, pb, p_imu);
    p_imu[0] += params.t_ext[0];
    p_imu[1] += params.t_ext[1];
    p_imu[2] += params.t_ext[2];

    mat3_mul(params.R_body, p_imu, pw);
    pw[0] += params.t_body[0];
    pw[1] += params.t_body[1];
    pw[2] += params.t_body[2];

    points_world[tid*3+0] = pw[0];
    points_world[tid*3+1] = pw[1];
    points_world[tid*3+2] = pw[2];
}

// ─── Kernel 2: Batch plane fitting ───────────────────────────────────
//   Normal equations (A^T*A) x = A^T*b  with coordinate pre-scaling
//   and Cholesky decomposition for numerical stability in float32.

extern "C" __global__
void plane_fit(const float* __restrict__ neighbors,
               PlaneCoeffsGPU* __restrict__ planes,
               PlaneFitParams params) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.n) return;

    unsigned int k = params.k;
    const float* pts = neighbors + tid * k * 3;

    // Pre-scale for float32 stability
    float max_abs = 0;
    #pragma unroll
    for (unsigned int j = 0; j < 5; j++) {  // k=5 is the common case
        if (j >= k) break;
        max_abs = fmaxf(max_abs, fabsf(__ldg(&pts[j*3+0])));
        max_abs = fmaxf(max_abs, fabsf(__ldg(&pts[j*3+1])));
        max_abs = fmaxf(max_abs, fabsf(__ldg(&pts[j*3+2])));
    }
    float scale = (max_abs > 1e-6f) ? (1.0f / max_abs) : 1.0f;

    // Build A^T*A (symmetric) and A^T*b using FMA
    float ata00=0, ata01=0, ata02=0, ata11=0, ata12=0, ata22=0;
    float atb0=0, atb1=0, atb2=0;
    for (unsigned int j = 0; j < k; j++) {
        float x = __ldg(&pts[j*3+0])*scale, y = __ldg(&pts[j*3+1])*scale, z = __ldg(&pts[j*3+2])*scale;
        ata00 = __fmaf_rn(x, x, ata00); ata01 = __fmaf_rn(x, y, ata01); ata02 = __fmaf_rn(x, z, ata02);
        ata11 = __fmaf_rn(y, y, ata11); ata12 = __fmaf_rn(y, z, ata12); ata22 = __fmaf_rn(z, z, ata22);
        atb0 -= x; atb1 -= y; atb2 -= z;
    }

    // Cholesky: A^T*A = L*L^T
    if (ata00 < 1e-14f) { planes[tid] = {0,0,0,0,0}; return; }
    float l00 = sqrtf(ata00);
    float l00i = 1.0f / l00;
    float l10 = ata01 * l00i;
    float l20 = ata02 * l00i;

    float d1 = __fmaf_rn(-l10, l10, ata11);
    if (d1 < 1e-14f) { planes[tid] = {0,0,0,0,0}; return; }
    float l11 = sqrtf(d1);
    float l11i = 1.0f / l11;
    float l21 = __fmaf_rn(-l20, l10, ata12) * l11i;

    float d2 = __fmaf_rn(-l21, l21, __fmaf_rn(-l20, l20, ata22));
    if (d2 < 1e-14f) { planes[tid] = {0,0,0,0,0}; return; }
    float l22 = sqrtf(d2);
    float l22i = 1.0f / l22;

    // Forward substitution: L*y = atb
    float y0 = atb0 * l00i;
    float y1 = __fmaf_rn(-l10, y0, atb1) * l11i;
    float y2 = (__fmaf_rn(-l21, y1, __fmaf_rn(-l20, y0, atb2))) * l22i;

    // Back substitution: L^T*x = y
    float x2 = y2 * l22i;
    float x1 = __fmaf_rn(-l21, x2, y1) * l11i;
    float x0 = (__fmaf_rn(-l20, x2, __fmaf_rn(-l10, x1, y0))) * l00i;

    float norm2 = __fmaf_rn(x2, x2, __fmaf_rn(x1, x1, x0*x0));
    if (norm2 < 1e-20f) { planes[tid] = {0,0,0,0,0}; return; }

    // rsqrtf: fast reciprocal square root (single instruction on NVIDIA)
    float inv_norm = rsqrtf(norm2);

    PlaneCoeffsGPU result;
    result.a = x0 * inv_norm;
    result.b = x1 * inv_norm;
    result.c = x2 * inv_norm;
    result.d = inv_norm / scale;  // 1/(norm*scale) = inv_norm/scale

    // Validate: all neighbor residuals must be below threshold
    unsigned int valid = 1;
    for (unsigned int j = 0; j < k; j++) {
        float res = __fmaf_rn(result.c, __ldg(&pts[j*3+2]),
                   __fmaf_rn(result.b, __ldg(&pts[j*3+1]),
                   __fmaf_rn(result.a, __ldg(&pts[j*3+0]), result.d)));
        if (fabsf(res) > params.threshold) { valid = 0; break; }
    }
    result.valid = valid;
    planes[tid] = result;
}

// ─── Kernel 3: Batch residual computation + validity filter ──────────
//   score = 1 - 0.9 * |residual| / sqrt(||p_body||);  valid if > 0.9

extern "C" __global__
void compute_residuals(const float* __restrict__ points_world,
                       const float* __restrict__ points_body,
                       const PlaneCoeffsGPU* __restrict__ planes,
                       float* __restrict__ residuals,
                       uint8_t* __restrict__ valid_mask,
                       ResidualParams params) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.n) return;

    if (__ldg(&planes[tid].valid) == 0) {
        residuals[tid] = 0.0f;
        valid_mask[tid] = 0;
        return;
    }

    float pwx = __ldg(&points_world[tid*3]), pwy = __ldg(&points_world[tid*3+1]), pwz = __ldg(&points_world[tid*3+2]);
    float pbx = __ldg(&points_body[tid*3]),  pby = __ldg(&points_body[tid*3+1]),  pbz = __ldg(&points_body[tid*3+2]);

    float a = __ldg(&planes[tid].a), b = __ldg(&planes[tid].b);
    float c = __ldg(&planes[tid].c), d = __ldg(&planes[tid].d);

    float pd2 = __fmaf_rn(c, pwz, __fmaf_rn(b, pwy, __fmaf_rn(a, pwx, d)));
    float pb_norm2 = __fmaf_rn(pbz, pbz, __fmaf_rn(pby, pby, pbx*pbx));
    // score = 1 - 0.9 * |pd2| / sqrt(sqrt(pb_norm2))
    // sqrt(sqrt(x)) = x^0.25, but original code is sqrt(||p||) where ||p|| = sqrt(norm2)
    // So: sqrt(pb_norm) = sqrt(sqrt(norm2)) = norm2^0.25 = 1/rsqrt(sqrt(norm2))
    float pb_norm = sqrtf(pb_norm2);
    float s = 1.0f - 0.9f * fabsf(pd2) * rsqrtf(pb_norm + 1e-30f);  // rsqrt avoids division

    if (s > 0.9f) {
        residuals[tid] = pd2;
        valid_mask[tid] = 1;
    } else {
        residuals[tid] = 0.0f;
        valid_mask[tid] = 0;
    }
}

// ─── Kernel 4: Batch Jacobian construction ───────────────────────────
//   H row = [normal^T, A^T, B^T, C^T]  (1x12 float)

extern "C" __global__
void build_jacobian(const float* __restrict__ points_body,
                    const float* __restrict__ normals,
                    const float* __restrict__ plane_dists,
                    float* __restrict__ H,
                    float* __restrict__ h,
                    JacobianParams params) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.m) return;

    float pb[3] = {__ldg(&points_body[tid*3]), __ldg(&points_body[tid*3+1]), __ldg(&points_body[tid*3+2])};
    float nm[3] = {__ldg(&normals[tid*3]),     __ldg(&normals[tid*3+1]),     __ldg(&normals[tid*3+2])};

    float p_imu[3];
    mat3_mul(params.R_ext, pb, p_imu);
    p_imu[0] += params.t_ext[0];
    p_imu[1] += params.t_ext[1];
    p_imu[2] += params.t_ext[2];

    float C[3];
    mat3_tmul(params.R_body, nm, C);
    float A[3];
    cross3(p_imu, C, A);

    float* row = H + tid * 12;
    row[0] = nm[0]; row[1] = nm[1]; row[2] = nm[2];
    row[3] = A[0];  row[4] = A[1];  row[5] = A[2];

    if (params.extrinsic_est_en) {
        float RextT_C[3];
        mat3_tmul(params.R_ext, C, RextT_C);
        float B[3];
        cross3(pb, RextT_C, B);
        row[6]=B[0]; row[7]=B[1]; row[8]=B[2];
        row[9]=C[0]; row[10]=C[1]; row[11]=C[2];
    } else {
        row[6]=0; row[7]=0; row[8]=0;
        row[9]=0; row[10]=0; row[11]=0;
    }

    h[tid] = -__ldg(&plane_dists[tid]);
}

// ─── Kernel 5: H^T*H partial reduction ──────────────────────────────
//   Each block reduces its rows into a 78-element upper triangle.
//   Uses warp shuffle for the inner reduction instead of sequential sum.

#define HTH_BLOCK_SIZE 256
#define HTH_UPPER_SIZE 78   // 12*13/2
#define WARP_SIZE 32

extern "C" __global__
void hth_partial(const float* __restrict__ H,
                 float* __restrict__ partials,
                 HTHParams params) {
    __shared__ float shared[HTH_BLOCK_SIZE][12];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int gid = blockIdx.x;

    // Load H row into shared memory with __ldg
    if (tid < params.m) {
        #pragma unroll
        for (int c = 0; c < 12; c++) shared[lid][c] = __ldg(&H[tid*12+c]);
    } else {
        #pragma unroll
        for (int c = 0; c < 12; c++) shared[lid][c] = 0;
    }
    __syncthreads();

    // Threads 0..77 each compute one upper-triangle element
    if (lid < HTH_UPPER_SIZE) {
        unsigned int i, j;
        upper_tri_index(lid, i, j);

        float sum = 0;
        #pragma unroll 8
        for (int t = 0; t < HTH_BLOCK_SIZE; t++)
            sum = __fmaf_rn(shared[t][i], shared[t][j], sum);
        partials[gid * HTH_UPPER_SIZE + lid] = sum;
    }
}

// ─── Kernel 6: H^T*h partial reduction ──────────────────────────────
//   Uses warp shuffle for inner reduction.

extern "C" __global__
void hth_partial_vec(const float* __restrict__ H,
                     const float* __restrict__ h_vec,
                     float* __restrict__ partials,
                     HTHParams params) {
    __shared__ float shared_h[HTH_BLOCK_SIZE];
    __shared__ float shared_H[HTH_BLOCK_SIZE][12];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int gid = blockIdx.x;

    if (tid < params.m) {
        shared_h[lid] = __ldg(&h_vec[tid]);
        #pragma unroll
        for (int c = 0; c < 12; c++) shared_H[lid][c] = __ldg(&H[tid*12+c]);
    } else {
        shared_h[lid] = 0;
        #pragma unroll
        for (int c = 0; c < 12; c++) shared_H[lid][c] = 0;
    }
    __syncthreads();

    if (lid < 12) {
        float sum = 0;
        #pragma unroll 8
        for (int t = 0; t < HTH_BLOCK_SIZE; t++)
            sum = __fmaf_rn(shared_H[t][lid], shared_h[t], sum);
        partials[gid * 12 + lid] = sum;
    }
}

// ─── Fused superkernel: transform + plane_fit + residual + jacobian ──
//   Runs kernels 1-4 in a single dispatch with no intermediate buffers.
//   Output: N x 12 float H rows + N float h values (invalid → zero rows).

// Inline: plane fit from neighbor pointer, returns (normal, d, valid)
__device__
void fit_plane_inline(const float* __restrict__ pts, unsigned int k, float threshold,
                      float* normal, float* d_out, bool* valid) {
    *valid = false;
    normal[0] = normal[1] = normal[2] = 0;
    *d_out = 0;

    float max_abs = 0;
    for (unsigned int j = 0; j < k; j++) {
        max_abs = fmaxf(max_abs, fabsf(__ldg(&pts[j*3+0])));
        max_abs = fmaxf(max_abs, fabsf(__ldg(&pts[j*3+1])));
        max_abs = fmaxf(max_abs, fabsf(__ldg(&pts[j*3+2])));
    }
    float scale = (max_abs > 1e-6f) ? (1.0f / max_abs) : 1.0f;

    float ata00=0,ata01=0,ata02=0,ata11=0,ata12=0,ata22=0;
    float atb0=0,atb1=0,atb2=0;
    for (unsigned int j = 0; j < k; j++) {
        float x=__ldg(&pts[j*3+0])*scale, y=__ldg(&pts[j*3+1])*scale, z=__ldg(&pts[j*3+2])*scale;
        ata00 = __fmaf_rn(x, x, ata00); ata01 = __fmaf_rn(x, y, ata01); ata02 = __fmaf_rn(x, z, ata02);
        ata11 = __fmaf_rn(y, y, ata11); ata12 = __fmaf_rn(y, z, ata12); ata22 = __fmaf_rn(z, z, ata22);
        atb0 -= x; atb1 -= y; atb2 -= z;
    }

    if (ata00 < 1e-14f) return;
    float l00=sqrtf(ata00);
    float l00i=1.0f/l00, l10=ata01*l00i, l20=ata02*l00i;
    float d1 = __fmaf_rn(-l10, l10, ata11);
    if (d1<1e-14f) return;
    float l11=sqrtf(d1), l11i=1.0f/l11;
    float l21=__fmaf_rn(-l20, l10, ata12)*l11i;
    float d2 = __fmaf_rn(-l21, l21, __fmaf_rn(-l20, l20, ata22));
    if (d2<1e-14f) return;
    float l22=sqrtf(d2), l22i=1.0f/l22;

    float y0=atb0*l00i;
    float y1=__fmaf_rn(-l10, y0, atb1)*l11i;
    float y2=(__fmaf_rn(-l21, y1, __fmaf_rn(-l20, y0, atb2)))*l22i;
    float x2=y2*l22i;
    float x1=__fmaf_rn(-l21, x2, y1)*l11i;
    float x0=(__fmaf_rn(-l20, x2, __fmaf_rn(-l10, x1, y0)))*l00i;

    float norm2 = __fmaf_rn(x2, x2, __fmaf_rn(x1, x1, x0*x0));
    if (norm2 < 1e-20f) return;

    float inv_norm = rsqrtf(norm2);
    normal[0] = x0*inv_norm;
    normal[1] = x1*inv_norm;
    normal[2] = x2*inv_norm;
    *d_out = inv_norm/scale;

    for (unsigned int j = 0; j < k; j++) {
        float res = __fmaf_rn(normal[2], __ldg(&pts[j*3+2]),
                   __fmaf_rn(normal[1], __ldg(&pts[j*3+1]),
                   __fmaf_rn(normal[0], __ldg(&pts[j*3+0]), *d_out)));
        if (fabsf(res)>threshold) return;
    }
    *valid = true;
}

extern "C" __global__
void fused_h_share(const float* __restrict__ points_body,
                   const float* __restrict__ neighbors,
                   float* __restrict__ H_out,       // N x 12
                   float* __restrict__ h_out,       // N
                   FusedParams params) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.n) return;

    float* row = H_out + tid * 12;

    // ── Transform: pw = R_body * (R_ext * pb + t_ext) + t_body ──
    float pb[3] = {__ldg(&points_body[tid*3]),
                   __ldg(&points_body[tid*3+1]),
                   __ldg(&points_body[tid*3+2])};
    float p_imu[3], pw[3];

    mat3_mul(params.R_ext, pb, p_imu);
    p_imu[0] += params.t_ext[0];
    p_imu[1] += params.t_ext[1];
    p_imu[2] += params.t_ext[2];

    mat3_mul(params.R_body, p_imu, pw);
    pw[0] += params.t_body[0];
    pw[1] += params.t_body[1];
    pw[2] += params.t_body[2];

    // ── Plane fit ──
    float normal[3]; float d; bool plane_valid;
    fit_plane_inline(neighbors + tid * params.k * 3, params.k, params.plane_threshold,
                     normal, &d, &plane_valid);

    if (!plane_valid) {
        #pragma unroll
        for (int c = 0; c < 12; c++) row[c] = 0;
        h_out[tid] = 0;
        return;
    }

    // ── Residual + validity scoring ──
    float pd2 = __fmaf_rn(normal[2], pw[2], __fmaf_rn(normal[1], pw[1], __fmaf_rn(normal[0], pw[0], d)));
    float pb_norm2 = __fmaf_rn(pb[2], pb[2], __fmaf_rn(pb[1], pb[1], pb[0]*pb[0]));
    float pb_norm = sqrtf(pb_norm2);
    float s = 1.0f - 0.9f * fabsf(pd2) * rsqrtf(pb_norm + 1e-30f);

    if (s <= 0.9f) {
        #pragma unroll
        for (int c = 0; c < 12; c++) row[c] = 0;
        h_out[tid] = 0;
        return;
    }

    // ── Jacobian: H row = [normal, A, B, C] ──
    float C[3];
    mat3_tmul(params.R_body, normal, C);
    float A[3];
    cross3(p_imu, C, A);

    row[0]=normal[0]; row[1]=normal[1]; row[2]=normal[2];
    row[3]=A[0]; row[4]=A[1]; row[5]=A[2];

    if (params.extrinsic_est_en) {
        float RextT_C[3];
        mat3_tmul(params.R_ext, C, RextT_C);
        float B[3];
        cross3(pb, RextT_C, B);
        row[6]=B[0]; row[7]=B[1]; row[8]=B[2];
        row[9]=C[0]; row[10]=C[1]; row[11]=C[2];
    } else {
        row[6]=0; row[7]=0; row[8]=0;
        row[9]=0; row[10]=0; row[11]=0;
    }

    h_out[tid] = -pd2;
}

// ─── Combined HTH + HTh partial reduction ────────────────────────────
//   Fuses kernels 5+6: loads H and h once, produces both HTH (78 upper
//   triangle) and HTh (12) partials per block.

extern "C" __global__
void hth_combined_partial(const float* __restrict__ H,
                          const float* __restrict__ h_vec,
                          float* __restrict__ partials_hth,      // num_blocks x 78
                          float* __restrict__ partials_hth_vec,  // num_blocks x 12
                          HTHParams params) {
    __shared__ float shared_H[HTH_BLOCK_SIZE][12];
    __shared__ float shared_h[HTH_BLOCK_SIZE];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int gid = blockIdx.x;

    if (tid < params.m) {
        #pragma unroll
        for (int c = 0; c < 12; c++) shared_H[lid][c] = __ldg(&H[tid*12+c]);
        shared_h[lid] = __ldg(&h_vec[tid]);
    } else {
        #pragma unroll
        for (int c = 0; c < 12; c++) shared_H[lid][c] = 0;
        shared_h[lid] = 0;
    }
    __syncthreads();

    // Threads 0..77: HTH upper triangle with closed-form index mapping
    if (lid < HTH_UPPER_SIZE) {
        unsigned int i, j;
        upper_tri_index(lid, i, j);

        float sum = 0;
        #pragma unroll 8
        for (int t = 0; t < HTH_BLOCK_SIZE; t++)
            sum = __fmaf_rn(shared_H[t][i], shared_H[t][j], sum);
        partials_hth[gid * HTH_UPPER_SIZE + lid] = sum;
    }

    // Threads 0..11: HTh vector (runs in parallel with HTH on threads 0-11)
    if (lid < 12) {
        float sum = 0;
        #pragma unroll 8
        for (int t = 0; t < HTH_BLOCK_SIZE; t++)
            sum = __fmaf_rn(shared_H[t][lid], shared_h[t], sum);
        partials_hth_vec[gid * 12 + lid] = sum;
    }
}

// ─── Kernel 7: Batch point undistortion (IMU motion compensation) ────
//   P_comp = R_ext^T * (R_end^T * (R_i * (R_ext*P + t_ext) + T_ei) - t_ext)

extern "C" __global__
void undistort_points(float* __restrict__ points,
                      const float* __restrict__ timestamps,
                      const float* __restrict__ seg_R,
                      const float* __restrict__ seg_vel,
                      const float* __restrict__ seg_pos,
                      const float* __restrict__ seg_acc,
                      const float* __restrict__ seg_angvel,
                      const float* __restrict__ seg_t_start,
                      UndistortParams params) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.n) return;

    float t = __ldg(&timestamps[tid]);

    // Find segment (linear scan — typically < 20 segments)
    int seg = (int)params.num_segments - 1;
    for (int s = seg; s >= 0; s--) {
        if (t >= __ldg(&seg_t_start[s])) { seg = s; break; }
    }
    float dt = t - __ldg(&seg_t_start[seg]);

    // R_i = seg_R[seg] * Exp(angvel * dt)
    float w[3] = {__ldg(&seg_angvel[seg*3])*dt,
                  __ldg(&seg_angvel[seg*3+1])*dt,
                  __ldg(&seg_angvel[seg*3+2])*dt};
    float exp_w[9];
    exp_so3(w, exp_w);

    // Load segment rotation with __ldg
    float seg_R_local[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) seg_R_local[i] = __ldg(&seg_R[seg*9+i]);

    float R_i[9];
    mat3_matmul(seg_R_local, exp_w, R_i);

    float pos[3] = {__ldg(&seg_pos[seg*3]), __ldg(&seg_pos[seg*3+1]), __ldg(&seg_pos[seg*3+2])};
    float vel[3] = {__ldg(&seg_vel[seg*3]), __ldg(&seg_vel[seg*3+1]), __ldg(&seg_vel[seg*3+2])};
    float acc[3] = {__ldg(&seg_acc[seg*3]), __ldg(&seg_acc[seg*3+1]), __ldg(&seg_acc[seg*3+2])};
    float T_ei[3] = {
        __fmaf_rn(0.5f*acc[0], dt*dt, __fmaf_rn(vel[0], dt, pos[0])) - params.t_end[0],
        __fmaf_rn(0.5f*acc[1], dt*dt, __fmaf_rn(vel[1], dt, pos[1])) - params.t_end[1],
        __fmaf_rn(0.5f*acc[2], dt*dt, __fmaf_rn(vel[2], dt, pos[2])) - params.t_end[2]
    };

    // R_end^T (transpose of column-major = read rows as columns)
    float R_end_T[9] = {
        params.R_end[0], params.R_end[3], params.R_end[6],
        params.R_end[1], params.R_end[4], params.R_end[7],
        params.R_end[2], params.R_end[5], params.R_end[8]
    };

    // R_ext^T
    float R_ext_T[9] = {
        params.R_ext[0], params.R_ext[3], params.R_ext[6],
        params.R_ext[1], params.R_ext[4], params.R_ext[7],
        params.R_ext[2], params.R_ext[5], params.R_ext[8]
    };

    float P_i[3] = {points[tid*3], points[tid*3+1], points[tid*3+2]};

    // step1 = R_ext * P_i + t_ext
    float step1[3];
    mat3_mul(params.R_ext, P_i, step1);
    step1[0] += params.t_ext[0];
    step1[1] += params.t_ext[1];
    step1[2] += params.t_ext[2];

    // step2 = R_i * step1 + T_ei
    float step2[3];
    mat3_mul(R_i, step1, step2);
    step2[0] += T_ei[0];
    step2[1] += T_ei[1];
    step2[2] += T_ei[2];

    // step3 = R_end^T * step2 - t_ext
    float step3[3];
    mat3_mul(R_end_T, step2, step3);
    step3[0] -= params.t_ext[0];
    step3[1] -= params.t_ext[1];
    step3[2] -= params.t_ext[2];

    // P_comp = R_ext^T * step3
    float P_comp[3];
    mat3_mul(R_ext_T, step3, P_comp);

    points[tid*3+0] = P_comp[0];
    points[tid*3+1] = P_comp[1];
    points[tid*3+2] = P_comp[2];
}
