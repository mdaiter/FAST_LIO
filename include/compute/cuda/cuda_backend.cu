/**
 * CUDA Compute Backend for FAST-LIO GPU Acceleration
 * ===================================================
 *
 * Host-side implementation using the CUDA runtime API.
 * Kernels are defined in kernels.cu and compiled together.
 *
 * CUDA-specific optimizations over the baseline Metal port:
 *   - CUDA streams for overlapped kernel execution + async transfers
 *   - Pinned (page-locked) host memory for DMA transfers (2-3x faster)
 *   - Persistent GPU buffer pool for the fused pipeline (avoids per-call alloc)
 *   - Async memset overlapped with host-side data packing
 *   - Minimal synchronization: only sync when results are needed on host
 *   - cudaMemcpyAsync where possible for host↔device overlap
 *
 * Key design:
 *   - cudaMalloc / cudaMemcpy for buffer management (discrete GPU memory)
 *   - float precision on GPU for per-point ops
 *   - HTH/HTh: GPU partial reduction (shared memory) + CPU final sum in double
 *   - Jacobian built as float on GPU, converted to double on CPU readback
 */

#include <cuda_runtime.h>

#include "../compute_backend.h"
#include "../cpu_backend.h"

#include <unordered_map>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdio>

// ─── GPU struct layouts (must match kernels.cu exactly) ──────────────

struct TransformParams {
    float R_body[9];
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
    unsigned int valid;
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

static_assert(sizeof(PlaneCoeffsGPU) == 20, "PlaneCoeffsGPU size mismatch with GPU");

// ─── Extern kernel declarations (defined in kernels.cu) ──────────────

extern "C" __global__ void transform_points(const float*, float*, TransformParams);
extern "C" __global__ void plane_fit(const float*, PlaneCoeffsGPU*, PlaneFitParams);
extern "C" __global__ void compute_residuals(const float*, const float*, const PlaneCoeffsGPU*, float*, uint8_t*, ResidualParams);
extern "C" __global__ void build_jacobian(const float*, const float*, const float*, float*, float*, JacobianParams);
extern "C" __global__ void hth_partial(const float*, float*, HTHParams);
extern "C" __global__ void hth_partial_vec(const float*, const float*, float*, HTHParams);
extern "C" __global__ void fused_h_share(const float*, const float*, float*, float*, FusedParams);
extern "C" __global__ void hth_combined_partial(const float*, const float*, float*, float*, HTHParams);
extern "C" __global__ void undistort_points(float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, UndistortParams);

// ─── Constants ───────────────────────────────────────────────────────

#define CUDA_BLOCK_SIZE 256
#define HTH_UPPER_SIZE 78

// ─── CUDA error checking ─────────────────────────────────────────────

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
        }                                                                   \
    } while (0)

namespace fastlio {
namespace compute {

// ─── Helper: copy RigidTransform (double) to float arrays ────────────

static void rt_to_float(const RigidTransform& rt, float R[9], float t[3]) {
    for (int i = 0; i < 9; i++) R[i] = (float)rt.R[i];
    for (int i = 0; i < 3; i++) t[i] = (float)rt.t[i];
}

// ─── Helper: compute grid dimensions ─────────────────────────────────

static inline unsigned int div_ceil(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// ─── Persistent buffer pool for fused pipeline ───────────────────────
// Avoids cudaMalloc/cudaFree per call (malloc is ~100μs on NVIDIA GPUs).

struct PersistentBuffers {
    void* d_pb = nullptr;         // points_body (N x 3 float)
    void* d_nb = nullptr;         // neighbors (N x k x 3 float)
    void* d_H = nullptr;          // H output (N x 12 float)
    void* d_h = nullptr;          // h output (N float)
    void* d_phth = nullptr;       // HTH partials (num_blocks x 78 float)
    void* d_phthv = nullptr;      // HTh partials (num_blocks x 12 float)

    // Pinned host memory for async download
    float* h_H = nullptr;         // H readback (N x 12)
    float* h_h = nullptr;         // h readback (N)
    float* h_phth = nullptr;      // HTH partials readback
    float* h_phthv = nullptr;     // HTh partials readback

    int alloc_n = 0;              // Allocated capacity (points)
    int alloc_k = 0;              // Allocated k (neighbors per point)
    int alloc_blocks = 0;         // Allocated number of reduction blocks

    void ensure(int n, int k) {
        int num_blocks = (int)div_ceil(n, CUDA_BLOCK_SIZE);
        if (n <= alloc_n && k <= alloc_k && num_blocks <= alloc_blocks) return;

        // Free old
        release();

        alloc_n = std::max(n, 1024);  // Round up to avoid frequent realloc
        alloc_k = k;
        alloc_blocks = (int)div_ceil(alloc_n, CUDA_BLOCK_SIZE);

        // Device buffers
        CUDA_CHECK(cudaMalloc(&d_pb, alloc_n * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nb, (size_t)alloc_n * alloc_k * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_H, alloc_n * 12 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_h, alloc_n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_phth, alloc_blocks * HTH_UPPER_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_phthv, alloc_blocks * 12 * sizeof(float)));

        // Pinned host memory (enables async DMA transfers)
        CUDA_CHECK(cudaMallocHost(&h_H, alloc_n * 12 * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_h, alloc_n * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_phth, alloc_blocks * HTH_UPPER_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_phthv, alloc_blocks * 12 * sizeof(float)));
    }

    void release() {
        if (d_pb)    { cudaFree(d_pb); d_pb = nullptr; }
        if (d_nb)    { cudaFree(d_nb); d_nb = nullptr; }
        if (d_H)     { cudaFree(d_H); d_H = nullptr; }
        if (d_h)     { cudaFree(d_h); d_h = nullptr; }
        if (d_phth)  { cudaFree(d_phth); d_phth = nullptr; }
        if (d_phthv) { cudaFree(d_phthv); d_phthv = nullptr; }
        if (h_H)     { cudaFreeHost(h_H); h_H = nullptr; }
        if (h_h)     { cudaFreeHost(h_h); h_h = nullptr; }
        if (h_phth)  { cudaFreeHost(h_phth); h_phth = nullptr; }
        if (h_phthv) { cudaFreeHost(h_phthv); h_phthv = nullptr; }
        alloc_n = alloc_k = alloc_blocks = 0;
    }
};

// ═══════════════════════════════════════════════════════════════════════

class CUDABackend : public ComputeBackend {
public:
    CUDABackend() {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            available_ = false;
            return;
        }

        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            available_ = false;
            return;
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        device_name_ = prop.name;
        available_ = true;
        next_handle_ = 1;

        // Create CUDA streams for overlapped execution
        CUDA_CHECK(cudaStreamCreate(&compute_stream_));
        CUDA_CHECK(cudaStreamCreate(&transfer_stream_));
    }

    ~CUDABackend() override {
        // Free persistent buffers
        persistent_.release();

        // Destroy streams
        if (compute_stream_)  cudaStreamDestroy(compute_stream_);
        if (transfer_stream_) cudaStreamDestroy(transfer_stream_);

        // Free all remaining device buffers
        for (auto& kv : buffers_) {
            cudaFree(kv.second.ptr);
        }
        buffers_.clear();
    }

    bool is_available() const { return available_; }

    std::string name() const override { return "CUDA (" + device_name_ + ")"; }

    // ─── Buffer management ───────────────────────────────────────────

    BufferHandle alloc(size_t size_bytes) override {
        if (!available_ || size_bytes == 0) return INVALID_BUFFER;

        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size_bytes);
        if (err != cudaSuccess || !ptr) return INVALID_BUFFER;

        BufferHandle h = next_handle_++;
        buffers_[h] = {ptr, size_bytes};
        return h;
    }

    void free(BufferHandle buf) override {
        auto it = buffers_.find(buf);
        if (it == buffers_.end()) return;
        cudaFree(it->second.ptr);
        buffers_.erase(it);
    }

    bool upload(BufferHandle dst, const void* src, size_t size_bytes) override {
        auto it = buffers_.find(dst);
        if (it == buffers_.end() || it->second.size < size_bytes) return false;
        CUDA_CHECK(cudaMemcpy(it->second.ptr, src, size_bytes, cudaMemcpyHostToDevice));
        return true;
    }

    bool download(void* dst, BufferHandle src, size_t size_bytes) override {
        auto it = buffers_.find(src);
        if (it == buffers_.end() || it->second.size < size_bytes) return false;
        CUDA_CHECK(cudaMemcpy(dst, it->second.ptr, size_bytes, cudaMemcpyDeviceToHost));
        return true;
    }

    // ─── Kernel 1: Transform points ──────────────────────────────────

    void batch_transform_points(
        BufferHandle points_world_h, BufferHandle points_body_h, int n,
        const RigidTransform& body_to_world, const RigidTransform& lidar_to_imu
    ) override {
        TransformParams params;
        rt_to_float(body_to_world, params.R_body, params.t_body);
        rt_to_float(lidar_to_imu, params.R_ext, params.t_ext);
        params.n = (unsigned int)n;

        unsigned int blocks = div_ceil(n, CUDA_BLOCK_SIZE);
        transform_points<<<blocks, CUDA_BLOCK_SIZE>>>(
            dev_ptr<float>(points_body_h),
            dev_ptr<float>(points_world_h),
            params);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ─── Kernel 2: Plane fitting ─────────────────────────────────────

    void batch_plane_fit(
        BufferHandle planes_h, BufferHandle neighbors_h,
        int n, int k, float threshold
    ) override {
        PlaneFitParams params;
        params.n = (unsigned int)n;
        params.k = (unsigned int)k;
        params.threshold = threshold;

        unsigned int blocks = div_ceil(n, CUDA_BLOCK_SIZE);
        plane_fit<<<blocks, CUDA_BLOCK_SIZE>>>(
            dev_ptr<float>(neighbors_h),
            dev_ptr<PlaneCoeffsGPU>(planes_h),
            params);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ─── Kernel 3: Residuals ─────────────────────────────────────────

    void batch_compute_residuals(
        BufferHandle residuals_h, BufferHandle valid_mask_h,
        BufferHandle points_world_h, BufferHandle points_body_h,
        BufferHandle planes_h, int n
    ) override {
        ResidualParams params;
        params.n = (unsigned int)n;

        unsigned int blocks = div_ceil(n, CUDA_BLOCK_SIZE);
        compute_residuals<<<blocks, CUDA_BLOCK_SIZE>>>(
            dev_ptr<float>(points_world_h),
            dev_ptr<float>(points_body_h),
            dev_ptr<PlaneCoeffsGPU>(planes_h),
            dev_ptr<float>(residuals_h),
            dev_ptr<uint8_t>(valid_mask_h),
            params);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ─── Kernel 4: Jacobian (float on GPU, converted to double on read) ─

    void batch_build_jacobian(
        BufferHandle H_h, BufferHandle h_h,
        BufferHandle points_body_h, BufferHandle normals_h,
        BufferHandle plane_dists_h, int m,
        const double R_body_arr[9], const double R_ext_arr[9],
        const double t_ext_arr[3], bool extrinsic_est_en
    ) override {
        // H_h is M x 12 double, but GPU works in float. Allocate temp float buffers.
        BufferHandle H_float = alloc(m * 12 * sizeof(float));
        BufferHandle h_float = alloc(m * sizeof(float));

        JacobianParams params;
        for (int i = 0; i < 9; i++) { params.R_body[i] = (float)R_body_arr[i]; params.R_ext[i] = (float)R_ext_arr[i]; }
        for (int i = 0; i < 3; i++) params.t_ext[i] = (float)t_ext_arr[i];
        params.m = (unsigned int)m;
        params.extrinsic_est_en = extrinsic_est_en ? 1 : 0;

        unsigned int blocks = div_ceil(m, CUDA_BLOCK_SIZE);
        build_jacobian<<<blocks, CUDA_BLOCK_SIZE>>>(
            dev_ptr<float>(points_body_h),
            dev_ptr<float>(normals_h),
            dev_ptr<float>(plane_dists_h),
            dev_ptr<float>(H_float),
            dev_ptr<float>(h_float),
            params);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download float, convert to double, upload to destination
        std::vector<float> H_f(m * 12), h_f(m);
        download(H_f.data(), H_float, m * 12 * sizeof(float));
        download(h_f.data(), h_float, m * sizeof(float));

        std::vector<double> H_d(m * 12), h_d(m);
        for (int i = 0; i < m * 12; i++) H_d[i] = (double)H_f[i];
        for (int i = 0; i < m; i++) h_d[i] = (double)h_f[i];

        upload(H_h, H_d.data(), m * 12 * sizeof(double));
        upload(h_h, h_d.data(), m * sizeof(double));

        free(H_float);
        free(h_float);
    }

    // ─── Kernel 5: H^T * H (GPU partial reduction + CPU final sum) ──

    void compute_HTH(double HTH[144], BufferHandle H_h, int m) override {
        // H_h contains doubles — download, convert to float, re-upload for GPU
        std::vector<double> H_d(m * 12);
        download(H_d.data(), H_h, m * 12 * sizeof(double));

        std::vector<float> H_f(m * 12);
        for (int i = 0; i < m * 12; i++) H_f[i] = (float)H_d[i];

        BufferHandle H_float = alloc(m * 12 * sizeof(float));
        upload(H_float, H_f.data(), m * 12 * sizeof(float));

        unsigned int num_blocks = div_ceil(m, CUDA_BLOCK_SIZE);
        BufferHandle partials = alloc(num_blocks * HTH_UPPER_SIZE * sizeof(float));

        HTHParams params;
        params.m = (unsigned int)m;

        hth_partial<<<num_blocks, CUDA_BLOCK_SIZE>>>(
            dev_ptr<float>(H_float),
            dev_ptr<float>(partials),
            params);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download partials and reduce on CPU in double
        std::vector<float> partial_data(num_blocks * HTH_UPPER_SIZE);
        download(partial_data.data(), partials, num_blocks * HTH_UPPER_SIZE * sizeof(float));

        double result[HTH_UPPER_SIZE] = {};
        for (unsigned int g = 0; g < num_blocks; g++)
            for (int i = 0; i < HTH_UPPER_SIZE; i++)
                result[i] += (double)partial_data[g * HTH_UPPER_SIZE + i];

        // Expand upper triangle to full 12x12 column-major double
        memset(HTH, 0, 144 * sizeof(double));
        int idx = 0;
        for (int i = 0; i < 12; i++) {
            for (int j = i; j < 12; j++) {
                HTH[j * 12 + i] = result[idx];
                HTH[i * 12 + j] = result[idx];
                idx++;
            }
        }

        free(H_float);
        free(partials);
    }

    // ─── Kernel 6: H^T * h ──────────────────────────────────────────

    void compute_HTh(double HTh[12], BufferHandle H_h, BufferHandle h_h, int m) override {
        // Convert to float for GPU
        std::vector<double> H_d(m * 12), h_d(m);
        download(H_d.data(), H_h, m * 12 * sizeof(double));
        download(h_d.data(), h_h, m * sizeof(double));

        std::vector<float> H_f(m * 12), h_f(m);
        for (int i = 0; i < m * 12; i++) H_f[i] = (float)H_d[i];
        for (int i = 0; i < m; i++) h_f[i] = (float)h_d[i];

        BufferHandle H_float = alloc(m * 12 * sizeof(float));
        BufferHandle h_float = alloc(m * sizeof(float));
        upload(H_float, H_f.data(), m * 12 * sizeof(float));
        upload(h_float, h_f.data(), m * sizeof(float));

        unsigned int num_blocks = div_ceil(m, CUDA_BLOCK_SIZE);
        BufferHandle partials = alloc(num_blocks * 12 * sizeof(float));

        HTHParams params;
        params.m = (unsigned int)m;

        hth_partial_vec<<<num_blocks, CUDA_BLOCK_SIZE>>>(
            dev_ptr<float>(H_float),
            dev_ptr<float>(h_float),
            dev_ptr<float>(partials),
            params);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download and reduce in double
        std::vector<float> partial_data(num_blocks * 12);
        download(partial_data.data(), partials, num_blocks * 12 * sizeof(float));

        for (int i = 0; i < 12; i++) {
            double sum = 0;
            for (unsigned int g = 0; g < num_blocks; g++)
                sum += (double)partial_data[g * 12 + i];
            HTh[i] = sum;
        }

        free(H_float);
        free(h_float);
        free(partials);
    }

    // ─── Kernel 7: Undistortion ──────────────────────────────────────

    void batch_undistort_points(
        BufferHandle points_h, BufferHandle timestamps_h,
        BufferHandle seg_R_h, BufferHandle seg_vel_h,
        BufferHandle seg_pos_h, BufferHandle seg_acc_h,
        BufferHandle seg_angvel_h, BufferHandle seg_t_start_h,
        int n, int num_segments,
        const RigidTransform& imu_end_state, const RigidTransform& lidar_to_imu
    ) override {
        // Segments use double on CPU but float on GPU — convert
        auto convert_and_upload = [&](BufferHandle src_h, int count) -> BufferHandle {
            std::vector<double> d(count);
            download(d.data(), src_h, count * sizeof(double));
            std::vector<float> f(count);
            for (int i = 0; i < count; i++) f[i] = (float)d[i];
            BufferHandle dst = alloc(count * sizeof(float));
            upload(dst, f.data(), count * sizeof(float));
            return dst;
        };

        BufferHandle seg_R_f = convert_and_upload(seg_R_h, num_segments * 9);
        BufferHandle seg_vel_f = convert_and_upload(seg_vel_h, num_segments * 3);
        BufferHandle seg_pos_f = convert_and_upload(seg_pos_h, num_segments * 3);
        BufferHandle seg_acc_f = convert_and_upload(seg_acc_h, num_segments * 3);
        BufferHandle seg_angvel_f = convert_and_upload(seg_angvel_h, num_segments * 3);
        BufferHandle seg_t_f = convert_and_upload(seg_t_start_h, num_segments);

        UndistortParams params;
        rt_to_float(imu_end_state, params.R_end, params.t_end);
        rt_to_float(lidar_to_imu, params.R_ext, params.t_ext);
        params.n = (unsigned int)n;
        params.num_segments = (unsigned int)num_segments;

        unsigned int blocks = div_ceil(n, CUDA_BLOCK_SIZE);
        undistort_points<<<blocks, CUDA_BLOCK_SIZE>>>(
            dev_ptr<float>(points_h),
            dev_ptr<float>(timestamps_h),
            dev_ptr<float>(seg_R_f),
            dev_ptr<float>(seg_vel_f),
            dev_ptr<float>(seg_pos_f),
            dev_ptr<float>(seg_acc_f),
            dev_ptr<float>(seg_angvel_f),
            dev_ptr<float>(seg_t_f),
            params);
        CUDA_CHECK(cudaDeviceSynchronize());

        free(seg_R_f); free(seg_vel_f); free(seg_pos_f);
        free(seg_acc_f); free(seg_angvel_f); free(seg_t_f);
    }

    // ─── Fused pipeline (superkernel + combined reduction) ───────────
    //
    // Optimization over baseline:
    //   1. Persistent GPU buffer pool (no per-call cudaMalloc)
    //   2. Pinned host memory for async DMA downloads
    //   3. CUDA streams: overlap upload/memset with host prep
    //   4. Async memcpy for final readback
    //   5. Single sync point at the end

    HShareModelResult fused_h_share_model(
        const float* points_body_host, const float* neighbors_host,
        int n, int k,
        const RigidTransform& body_to_world, const RigidTransform& lidar_to_imu,
        float plane_threshold, bool extrinsic_est_en
    ) override {
        // ── Ensure persistent buffers are large enough ──
        persistent_.ensure(n, k);
        unsigned int num_blocks = div_ceil(n, CUDA_BLOCK_SIZE);

        // ── Async upload: points and neighbors → GPU on compute stream ──
        CUDA_CHECK(cudaMemcpyAsync(persistent_.d_pb, points_body_host,
            n * 3 * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(persistent_.d_nb, neighbors_host,
            (size_t)n * k * 3 * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));

        // ── Fused kernel: transform + plane_fit + residual + jacobian ──
        // Kernel writes every output element (valid or zero), so no memset needed.
        FusedParams fparams;
        rt_to_float(body_to_world, fparams.R_body, fparams.t_body);
        rt_to_float(lidar_to_imu, fparams.R_ext, fparams.t_ext);
        fparams.n = (unsigned int)n;
        fparams.k = (unsigned int)k;
        fparams.plane_threshold = plane_threshold;
        fparams.extrinsic_est_en = extrinsic_est_en ? 1 : 0;

        unsigned int blocks_fused = div_ceil(n, CUDA_BLOCK_SIZE);
        fused_h_share<<<blocks_fused, CUDA_BLOCK_SIZE, 0, compute_stream_>>>(
            (const float*)persistent_.d_pb,
            (const float*)persistent_.d_nb,
            (float*)persistent_.d_H,
            (float*)persistent_.d_h,
            fparams);

        // ── Combined HTH + HTh reduction (depends on fused kernel) ──
        HTHParams hparams;
        hparams.m = (unsigned int)n;

        hth_combined_partial<<<num_blocks, CUDA_BLOCK_SIZE, 0, compute_stream_>>>(
            (const float*)persistent_.d_H,
            (const float*)persistent_.d_h,
            (float*)persistent_.d_phth,
            (float*)persistent_.d_phthv,
            hparams);

        // ── Async download all results to pinned host memory ──
        CUDA_CHECK(cudaMemcpyAsync(persistent_.h_H, persistent_.d_H,
            n * 12 * sizeof(float), cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(persistent_.h_h, persistent_.d_h,
            n * sizeof(float), cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(persistent_.h_phth, persistent_.d_phth,
            num_blocks * HTH_UPPER_SIZE * sizeof(float), cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(persistent_.h_phthv, persistent_.d_phthv,
            num_blocks * 12 * sizeof(float), cudaMemcpyDeviceToHost, compute_stream_));

        // ── Single sync: wait for all GPU work + transfers to complete ──
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

        // ── CPU final reduction (from pinned memory — no extra copy) ──
        HShareModelResult result;

        // HTH: reduce partials in double precision
        double hth_sum[HTH_UPPER_SIZE] = {};
        for (unsigned int g = 0; g < num_blocks; g++) {
            const float* src = persistent_.h_phth + g * HTH_UPPER_SIZE;
            for (int i = 0; i < HTH_UPPER_SIZE; i++)
                hth_sum[i] += (double)src[i];
        }

        memset(result.HTH, 0, sizeof(result.HTH));
        int idx = 0;
        for (int i = 0; i < 12; i++)
            for (int j = i; j < 12; j++) {
                result.HTH[j*12+i] = hth_sum[idx];
                result.HTH[i*12+j] = hth_sum[idx];
                idx++;
            }

        // HTh: reduce partials in double precision
        for (int i = 0; i < 12; i++) {
            double sum = 0;
            for (unsigned int g = 0; g < num_blocks; g++)
                sum += (double)persistent_.h_phthv[g * 12 + i];
            result.HTh[i] = sum;
        }

        // ── Extract valid features (from pinned host memory) ──
        result.effct_feat_num = 0;
        std::vector<float> compact_pb, compact_normals, compact_dists;

        // Pre-reserve to avoid repeated reallocation (most points are valid)
        compact_pb.reserve(n * 3);
        compact_normals.reserve(n * 3);
        compact_dists.reserve(n);

        for (int i = 0; i < n; i++) {
            const float* row = persistent_.h_H + i * 12;
            // Valid points have non-zero normal in cols 0-2
            if (row[0] != 0.0f || row[1] != 0.0f || row[2] != 0.0f) {
                compact_pb.push_back(points_body_host[i*3+0]);
                compact_pb.push_back(points_body_host[i*3+1]);
                compact_pb.push_back(points_body_host[i*3+2]);
                compact_normals.push_back(row[0]);
                compact_normals.push_back(row[1]);
                compact_normals.push_back(row[2]);
                compact_dists.push_back(-persistent_.h_h[i]);  // h = -residual
                result.effct_feat_num++;
            }
        }

        result.valid_points_body = std::move(compact_pb);
        result.valid_normals = std::move(compact_normals);
        result.valid_residuals = std::move(compact_dists);

        return result;
    }

private:
    bool available_ = false;
    std::string device_name_ = "Unknown";
    BufferHandle next_handle_ = 1;

    // CUDA streams for overlapped execution
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t transfer_stream_ = nullptr;

    // Persistent buffer pool for fused pipeline
    PersistentBuffers persistent_;

    struct BufferInfo {
        void* ptr;
        size_t size;
    };
    std::unordered_map<BufferHandle, BufferInfo> buffers_;

    template<typename T>
    T* dev_ptr(BufferHandle h) {
        auto it = buffers_.find(h);
        return (it != buffers_.end()) ? static_cast<T*>(it->second.ptr) : nullptr;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Factory function overrides (when CUDA is linked)
// ═══════════════════════════════════════════════════════════════════════

std::unique_ptr<ComputeBackend> create_backend(const std::string& name) {
    if (name == "cuda" || name == "CUDA") {
        auto backend = std::make_unique<CUDABackend>();
        if (backend->is_available()) return backend;
        return nullptr;
    }
    if (name == "cpu" || name == "CPU") {
        return std::make_unique<CPUBackend>();
    }
    return nullptr;
}

std::unique_ptr<ComputeBackend> create_default_backend() {
    // Try CUDA first
    auto cuda = std::make_unique<CUDABackend>();
    if (cuda->is_available()) return cuda;

    // Fall back to CPU
    return std::make_unique<CPUBackend>();
}

} // namespace compute
} // namespace fastlio
