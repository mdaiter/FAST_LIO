/**
 * Metal Compute Backend for FAST-LIO GPU Acceleration
 * ===================================================
 *
 * Objective-C++ implementation using the Metal framework.
 * Compiles .metal shaders at build time into a .metallib,
 * which is loaded at runtime.
 *
 * Key design:
 *   - Shared memory (Apple Silicon unified memory) — MTLResourceStorageModeShared
 *     means zero-copy between CPU and GPU. BufferHandles map directly to MTLBuffer.
 *   - float precision on GPU for per-point ops
 *   - HTH/HTh: GPU partial reduction + CPU final sum
 *   - Jacobian built as float on GPU, converted to double on CPU readback
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "compute_backend.h"
#include "cpu_backend.h"   // for fused_h_share_model fallback / factory

#include <unordered_map>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

// ─── GPU struct layouts (must match kernels.metal exactly) ───────────

struct TransformParams {
    float R_body[9];
    float t_body[3];
    float R_ext[9];
    float t_ext[3];
    uint32_t n;
};

struct PlaneFitParams {
    uint32_t n;
    uint32_t k;
    float threshold;
};

struct PlaneCoeffsGPU {
    float a, b, c, d;
    uint32_t valid;
};

struct ResidualParams {
    uint32_t n;
};

struct JacobianParams {
    float R_body[9];
    float R_ext[9];
    float t_ext[3];
    uint32_t m;
    uint32_t extrinsic_est_en;
};

struct HTHParams {
    uint32_t m;
};

struct UndistortParams {
    float R_end[9];
    float t_end[3];
    float R_ext[9];
    float t_ext[3];
    uint32_t n;
    uint32_t num_segments;
};

struct FusedParams {
    float R_body[9];
    float t_body[3];
    float R_ext[9];
    float t_ext[3];
    uint32_t n;
    uint32_t k;
    float plane_threshold;
    uint32_t extrinsic_est_en;
};

static_assert(sizeof(PlaneCoeffsGPU) == 20, "PlaneCoeffsGPU size mismatch with GPU");

namespace fastlio {
namespace compute {

// ─── Helper: copy RigidTransform (double) to float arrays ────────────

static void rt_to_float(const RigidTransform& rt, float R[9], float t[3]) {
    for (int i = 0; i < 9; i++) R[i] = (float)rt.R[i];
    for (int i = 0; i < 3; i++) t[i] = (float)rt.t[i];
}

// ═══════════════════════════════════════════════════════════════════════

class MetalBackend : public ComputeBackend {
public:
    MetalBackend() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) return;

            queue_ = [device_ newCommandQueue];

            // Load the metallib from the same directory as the executable,
            // or from the compile-time embedded path
            NSError* error = nil;
            NSString* libPath = find_metallib();
            if (libPath) {
                NSURL* libURL = [NSURL fileURLWithPath:libPath];
                library_ = [device_ newLibraryWithURL:libURL error:&error];
            }

            if (!library_) {
                NSLog(@"Metal: Failed to load library: %@", error);
                device_ = nil;
                return;
            }

            // Create compute pipeline states for each kernel
            pso_transform_     = make_pso(@"transform_points");
            pso_plane_fit_     = make_pso(@"plane_fit");
            pso_residuals_     = make_pso(@"compute_residuals");
            pso_jacobian_      = make_pso(@"build_jacobian");
            pso_hth_partial_   = make_pso(@"hth_partial");
            pso_hth_vec_       = make_pso(@"hth_partial_vec");
            pso_undistort_     = make_pso(@"undistort_points");
            pso_fused_         = make_pso(@"fused_h_share");
            pso_hth_combined_  = make_pso(@"hth_combined_partial");

            if (!pso_transform_ || !pso_plane_fit_ || !pso_residuals_ ||
                !pso_jacobian_ || !pso_hth_partial_ || !pso_hth_vec_ || !pso_undistort_ ||
                !pso_fused_ || !pso_hth_combined_) {
                NSLog(@"Metal: Failed to create one or more pipeline states");
                device_ = nil;
                return;
            }

            next_handle_ = 1;
        }
    }

    ~MetalBackend() override {
        @autoreleasepool {
            buffers_.clear();
            // ARC handles release of Metal objects
        }
    }

    bool is_available() const { return device_ != nil; }

    std::string name() const override { return "Metal"; }

    // ─── Buffer management (shared memory — zero copy on Apple Silicon) ──

    BufferHandle alloc(size_t size_bytes) override {
        @autoreleasepool {
            if (!device_ || size_bytes == 0) return INVALID_BUFFER;
            // Metal requires minimum 16-byte alignment, and buffer size > 0
            size_t alloc_size = std::max(size_bytes, (size_t)16);
            id<MTLBuffer> buf = [device_ newBufferWithLength:alloc_size
                                         options:MTLResourceStorageModeShared];
            if (!buf) return INVALID_BUFFER;

            BufferHandle h = next_handle_++;
            buffers_[h] = {buf, alloc_size};
            return h;
        }
    }

    void free(BufferHandle buf) override {
        buffers_.erase(buf);
    }

    bool upload(BufferHandle dst, const void* src, size_t size_bytes) override {
        auto it = buffers_.find(dst);
        if (it == buffers_.end() || it->second.size < size_bytes) return false;
        memcpy([it->second.buffer contents], src, size_bytes);
        return true;
    }

    bool download(void* dst, BufferHandle src, size_t size_bytes) override {
        auto it = buffers_.find(src);
        if (it == buffers_.end() || it->second.size < size_bytes) return false;
        memcpy(dst, [it->second.buffer contents], size_bytes);
        return true;
    }

    // ─── Kernel 1: Transform points ──────────────────────────────────

    void batch_transform_points(
        BufferHandle points_world_h, BufferHandle points_body_h, int n,
        const RigidTransform& body_to_world, const RigidTransform& lidar_to_imu
    ) override {
        @autoreleasepool {
            TransformParams params;
            rt_to_float(body_to_world, params.R_body, params.t_body);
            rt_to_float(lidar_to_imu, params.R_ext, params.t_ext);
            params.n = (uint32_t)n;

            dispatch_kernel(pso_transform_, n,
                {mtl_buf(points_body_h), mtl_buf(points_world_h)},
                &params, sizeof(params));
        }
    }

    // ─── Kernel 2: Plane fitting ─────────────────────────────────────

    void batch_plane_fit(
        BufferHandle planes_h, BufferHandle neighbors_h,
        int n, int k, float threshold
    ) override {
        @autoreleasepool {
            PlaneFitParams params;
            params.n = (uint32_t)n;
            params.k = (uint32_t)k;
            params.threshold = threshold;

            dispatch_kernel(pso_plane_fit_, n,
                {mtl_buf(neighbors_h), mtl_buf(planes_h)},
                &params, sizeof(params));
        }
    }

    // ─── Kernel 3: Residuals ─────────────────────────────────────────

    void batch_compute_residuals(
        BufferHandle residuals_h, BufferHandle valid_mask_h,
        BufferHandle points_world_h, BufferHandle points_body_h,
        BufferHandle planes_h, int n
    ) override {
        @autoreleasepool {
            ResidualParams params;
            params.n = (uint32_t)n;

            dispatch_kernel(pso_residuals_, n,
                {mtl_buf(points_world_h), mtl_buf(points_body_h),
                 mtl_buf(planes_h), mtl_buf(residuals_h), mtl_buf(valid_mask_h)},
                &params, sizeof(params));
        }
    }

    // ─── Kernel 4: Jacobian (float on GPU, converted to double on read) ─

    void batch_build_jacobian(
        BufferHandle H_h, BufferHandle h_h,
        BufferHandle points_body_h, BufferHandle normals_h,
        BufferHandle plane_dists_h, int m,
        const double R_body_arr[9], const double R_ext_arr[9],
        const double t_ext_arr[3], bool extrinsic_est_en
    ) override {
        @autoreleasepool {
            // H_h is supposed to be M x 12 double, but GPU works in float.
            // We allocate a temp float buffer on GPU, compute there, then convert.
            BufferHandle H_float = alloc(m * 12 * sizeof(float));
            BufferHandle h_float = alloc(m * sizeof(float));

            JacobianParams params;
            for (int i = 0; i < 9; i++) { params.R_body[i] = (float)R_body_arr[i]; params.R_ext[i] = (float)R_ext_arr[i]; }
            for (int i = 0; i < 3; i++) params.t_ext[i] = (float)t_ext_arr[i];
            params.m = (uint32_t)m;
            params.extrinsic_est_en = extrinsic_est_en ? 1 : 0;

            dispatch_kernel(pso_jacobian_, m,
                {mtl_buf(points_body_h), mtl_buf(normals_h), mtl_buf(plane_dists_h),
                 mtl_buf(H_float), mtl_buf(h_float)},
                &params, sizeof(params));

            // Convert float → double on CPU (via shared memory, no copy needed)
            const float* H_f = (const float*)[mtl_buf(H_float) contents];
            const float* h_f = (const float*)[mtl_buf(h_float) contents];
            double* H_d = (double*)[mtl_buf(H_h) contents];
            double* h_d = (double*)[mtl_buf(h_h) contents];

            for (int i = 0; i < m * 12; i++) H_d[i] = (double)H_f[i];
            for (int i = 0; i < m; i++) h_d[i] = (double)h_f[i];

            free(H_float);
            free(h_float);
        }
    }

    // ─── Kernel 5: H^T * H (GPU partial reduction + CPU final sum) ──

    void compute_HTH(double HTH[144], BufferHandle H_h, int m) override {
        @autoreleasepool {
            // We need H as float for GPU reduction. If H_h contains doubles
            // (from batch_build_jacobian), convert to float first.
            BufferHandle H_float = alloc(m * 12 * sizeof(float));
            const double* H_d = (const double*)[mtl_buf(H_h) contents];
            float* H_f = (float*)[mtl_buf(H_float) contents];
            for (int i = 0; i < m * 12; i++) H_f[i] = (float)H_d[i];

            uint32_t group_size = 256;
            uint32_t num_groups = ((uint32_t)m + group_size - 1) / group_size;

            // Partial results: num_groups x 78 floats (upper triangle)
            BufferHandle partials = alloc(num_groups * 78 * sizeof(float));

            HTHParams params;
            params.m = (uint32_t)m;

            dispatch_kernel_groups(pso_hth_partial_, num_groups, group_size,
                {mtl_buf(H_float), mtl_buf(partials)},
                &params, sizeof(params));

            // Final reduction on CPU
            const float* partial_data = (const float*)[mtl_buf(partials) contents];
            float result[78] = {};
            for (uint32_t g = 0; g < num_groups; g++) {
                for (int i = 0; i < 78; i++) {
                    result[i] += partial_data[g * 78 + i];
                }
            }

            // Expand upper triangle to full 12x12 column-major double
            memset(HTH, 0, 144 * sizeof(double));
            int idx = 0;
            for (int i = 0; i < 12; i++) {
                for (int j = i; j < 12; j++) {
                    HTH[j * 12 + i] = (double)result[idx];  // col-major: (i,j) = [j*12+i]
                    HTH[i * 12 + j] = (double)result[idx];  // symmetric
                    idx++;
                }
            }

            free(H_float);
            free(partials);
        }
    }

    // ─── Kernel 6: H^T * h ──────────────────────────────────────────

    void compute_HTh(double HTh[12], BufferHandle H_h, BufferHandle h_h, int m) override {
        @autoreleasepool {
            // Convert to float for GPU
            BufferHandle H_float = alloc(m * 12 * sizeof(float));
            BufferHandle h_float = alloc(m * sizeof(float));

            const double* H_d = (const double*)[mtl_buf(H_h) contents];
            const double* h_d = (const double*)[mtl_buf(h_h) contents];
            float* H_f = (float*)[mtl_buf(H_float) contents];
            float* h_f = (float*)[mtl_buf(h_float) contents];
            for (int i = 0; i < m * 12; i++) H_f[i] = (float)H_d[i];
            for (int i = 0; i < m; i++) h_f[i] = (float)h_d[i];

            uint32_t group_size = 256;
            uint32_t num_groups = ((uint32_t)m + group_size - 1) / group_size;

            BufferHandle partials = alloc(num_groups * 12 * sizeof(float));

            HTHParams params;
            params.m = (uint32_t)m;

            dispatch_kernel_groups(pso_hth_vec_, num_groups, group_size,
                {mtl_buf(H_float), mtl_buf(h_float), mtl_buf(partials)},
                &params, sizeof(params));

            // Final reduction
            const float* partial_data = (const float*)[mtl_buf(partials) contents];
            for (int i = 0; i < 12; i++) {
                double sum = 0;
                for (uint32_t g = 0; g < num_groups; g++) {
                    sum += (double)partial_data[g * 12 + i];
                }
                HTh[i] = sum;
            }

            free(H_float);
            free(h_float);
            free(partials);
        }
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
        @autoreleasepool {
            // Segments use double on CPU but float on GPU — convert
            BufferHandle seg_R_f = alloc(num_segments * 9 * sizeof(float));
            BufferHandle seg_vel_f = alloc(num_segments * 3 * sizeof(float));
            BufferHandle seg_pos_f = alloc(num_segments * 3 * sizeof(float));
            BufferHandle seg_acc_f = alloc(num_segments * 3 * sizeof(float));
            BufferHandle seg_angvel_f = alloc(num_segments * 3 * sizeof(float));
            BufferHandle seg_t_f = alloc(num_segments * sizeof(float));

            auto convert = [&](BufferHandle src, BufferHandle dst, int count) {
                const double* s = (const double*)[mtl_buf(src) contents];
                float* d = (float*)[mtl_buf(dst) contents];
                for (int i = 0; i < count; i++) d[i] = (float)s[i];
            };

            convert(seg_R_h, seg_R_f, num_segments * 9);
            convert(seg_vel_h, seg_vel_f, num_segments * 3);
            convert(seg_pos_h, seg_pos_f, num_segments * 3);
            convert(seg_acc_h, seg_acc_f, num_segments * 3);
            convert(seg_angvel_h, seg_angvel_f, num_segments * 3);
            convert(seg_t_start_h, seg_t_f, num_segments);

            UndistortParams params;
            rt_to_float(imu_end_state, params.R_end, params.t_end);
            rt_to_float(lidar_to_imu, params.R_ext, params.t_ext);
            params.n = (uint32_t)n;
            params.num_segments = (uint32_t)num_segments;

            dispatch_kernel(pso_undistort_, n,
                {mtl_buf(points_h), mtl_buf(timestamps_h),
                 mtl_buf(seg_R_f), mtl_buf(seg_vel_f), mtl_buf(seg_pos_f),
                 mtl_buf(seg_acc_f), mtl_buf(seg_angvel_f), mtl_buf(seg_t_f)},
                &params, sizeof(params));

            free(seg_R_f); free(seg_vel_f); free(seg_pos_f);
            free(seg_acc_f); free(seg_angvel_f); free(seg_t_f);
        }
    }

    // ─── Fused pipeline (superkernel + combined reduction) ─────────

    HShareModelResult fused_h_share_model(
        const float* points_body_host, const float* neighbors_host,
        int n, int k,
        const RigidTransform& body_to_world, const RigidTransform& lidar_to_imu,
        float plane_threshold, bool extrinsic_est_en
    ) override {
        @autoreleasepool {
            // Allocate GPU buffers
            BufferHandle b_pb = alloc(n * 3 * sizeof(float));
            BufferHandle b_nb = alloc((size_t)n * k * 3 * sizeof(float));
            BufferHandle b_H  = alloc(n * 12 * sizeof(float));  // N x 12 float (invalid → zero)
            BufferHandle b_h  = alloc(n * sizeof(float));        // N float

            upload(b_pb, points_body_host, n * 3 * sizeof(float));
            upload(b_nb, neighbors_host, (size_t)n * k * 3 * sizeof(float));

            // ── Single fused dispatch: transform + plane_fit + residual + jacobian ──
            FusedParams fparams;
            rt_to_float(body_to_world, fparams.R_body, fparams.t_body);
            rt_to_float(lidar_to_imu, fparams.R_ext, fparams.t_ext);
            fparams.n = (uint32_t)n;
            fparams.k = (uint32_t)k;
            fparams.plane_threshold = plane_threshold;
            fparams.extrinsic_est_en = extrinsic_est_en ? 1 : 0;

            dispatch_kernel(pso_fused_, n,
                {mtl_buf(b_pb), mtl_buf(b_nb), mtl_buf(b_H), mtl_buf(b_h)},
                &fparams, sizeof(fparams));

            // ── Combined HTH + HTh reduction ──
            uint32_t group_size = 256;
            uint32_t num_groups = ((uint32_t)n + group_size - 1) / group_size;

            BufferHandle b_partials_hth = alloc(num_groups * 78 * sizeof(float));
            BufferHandle b_partials_hth_vec = alloc(num_groups * 12 * sizeof(float));

            HTHParams hparams;
            hparams.m = (uint32_t)n;

            dispatch_kernel_groups(pso_hth_combined_, num_groups, group_size,
                {mtl_buf(b_H), mtl_buf(b_h),
                 mtl_buf(b_partials_hth), mtl_buf(b_partials_hth_vec)},
                &hparams, sizeof(hparams));

            // ── CPU final reduction ──
            HShareModelResult result;

            // HTH
            const float* phth = (const float*)[mtl_buf(b_partials_hth) contents];
            float hth_sum[78] = {};
            for (uint32_t g = 0; g < num_groups; g++)
                for (int i = 0; i < 78; i++)
                    hth_sum[i] += phth[g * 78 + i];

            memset(result.HTH, 0, sizeof(result.HTH));
            int idx = 0;
            for (int i = 0; i < 12; i++)
                for (int j = i; j < 12; j++) {
                    result.HTH[j*12+i] = (double)hth_sum[idx];
                    result.HTH[i*12+j] = (double)hth_sum[idx];
                    idx++;
                }

            // HTh
            const float* phthv = (const float*)[mtl_buf(b_partials_hth_vec) contents];
            for (int i = 0; i < 12; i++) {
                double sum = 0;
                for (uint32_t g = 0; g < num_groups; g++)
                    sum += (double)phthv[g * 12 + i];
                result.HTh[i] = sum;
            }

            // ── Count valid features + extract compacted results ──
            // Read back h_out: non-zero h means valid point
            const float* h_data = (const float*)[mtl_buf(b_h) contents];
            const float* H_data = (const float*)[mtl_buf(b_H) contents];

            result.effct_feat_num = 0;
            std::vector<float> compact_pb, compact_normals, compact_dists;

            for (int i = 0; i < n; i++) {
                // Check if this point produced a non-zero H row
                // (valid points have non-zero normal in cols 0-2)
                const float* row = H_data + i * 12;
                if (row[0] != 0.0f || row[1] != 0.0f || row[2] != 0.0f) {
                    compact_pb.push_back(points_body_host[i*3+0]);
                    compact_pb.push_back(points_body_host[i*3+1]);
                    compact_pb.push_back(points_body_host[i*3+2]);
                    compact_normals.push_back(row[0]);
                    compact_normals.push_back(row[1]);
                    compact_normals.push_back(row[2]);
                    compact_dists.push_back(-h_data[i]);  // h = -residual
                    result.effct_feat_num++;
                }
            }

            result.valid_points_body = std::move(compact_pb);
            result.valid_normals = std::move(compact_normals);
            result.valid_residuals = std::move(compact_dists);

            free(b_pb); free(b_nb); free(b_H); free(b_h);
            free(b_partials_hth); free(b_partials_hth_vec);

            return result;
        }
    }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;

    id<MTLComputePipelineState> pso_transform_ = nil;
    id<MTLComputePipelineState> pso_plane_fit_ = nil;
    id<MTLComputePipelineState> pso_residuals_ = nil;
    id<MTLComputePipelineState> pso_jacobian_ = nil;
    id<MTLComputePipelineState> pso_hth_partial_ = nil;
    id<MTLComputePipelineState> pso_hth_vec_ = nil;
    id<MTLComputePipelineState> pso_undistort_ = nil;
    id<MTLComputePipelineState> pso_fused_ = nil;
    id<MTLComputePipelineState> pso_hth_combined_ = nil;

    struct BufferInfo {
        id<MTLBuffer> buffer;
        size_t size;
    };
    std::unordered_map<BufferHandle, BufferInfo> buffers_;
    BufferHandle next_handle_ = 1;

    // ─── Helpers ─────────────────────────────────────────────────────

    id<MTLBuffer> mtl_buf(BufferHandle h) {
        auto it = buffers_.find(h);
        return (it != buffers_.end()) ? it->second.buffer : nil;
    }

    id<MTLComputePipelineState> make_pso(NSString* name) {
        NSError* error = nil;
        id<MTLFunction> fn = [library_ newFunctionWithName:name];
        if (!fn) {
            NSLog(@"Metal: function '%@' not found in library", name);
            return nil;
        }
        id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            NSLog(@"Metal: failed to create PSO for '%@': %@", name, error);
        }
        return pso;
    }

    NSString* find_metallib() {
        // Try several paths:
        // 1. Next to the executable
        NSString* execPath = [[NSProcessInfo processInfo] arguments][0];
        NSString* execDir = [execPath stringByDeletingLastPathComponent];
        NSString* path = [execDir stringByAppendingPathComponent:@"kernels.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) return path;

        // 2. In the source tree (development mode)
        // Walk up from executable looking for include/compute/metal/
        NSString* srcPath = @__FILE__;
        NSString* srcDir = [srcPath stringByDeletingLastPathComponent];
        path = [srcDir stringByAppendingPathComponent:@"metal/kernels.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) return path;

        // 3. Current working directory
        path = @"kernels.metallib";
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) return path;

        return nil;
    }

    // Dispatch a simple 1D compute kernel with N threads
    void dispatch_kernel(id<MTLComputePipelineState> pso, int n,
                         std::initializer_list<id<MTLBuffer>> buffers,
                         const void* params, size_t params_size) {
        id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];

        int idx = 0;
        for (auto buf : buffers) {
            [enc setBuffer:buf offset:0 atIndex:idx++];
        }
        [enc setBytes:params length:params_size atIndex:idx];

        NSUInteger threadWidth = pso.maxTotalThreadsPerThreadgroup;
        if (threadWidth > 256) threadWidth = 256;
        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadWidth, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }

    // Dispatch with explicit threadgroup count and size
    void dispatch_kernel_groups(id<MTLComputePipelineState> pso,
                                uint32_t num_groups, uint32_t group_size,
                                std::initializer_list<id<MTLBuffer>> buffers,
                                const void* params, size_t params_size) {
        id<MTLCommandBuffer> cmdBuf = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];

        int idx = 0;
        for (auto buf : buffers) {
            [enc setBuffer:buf offset:0 atIndex:idx++];
        }
        [enc setBytes:params length:params_size atIndex:idx];

        MTLSize gridSize = MTLSizeMake(num_groups * group_size, 1, 1);
        MTLSize groupSizeMTL = MTLSizeMake(group_size, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSizeMTL];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Factory function updates (override the inline ones from cpu_backend.h)
// ═══════════════════════════════════════════════════════════════════════

std::unique_ptr<ComputeBackend> create_backend(const std::string& name) {
    if (name == "metal" || name == "Metal") {
        auto backend = std::make_unique<MetalBackend>();
        if (backend->is_available()) return backend;
        return nullptr;
    }
    if (name == "cpu" || name == "CPU") {
        return std::make_unique<CPUBackend>();
    }
    return nullptr;
}

std::unique_ptr<ComputeBackend> create_default_backend() {
    // Try Metal first on macOS
    auto metal = std::make_unique<MetalBackend>();
    if (metal->is_available()) return metal;

    // Fall back to CPU
    return std::make_unique<CPUBackend>();
}

} // namespace compute
} // namespace fastlio
