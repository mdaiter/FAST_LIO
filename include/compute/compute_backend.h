#ifndef FAST_LIO_COMPUTE_BACKEND_H
#define FAST_LIO_COMPUTE_BACKEND_H

/**
 * Abstract Compute Backend for FAST-LIO GPU Acceleration
 * ======================================================
 *
 * This interface captures all GPU-amenable operations from the FAST-LIO pipeline:
 *
 * 1. h_share_model() inner loop (laserMapping.cpp:650-692):
 *    - Batch point transformation: p_world = R * (R_ext * p_body + t_ext) + pos
 *    - Batch k-NN search against ikd-tree (deferred — tree structure is complex)
 *    - Batch plane fitting: 5x3 QR per point → plane coefficients
 *    - Batch residual + validity filtering
 *
 * 2. Jacobian construction (laserMapping.cpp:723-752):
 *    - Per-point: skew(R_ext * p_body) * (R^T * normal), etc → H row
 *
 * 3. ESKF update (esekfom.hpp:1784-1809):
 *    - H^T * H computation (Nx12 → 12x12)
 *    - Matrix operations stay on CPU (23x23 is too small for GPU)
 *
 * 4. IMU undistortion (IMU_Processing.hpp:307-345):
 *    - Batch point compensation: R_comp * p + t_comp per point
 *
 * Design principles:
 *   - Batch-first: every operation takes arrays, not single elements
 *   - Data stays on device between operations (minimize transfers)
 *   - CPU backend is the reference implementation (validates correctness)
 *   - Interface is Rust FFI-friendly (flat buffers, no templates in API)
 */

#include <Eigen/Eigen>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace fastlio {
namespace compute {

// ─── Device buffer handle ────────────────────────────────────────────
// Opaque handle for data residing on the compute device (GPU/CPU).
// Each backend interprets the handle differently.

using BufferHandle = uint64_t;
constexpr BufferHandle INVALID_BUFFER = 0;

// ─── Point type for the interface ────────────────────────────────────
// We use flat float arrays at the boundary (no PCL/Eigen dependency in the
// actual GPU kernels). Points are [x, y, z] packed contiguously.

struct Point3f {
    float x, y, z;
};

// ─── Plane result ────────────────────────────────────────────────────
struct PlaneCoeffs {
    float a, b, c, d;   // ax + by + cz + d = 0, with (a,b,c) unit normal
    bool  valid;         // false if plane fit was rejected
};

// ─── Jacobian row (12 doubles, matching FAST-LIO's h_x layout) ──────
// Layout: [norm_x, norm_y, norm_z, A0, A1, A2, B0, B1, B2, C0, C1, C2]
//   where A = skew(R_ext*p) * R^T*n,  B = skew(p_body) * R_ext^T * R^T * n,  C = R^T * n

// ─── Rigid transform (rotation + translation) ───────────────────────
struct RigidTransform {
    double R[9];   // 3x3 rotation, column-major (Eigen default)
    double t[3];   // translation
};

// ─── The abstract compute backend ────────────────────────────────────

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    /// Human-readable backend name (e.g., "CPU", "CUDA", "Metal")
    virtual std::string name() const = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Buffer management
    // ═══════════════════════════════════════════════════════════════════

    /// Allocate a device buffer of `size_bytes`. Returns INVALID_BUFFER on failure.
    virtual BufferHandle alloc(size_t size_bytes) = 0;

    /// Free a device buffer.
    virtual void free(BufferHandle buf) = 0;

    /// Upload from host to device. Returns false on failure.
    virtual bool upload(BufferHandle dst, const void* src, size_t size_bytes) = 0;

    /// Download from device to host. Returns false on failure.
    virtual bool download(void* dst, BufferHandle src, size_t size_bytes) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Kernel 1: Batch point transformation
    //   p_world[i] = R_body * (R_ext * p_body[i] + t_ext) + t_body
    //
    //   This is the two-stage extrinsic + body transform from h_share_model.
    //   Inputs:
    //     points_body: N x 3 floats (device buffer)
    //     R_body, t_body: body-to-world rotation/translation
    //     R_ext, t_ext:   LiDAR-to-IMU extrinsic
    //   Output:
    //     points_world: N x 3 floats (device buffer, pre-allocated)
    // ═══════════════════════════════════════════════════════════════════

    virtual void batch_transform_points(
        BufferHandle points_world,       // output: N x 3 float
        BufferHandle points_body,        // input:  N x 3 float
        int n,
        const RigidTransform& body_to_world,
        const RigidTransform& lidar_to_imu
    ) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Kernel 2: Batch plane fitting
    //   For each of N points, given its NUM_MATCH_POINTS (5) nearest
    //   neighbors, fit a plane via least-squares and check residuals.
    //
    //   Inputs:
    //     neighbors: N * k * 3 floats — k neighbors per query point,
    //                packed [pt0_nb0_x, pt0_nb0_y, pt0_nb0_z,
    //                        pt0_nb1_x, ..., pt1_nb0_x, ...]
    //     k:         number of neighbors per point (typically 5)
    //     threshold: max per-point residual for valid plane
    //   Output:
    //     planes: N PlaneCoeffs (device buffer)
    // ═══════════════════════════════════════════════════════════════════

    virtual void batch_plane_fit(
        BufferHandle planes,             // output: N x PlaneCoeffs
        BufferHandle neighbors,          // input:  N * k * 3 float
        int n,
        int k,
        float threshold
    ) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Kernel 3: Batch residual computation + validity filter
    //   For each point with a valid plane, compute:
    //     residual = a*wx + b*wy + c*wz + d
    //     score    = 1 - 0.9 * |residual| / sqrt(||p_body||)
    //     valid    = plane.valid && score > 0.9
    //
    //   Inputs:
    //     points_world: N x 3 float (from kernel 1)
    //     points_body:  N x 3 float (original body points)
    //     planes:       N x PlaneCoeffs (from kernel 2)
    //   Output:
    //     residuals:  N float
    //     valid_mask: N uint8_t (1 = valid, 0 = invalid)
    // ═══════════════════════════════════════════════════════════════════

    virtual void batch_compute_residuals(
        BufferHandle residuals,          // output: N float
        BufferHandle valid_mask,         // output: N uint8_t
        BufferHandle points_world,       // input:  N x 3 float
        BufferHandle points_body,        // input:  N x 3 float
        BufferHandle planes,             // input:  N x PlaneCoeffs
        int n
    ) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Kernel 4: Batch Jacobian construction
    //   For each valid point, compute one row of the measurement Jacobian H.
    //
    //   H_i = [norm^T, A^T, B^T, C^T]  (1x12)
    //   where:
    //     point_imu  = R_ext * p_body + t_ext
    //     C          = R_body^T * normal
    //     A          = skew(point_imu) * C
    //     B          = skew(p_body) * R_ext^T * C   (if extrinsic_est_en)
    //
    //   Inputs:
    //     points_body:  M x 3 float  (only valid points, compacted)
    //     normals:      M x 3 float  (plane normals for valid points)
    //     R_body:       3x3 rotation body-to-world
    //     R_ext:        3x3 LiDAR-to-IMU rotation
    //     t_ext:        3x1 LiDAR-to-IMU translation
    //     extrinsic_est_en: whether to fill columns 6-11
    //   Output:
    //     H:            M x 12 double (device buffer)
    //     h:            M double      (measurement residuals = -plane_dist)
    // ═══════════════════════════════════════════════════════════════════

    virtual void batch_build_jacobian(
        BufferHandle H,                  // output: M x 12 double
        BufferHandle h,                  // output: M double
        BufferHandle points_body,        // input:  M x 3 float
        BufferHandle normals,            // input:  M x 3 float
        BufferHandle plane_dists,        // input:  M float (signed distance)
        int m,
        const double R_body[9],          // 3x3 column-major
        const double R_ext[9],           // 3x3 column-major
        const double t_ext[3],           // 3x1
        bool extrinsic_est_en
    ) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Kernel 5: H^T * H computation
    //   Compute the 12x12 symmetric matrix H^T * H from the Mx12 Jacobian.
    //   This is a parallel reduction (the only non-embarrassingly-parallel op).
    //
    //   Input:
    //     H:   M x 12 double (device buffer)
    //     m:   number of rows
    //   Output:
    //     HTH: 12x12 double (host memory — small enough to return directly)
    // ═══════════════════════════════════════════════════════════════════

    virtual void compute_HTH(
        double HTH[144],                 // output: 12x12 column-major (host)
        BufferHandle H,                  // input:  M x 12 double
        int m
    ) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Kernel 6: H^T * h computation  
    //   Compute the 12x1 vector H^T * h (needed for Kalman gain path
    //   where n > dof_Measurement, i.e., K_h = P_inv * H^T * h).
    //
    //   Input:
    //     H: M x 12 double (device buffer)
    //     h: M double (device buffer)
    //     m: number of measurements
    //   Output:
    //     HTh: 12 doubles (host memory)
    // ═══════════════════════════════════════════════════════════════════

    virtual void compute_HTh(
        double HTh[12],                  // output: 12x1 (host)
        BufferHandle H,                  // input:  M x 12 double
        BufferHandle h,                  // input:  M double
        int m
    ) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Kernel 7: Batch point undistortion (IMU motion compensation)
    //   For each point, given per-point rotation + translation:
    //     P_comp = R_comp_inv * (R_i * (R_ext * p + t_ext) + T_ei) - t_comp
    //
    //   This is a simplified interface — the caller pre-computes per-segment
    //   R_i and T_ei on the CPU (since there are only ~10 IMU segments per
    //   scan) and provides indices mapping points to segments.
    //
    //   Inputs:
    //     points:       N x 3 float (LiDAR frame, in-place update)
    //     timestamps:   N float (per-point timestamp offsets)
    //     seg_R:        S x 9 double (per-segment start rotation)
    //     seg_vel:      S x 3 double (per-segment velocity)
    //     seg_pos:      S x 3 double (per-segment position)
    //     seg_acc:      S x 3 double (per-segment acceleration)
    //     seg_angvel:   S x 3 double (per-segment angular velocity)
    //     seg_t_start:  S double (segment start timestamps)
    //     num_segments: S
    //     R_end, t_end: end-of-scan IMU state
    //     R_ext, t_ext: LiDAR-to-IMU extrinsic
    //   Output:
    //     points: N x 3 float (updated in-place, compensated)
    // ═══════════════════════════════════════════════════════════════════

    virtual void batch_undistort_points(
        BufferHandle points,             // in/out: N x 3 float
        BufferHandle timestamps,         // input:  N float
        BufferHandle seg_R,              // input:  S x 9 double
        BufferHandle seg_vel,            // input:  S x 3 double
        BufferHandle seg_pos,            // input:  S x 3 double
        BufferHandle seg_acc,            // input:  S x 3 double
        BufferHandle seg_angvel,         // input:  S x 3 double
        BufferHandle seg_t_start,        // input:  S double
        int n,
        int num_segments,
        const RigidTransform& imu_end_state,
        const RigidTransform& lidar_to_imu
    ) = 0;

    // ═══════════════════════════════════════════════════════════════════
    // Convenience: Fused h_share_model pipeline
    //   Runs kernels 1 + 2 + 3 + 4 + 5 + 6 in sequence, reusing device
    //   buffers. This avoids round-tripping data to host between steps.
    //
    //   Returns effective feature count (number of valid points).
    // ═══════════════════════════════════════════════════════════════════

    struct HShareModelResult {
        int    effct_feat_num;           // number of valid features
        double HTH[144];                 // 12x12 column-major
        double HTh[12];                  // 12x1

        // Valid points + normals + residuals for the ESKF (host memory).
        // These are compacted (only valid features).
        std::vector<float>  valid_points_body;   // effct_feat_num x 3
        std::vector<float>  valid_normals;       // effct_feat_num x 3
        std::vector<float>  valid_residuals;     // effct_feat_num
    };

    /// Fused pipeline: transform → plane fit → residual → compact → jacobian → HTH/HTh.
    /// `neighbors` must already contain k-NN results (k*3 floats per point).
    /// This is the main acceleration target.
    virtual HShareModelResult fused_h_share_model(
        const float* points_body,        // N x 3 float (host)
        const float* neighbors,          // N * k * 3 float (host, from ikd-tree)
        int n,
        int k,
        const RigidTransform& body_to_world,
        const RigidTransform& lidar_to_imu,
        float plane_threshold,
        bool  extrinsic_est_en
    ) = 0;
};

// ─── Backend factory ─────────────────────────────────────────────────

/// Create a compute backend by name. Returns nullptr if not available.
///   "cpu"   — always available (reference implementation)
///   "metal" — available on macOS with Metal-capable GPU
///   "cuda"  — available on systems with NVIDIA GPU + CUDA toolkit
std::unique_ptr<ComputeBackend> create_backend(const std::string& name);

/// Create the best available backend for this system.
std::unique_ptr<ComputeBackend> create_default_backend();

} // namespace compute
} // namespace fastlio

#endif // FAST_LIO_COMPUTE_BACKEND_H
