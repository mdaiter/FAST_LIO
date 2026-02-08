#ifndef FAST_LIO_CPU_BACKEND_H
#define FAST_LIO_CPU_BACKEND_H

/**
 * CPU Reference Backend for FAST-LIO Compute Abstraction
 * ======================================================
 * 
 * This implements all ComputeBackend operations using Eigen on the CPU.
 * It serves as:
 *   1. The correctness reference for GPU backends
 *   2. A fallback when no GPU is available
 *   3. Baseline for performance comparisons
 *
 * "Device buffers" are just heap-allocated host memory with a handle map.
 */

#include "compute_backend.h"

#include <Eigen/Eigen>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace fastlio {
namespace compute {

class CPUBackend : public ComputeBackend {
public:
    CPUBackend() : next_handle_(1) {}
    ~CPUBackend() override {
        for (auto& [handle, info] : buffers_) {
            ::free(info.data);
        }
    }

    std::string name() const override { return "CPU"; }

    // ─── Buffer management ───────────────────────────────────────────

    BufferHandle alloc(size_t size_bytes) override {
        void* ptr = std::malloc(size_bytes);
        if (!ptr) return INVALID_BUFFER;
        BufferHandle h = next_handle_++;
        buffers_[h] = {ptr, size_bytes};
        return h;
    }

    void free(BufferHandle buf) override {
        auto it = buffers_.find(buf);
        if (it != buffers_.end()) {
            ::free(it->second.data);
            buffers_.erase(it);
        }
    }

    bool upload(BufferHandle dst, const void* src, size_t size_bytes) override {
        auto it = buffers_.find(dst);
        if (it == buffers_.end() || it->second.size < size_bytes) return false;
        std::memcpy(it->second.data, src, size_bytes);
        return true;
    }

    bool download(void* dst, BufferHandle src, size_t size_bytes) override {
        auto it = buffers_.find(src);
        if (it == buffers_.end() || it->second.size < size_bytes) return false;
        std::memcpy(dst, it->second.data, size_bytes);
        return true;
    }

    // ─── Kernel 1: Batch point transformation ────────────────────────

    void batch_transform_points(
        BufferHandle points_world_h,
        BufferHandle points_body_h,
        int n,
        const RigidTransform& body_to_world,
        const RigidTransform& lidar_to_imu
    ) override {
        float* out = ptr<float>(points_world_h);
        const float* in = ptr<float>(points_body_h);

        // Build Eigen matrices from flat arrays
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_body(body_to_world.R);
        Eigen::Map<const Eigen::Vector3d> t_body(body_to_world.t);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_ext(lidar_to_imu.R);
        Eigen::Map<const Eigen::Vector3d> t_ext(lidar_to_imu.t);

        for (int i = 0; i < n; i++) {
            Eigen::Vector3d p_body(in[i*3+0], in[i*3+1], in[i*3+2]);
            Eigen::Vector3d p_world = R_body * (R_ext * p_body + t_ext) + t_body;
            out[i*3+0] = static_cast<float>(p_world.x());
            out[i*3+1] = static_cast<float>(p_world.y());
            out[i*3+2] = static_cast<float>(p_world.z());
        }
    }

    // ─── Kernel 2: Batch plane fitting ───────────────────────────────

    void batch_plane_fit(
        BufferHandle planes_h,
        BufferHandle neighbors_h,
        int n,
        int k,
        float threshold
    ) override {
        PlaneCoeffs* planes = ptr<PlaneCoeffs>(planes_h);
        const float* nb = ptr<float>(neighbors_h);

        for (int i = 0; i < n; i++) {
            const float* pts = nb + i * k * 3;

            // Solve: A * x = b where A = [x,y,z], b = [-1,...,-1]
            // x = [A/D, B/D, C/D] of plane Ax+By+Cz+D=0
            Eigen::Matrix<float, Eigen::Dynamic, 3> A(k, 3);
            Eigen::Matrix<float, Eigen::Dynamic, 1> b(k, 1);
            b.setConstant(-1.0f);

            for (int j = 0; j < k; j++) {
                A(j, 0) = pts[j*3+0];
                A(j, 1) = pts[j*3+1];
                A(j, 2) = pts[j*3+2];
            }

            Eigen::Vector3f normvec = A.colPivHouseholderQr().solve(b);
            float norm = normvec.norm();

            if (norm < 1e-10f) {
                planes[i] = {0, 0, 0, 0, false};
                continue;
            }

            planes[i].a = normvec(0) / norm;
            planes[i].b = normvec(1) / norm;
            planes[i].c = normvec(2) / norm;
            planes[i].d = 1.0f / norm;

            // Check residuals
            bool valid = true;
            for (int j = 0; j < k; j++) {
                float residual = planes[i].a * pts[j*3+0]
                               + planes[i].b * pts[j*3+1]
                               + planes[i].c * pts[j*3+2]
                               + planes[i].d;
                if (std::fabs(residual) > threshold) {
                    valid = false;
                    break;
                }
            }
            planes[i].valid = valid;
        }
    }

    // ─── Kernel 3: Batch residual computation ────────────────────────

    void batch_compute_residuals(
        BufferHandle residuals_h,
        BufferHandle valid_mask_h,
        BufferHandle points_world_h,
        BufferHandle points_body_h,
        BufferHandle planes_h,
        int n
    ) override {
        float* residuals = ptr<float>(residuals_h);
        uint8_t* valid = ptr<uint8_t>(valid_mask_h);
        const float* pw = ptr<float>(points_world_h);
        const float* pb = ptr<float>(points_body_h);
        const PlaneCoeffs* planes = ptr<PlaneCoeffs>(planes_h);

        for (int i = 0; i < n; i++) {
            if (!planes[i].valid) {
                residuals[i] = 0.0f;
                valid[i] = 0;
                continue;
            }

            float pd2 = planes[i].a * pw[i*3+0]
                       + planes[i].b * pw[i*3+1]
                       + planes[i].c * pw[i*3+2]
                       + planes[i].d;

            float body_norm = std::sqrt(pb[i*3+0]*pb[i*3+0]
                                      + pb[i*3+1]*pb[i*3+1]
                                      + pb[i*3+2]*pb[i*3+2]);

            float s = 1.0f - 0.9f * std::fabs(pd2) / std::sqrt(body_norm);

            if (s > 0.9f) {
                residuals[i] = pd2;
                valid[i] = 1;
            } else {
                residuals[i] = 0.0f;
                valid[i] = 0;
            }
        }
    }

    // ─── Kernel 4: Batch Jacobian construction ───────────────────────

    void batch_build_jacobian(
        BufferHandle H_h,
        BufferHandle h_h,
        BufferHandle points_body_h,
        BufferHandle normals_h,
        BufferHandle plane_dists_h,
        int m,
        const double R_body_arr[9],
        const double R_ext_arr[9],
        const double t_ext_arr[3],
        bool extrinsic_est_en
    ) override {
        double* H = ptr<double>(H_h);
        double* h = ptr<double>(h_h);
        const float* pb = ptr<float>(points_body_h);
        const float* normals = ptr<float>(normals_h);
        const float* dists = ptr<float>(plane_dists_h);

        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_body(R_body_arr);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_ext(R_ext_arr);
        Eigen::Map<const Eigen::Vector3d> t_ext(t_ext_arr);

        Eigen::Matrix3d R_body_T = R_body.transpose();

        for (int i = 0; i < m; i++) {
            Eigen::Vector3d p_body(pb[i*3+0], pb[i*3+1], pb[i*3+2]);
            Eigen::Vector3d norm_vec(normals[i*3+0], normals[i*3+1], normals[i*3+2]);

            // point_imu = R_ext * p_body + t_ext
            Eigen::Vector3d point_imu = R_ext * p_body + t_ext;

            // C = R_body^T * normal
            Eigen::Vector3d C = R_body_T * norm_vec;

            // A = skew(point_imu) * C
            Eigen::Matrix3d point_crossmat;
            point_crossmat << 0, -point_imu.z(), point_imu.y(),
                              point_imu.z(), 0, -point_imu.x(),
                              -point_imu.y(), point_imu.x(), 0;
            Eigen::Vector3d A = point_crossmat * C;

            // H row: [norm, A, B, C]
            double* row = H + i * 12;
            row[0] = norm_vec.x();
            row[1] = norm_vec.y();
            row[2] = norm_vec.z();
            row[3] = A.x();
            row[4] = A.y();
            row[5] = A.z();

            if (extrinsic_est_en) {
                // B = skew(p_body) * R_ext^T * C
                Eigen::Matrix3d pb_crossmat;
                pb_crossmat << 0, -p_body.z(), p_body.y(),
                               p_body.z(), 0, -p_body.x(),
                               -p_body.y(), p_body.x(), 0;
                Eigen::Vector3d B = pb_crossmat * R_ext.transpose() * C;
                row[6]  = B.x();
                row[7]  = B.y();
                row[8]  = B.z();
                row[9]  = C.x();
                row[10] = C.y();
                row[11] = C.z();
            } else {
                row[6] = row[7] = row[8] = 0.0;
                row[9] = row[10] = row[11] = 0.0;
            }

            // h = -signed_distance
            h[i] = -static_cast<double>(dists[i]);
        }
    }

    // ─── Kernel 5: H^T * H ──────────────────────────────────────────

    void compute_HTH(
        double HTH[144],
        BufferHandle H_h,
        int m
    ) override {
        const double* H = ptr<double>(H_h);

        // Map as Eigen dynamic matrix
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor>> H_mat(H, m, 12);
        Eigen::Matrix<double, 12, 12> result = H_mat.transpose() * H_mat;

        // Copy to output (column-major)
        Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> out(HTH);
        out = result;
    }

    // ─── Kernel 6: H^T * h ──────────────────────────────────────────

    void compute_HTh(
        double HTh[12],
        BufferHandle H_h,
        BufferHandle h_h,
        int m
    ) override {
        const double* H = ptr<double>(H_h);
        const double* h = ptr<double>(h_h);

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor>> H_mat(H, m, 12);
        Eigen::Map<const Eigen::VectorXd> h_vec(h, m);
        Eigen::Matrix<double, 12, 1> result = H_mat.transpose() * h_vec;

        Eigen::Map<Eigen::Matrix<double, 12, 1>> out(HTh);
        out = result;
    }

    // ─── Kernel 7: Batch undistortion ────────────────────────────────

    void batch_undistort_points(
        BufferHandle points_h,
        BufferHandle timestamps_h,
        BufferHandle seg_R_h,
        BufferHandle seg_vel_h,
        BufferHandle seg_pos_h,
        BufferHandle seg_acc_h,
        BufferHandle seg_angvel_h,
        BufferHandle seg_t_start_h,
        int n,
        int num_segments,
        const RigidTransform& imu_end_state,
        const RigidTransform& lidar_to_imu
    ) override {
        float* points = ptr<float>(points_h);
        const float* timestamps = ptr<float>(timestamps_h);
        const double* seg_R = ptr<double>(seg_R_h);
        const double* seg_vel = ptr<double>(seg_vel_h);
        const double* seg_pos = ptr<double>(seg_pos_h);
        const double* seg_acc = ptr<double>(seg_acc_h);
        const double* seg_angvel = ptr<double>(seg_angvel_h);
        const double* seg_t_start = ptr<double>(seg_t_start_h);

        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_end(imu_end_state.R);
        Eigen::Map<const Eigen::Vector3d> pos_end(imu_end_state.t);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_ext(lidar_to_imu.R);
        Eigen::Map<const Eigen::Vector3d> t_ext(lidar_to_imu.t);

        Eigen::Matrix3d R_end_T = R_end.transpose();
        Eigen::Matrix3d R_ext_T = R_ext.transpose();

        for (int i = 0; i < n; i++) {
            float t = timestamps[i];

            // Find the segment for this timestamp (binary search)
            int seg = num_segments - 1;
            for (int s = num_segments - 1; s >= 0; s--) {
                if (t >= seg_t_start[s]) {
                    seg = s;
                    break;
                }
            }

            double dt = t - seg_t_start[seg];

            // R_i = seg_R * Exp(angvel * dt)
            Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> R_seg(seg_R + seg * 9);
            Eigen::Map<const Eigen::Vector3d> vel_seg(seg_vel + seg * 3);
            Eigen::Map<const Eigen::Vector3d> pos_seg(seg_pos + seg * 3);
            Eigen::Map<const Eigen::Vector3d> acc_seg(seg_acc + seg * 3);
            Eigen::Map<const Eigen::Vector3d> angvel_seg(seg_angvel + seg * 3);

            // Rodrigues formula for Exp(angvel * dt)
            Eigen::Vector3d w = angvel_seg * dt;
            double theta = w.norm();
            Eigen::Matrix3d exp_w;
            if (theta < 1e-10) {
                exp_w = Eigen::Matrix3d::Identity();
            } else {
                Eigen::Vector3d axis = w / theta;
                Eigen::Matrix3d K;
                K << 0, -axis.z(), axis.y(),
                     axis.z(), 0, -axis.x(),
                     -axis.y(), axis.x(), 0;
                exp_w = Eigen::Matrix3d::Identity() + std::sin(theta) * K
                        + (1.0 - std::cos(theta)) * K * K;
            }

            Eigen::Matrix3d R_i = R_seg * exp_w;

            Eigen::Vector3d P_i(points[i*3+0], points[i*3+1], points[i*3+2]);

            // T_ei = pos_seg + vel_seg*dt + 0.5*acc_seg*dt^2 - pos_end
            Eigen::Vector3d T_ei = pos_seg + vel_seg * dt + 0.5 * acc_seg * dt * dt - pos_end;

            // P_comp = R_ext^T * (R_end^T * (R_i * (R_ext * P_i + t_ext) + T_ei) - t_ext)
            Eigen::Vector3d P_comp = R_ext_T * (R_end_T * (R_i * (R_ext * P_i.cast<double>() + t_ext) + T_ei) - t_ext);

            points[i*3+0] = static_cast<float>(P_comp.x());
            points[i*3+1] = static_cast<float>(P_comp.y());
            points[i*3+2] = static_cast<float>(P_comp.z());
        }
    }

    // ─── Fused h_share_model pipeline ────────────────────────────────

    HShareModelResult fused_h_share_model(
        const float* points_body_host,
        const float* neighbors_host,
        int n,
        int k,
        const RigidTransform& body_to_world,
        const RigidTransform& lidar_to_imu,
        float plane_threshold,
        bool extrinsic_est_en
    ) override {
        // Allocate device (CPU) buffers
        BufferHandle b_points_body = alloc(n * 3 * sizeof(float));
        BufferHandle b_points_world = alloc(n * 3 * sizeof(float));
        BufferHandle b_neighbors = alloc(n * k * 3 * sizeof(float));
        BufferHandle b_planes = alloc(n * sizeof(PlaneCoeffs));
        BufferHandle b_residuals = alloc(n * sizeof(float));
        BufferHandle b_valid_mask = alloc(n * sizeof(uint8_t));

        // Upload
        upload(b_points_body, points_body_host, n * 3 * sizeof(float));
        upload(b_neighbors, neighbors_host, n * k * 3 * sizeof(float));

        // Step 1: Transform points
        batch_transform_points(b_points_world, b_points_body, n, body_to_world, lidar_to_imu);

        // Step 2: Plane fitting
        batch_plane_fit(b_planes, b_neighbors, n, k, plane_threshold);

        // Step 3: Residual + validity
        batch_compute_residuals(b_residuals, b_valid_mask, b_points_world, b_points_body, b_planes, n);

        // Download validity and results for compaction
        std::vector<uint8_t> valid_mask(n);
        std::vector<float> residuals(n);
        std::vector<PlaneCoeffs> planes(n);
        download(valid_mask.data(), b_valid_mask, n * sizeof(uint8_t));
        download(residuals.data(), b_residuals, n * sizeof(float));
        download(planes.data(), b_planes, n * sizeof(PlaneCoeffs));

        // Compact valid features
        HShareModelResult result;
        result.effct_feat_num = 0;

        std::vector<float> compact_points_body;
        std::vector<float> compact_normals;
        std::vector<float> compact_dists;

        for (int i = 0; i < n; i++) {
            if (valid_mask[i]) {
                compact_points_body.push_back(points_body_host[i*3+0]);
                compact_points_body.push_back(points_body_host[i*3+1]);
                compact_points_body.push_back(points_body_host[i*3+2]);
                compact_normals.push_back(planes[i].a);
                compact_normals.push_back(planes[i].b);
                compact_normals.push_back(planes[i].c);
                compact_dists.push_back(residuals[i]);
                result.effct_feat_num++;
            }
        }

        int m = result.effct_feat_num;

        if (m > 0) {
            // Step 4: Build Jacobian for valid points
            BufferHandle b_H = alloc(m * 12 * sizeof(double));
            BufferHandle b_h = alloc(m * sizeof(double));
            BufferHandle b_compact_pb = alloc(m * 3 * sizeof(float));
            BufferHandle b_compact_n = alloc(m * 3 * sizeof(float));
            BufferHandle b_compact_d = alloc(m * sizeof(float));

            upload(b_compact_pb, compact_points_body.data(), m * 3 * sizeof(float));
            upload(b_compact_n, compact_normals.data(), m * 3 * sizeof(float));
            upload(b_compact_d, compact_dists.data(), m * sizeof(float));

            batch_build_jacobian(b_H, b_h, b_compact_pb, b_compact_n, b_compact_d,
                                 m, body_to_world.R, lidar_to_imu.R, lidar_to_imu.t,
                                 extrinsic_est_en);

            // Step 5 + 6: H^T*H and H^T*h
            compute_HTH(result.HTH, b_H, m);
            compute_HTh(result.HTh, b_H, b_h, m);

            free(b_H);
            free(b_h);
            free(b_compact_pb);
            free(b_compact_n);
            free(b_compact_d);
        } else {
            std::memset(result.HTH, 0, sizeof(result.HTH));
            std::memset(result.HTh, 0, sizeof(result.HTh));
        }

        // Copy compacted results
        result.valid_points_body = std::move(compact_points_body);
        result.valid_normals = std::move(compact_normals);
        result.valid_residuals = std::move(compact_dists);

        // Cleanup
        free(b_points_body);
        free(b_points_world);
        free(b_neighbors);
        free(b_planes);
        free(b_residuals);
        free(b_valid_mask);

        return result;
    }

private:
    struct BufferInfo {
        void* data;
        size_t size;
    };

    template<typename T>
    T* ptr(BufferHandle h) {
        auto it = buffers_.find(h);
        return (it != buffers_.end()) ? static_cast<T*>(it->second.data) : nullptr;
    }

    std::unordered_map<BufferHandle, BufferInfo> buffers_;
    BufferHandle next_handle_;
};

// ─── Factory implementations (CPU-only, when no GPU backend is linked) ───

#if !defined(HAS_METAL) && !defined(HAS_CUDA)
inline std::unique_ptr<ComputeBackend> create_backend(const std::string& name) {
    if (name == "cpu" || name == "CPU") {
        return std::make_unique<CPUBackend>();
    }
    return nullptr;
}

inline std::unique_ptr<ComputeBackend> create_default_backend() {
    return std::make_unique<CPUBackend>();
}
#endif

} // namespace compute
} // namespace fastlio

#endif // FAST_LIO_CPU_BACKEND_H
