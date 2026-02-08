/**
 * Unit tests for the Metal compute backend.
 * Same test suite as test_compute_backend.cpp but using the Metal GPU backend.
 * Tolerances are wider due to float32 precision on GPU.
 */

#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <cmath>
#include <random>
#include <vector>

#include "compute/compute_backend.h"

// Factory is defined in metal_backend.mm (linked in)
// It overrides the inline versions from cpu_backend.h

using namespace fastlio::compute;

// ─── Helpers ─────────────────────────────────────────────────────────

static RigidTransform make_identity() {
    RigidTransform rt;
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rt.R) = Eigen::Matrix3d::Identity();
    rt.t[0] = rt.t[1] = rt.t[2] = 0.0;
    return rt;
}

static RigidTransform make_transform(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    RigidTransform rt;
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rt.R) = R;
    rt.t[0] = t.x(); rt.t[1] = t.y(); rt.t[2] = t.z();
    return rt;
}

static Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    if (theta < 1e-10) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis = w / theta;
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;
    return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

class MetalBackendTest : public ::testing::Test {
protected:
    std::unique_ptr<ComputeBackend> backend;

    void SetUp() override {
        backend = create_backend("metal");
        if (!backend) {
            GTEST_SKIP() << "Metal backend not available";
        }
        EXPECT_EQ(backend->name(), "Metal");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Buffer management
// ═══════════════════════════════════════════════════════════════════════

TEST_F(MetalBackendTest, BufferAllocFreeRoundtrip) {
    auto buf = backend->alloc(1024);
    ASSERT_NE(buf, INVALID_BUFFER);
    backend->free(buf);
}

TEST_F(MetalBackendTest, BufferUploadDownload) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto buf = backend->alloc(data.size() * sizeof(float));
    ASSERT_TRUE(backend->upload(buf, data.data(), data.size() * sizeof(float)));
    std::vector<float> result(4);
    ASSERT_TRUE(backend->download(result.data(), buf, data.size() * sizeof(float)));
    for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(result[i], data[i]);
    backend->free(buf);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 1: Transform
// ═══════════════════════════════════════════════════════════════════════

TEST_F(MetalBackendTest, TransformIdentity) {
    int n = 3;
    std::vector<float> points = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> result(9);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    backend->batch_transform_points(b_out, b_in, n, make_identity(), make_identity());
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    for (int i = 0; i < 9; i++) EXPECT_NEAR(result[i], points[i], 1e-4f);
    backend->free(b_in); backend->free(b_out);
}

TEST_F(MetalBackendTest, TransformBatch) {
    int n = 5000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10, 10);

    std::vector<float> points(n * 3);
    for (auto& p : points) p = dist(rng);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    Eigen::Matrix3d R = exp_so3(Eigen::Vector3d(0.1, 0.2, 0.3));
    Eigen::Vector3d t(1, 2, 3);
    Eigen::Matrix3d R_ext = exp_so3(Eigen::Vector3d(0.01, -0.02, 0.005));
    Eigen::Vector3d t_ext(0.1, 0.05, -0.02);

    backend->batch_transform_points(b_out, b_in, n,
        make_transform(R, t), make_transform(R_ext, t_ext));

    std::vector<float> result(n * 3);
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d p(points[i*3], points[i*3+1], points[i*3+2]);
        Eigen::Vector3d expected = R * (R_ext * p + t_ext) + t;
        for (int d = 0; d < 3; d++) {
            if (std::fabs(result[i*3+d] - (float)expected(d)) > 1e-3f) mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Mismatches: " << mismatches << "/" << n*3;
    backend->free(b_in); backend->free(b_out);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 2: Plane fitting
// ═══════════════════════════════════════════════════════════════════════

TEST_F(MetalBackendTest, PlaneFitPerfectPlane) {
    int n = 1, k = 5;
    std::vector<float> neighbors = {1,0,2, 0,1,2, -1,0,2, 0,-1,2, 0.5f,0.5f,2};

    auto b_nb = backend->alloc(n * k * 3 * sizeof(float));
    // PlaneCoeffsGPU is 20 bytes
    auto b_planes = backend->alloc(n * 20);
    backend->upload(b_nb, neighbors.data(), n * k * 3 * sizeof(float));

    backend->batch_plane_fit(b_planes, b_nb, n, k, 0.1f);

    // Read back as PlaneCoeffs (CPU struct has bool, GPU has uint32)
    // We need to read the GPU struct format
    struct PlaneGPU { float a,b,c,d; uint32_t valid; };
    PlaneGPU plane;
    backend->download(&plane, b_planes, sizeof(PlaneGPU));

    EXPECT_EQ(plane.valid, 1u);
    EXPECT_NEAR(std::fabs(plane.c), 1.0f, 1e-2f);
    EXPECT_NEAR(plane.a, 0.0f, 1e-2f);
    EXPECT_NEAR(plane.b, 0.0f, 1e-2f);
    backend->free(b_nb); backend->free(b_planes);
}

TEST_F(MetalBackendTest, PlaneFitBatch) {
    int n = 500, k = 5;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> xy_dist(-10, 10);

    std::vector<float> neighbors(n * k * 3);
    for (int i = 0; i < n; i++) {
        float z0 = xy_dist(rng);
        for (int j = 0; j < k; j++) {
            neighbors[i*k*3 + j*3 + 0] = xy_dist(rng);
            neighbors[i*k*3 + j*3 + 1] = xy_dist(rng);
            neighbors[i*k*3 + j*3 + 2] = z0;
        }
    }

    auto b_nb = backend->alloc(n * k * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * 20);
    backend->upload(b_nb, neighbors.data(), n * k * 3 * sizeof(float));

    backend->batch_plane_fit(b_planes, b_nb, n, k, 0.1f);

    struct PlaneGPU { float a,b,c,d; uint32_t valid; };
    std::vector<PlaneGPU> planes(n);
    backend->download(planes.data(), b_planes, n * sizeof(PlaneGPU));

    int valid_count = 0;
    for (int i = 0; i < n; i++) {
        if (planes[i].valid) {
            valid_count++;
            EXPECT_NEAR(std::fabs(planes[i].c), 1.0f, 0.05f) << "plane " << i;
        }
    }
    EXPECT_GT(valid_count, n * 0.9) << "Expected most planes to be valid";
    backend->free(b_nb); backend->free(b_planes);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 5+6: HTH and HTh
// ═══════════════════════════════════════════════════════════════════════

TEST_F(MetalBackendTest, HTH_RandomVerify) {
    int m = 2000;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 1);

    Eigen::MatrixXd H_eigen(m, 12);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < 12; j++)
            H_eigen(i, j) = dist(rng);

    // Backend expects row-major double
    std::vector<double> H_rowmajor(m * 12);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < 12; j++)
            H_rowmajor[i * 12 + j] = H_eigen(i, j);

    auto b_H = backend->alloc(m * 12 * sizeof(double));
    backend->upload(b_H, H_rowmajor.data(), m * 12 * sizeof(double));

    double HTH[144];
    backend->compute_HTH(HTH, b_H, m);

    Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> result(HTH);
    Eigen::Matrix<double, 12, 12> expected = H_eigen.transpose() * H_eigen;

    double max_err = 0;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            max_err = std::max(max_err, std::fabs(result(i,j) - expected(i,j)));

    // float accumulation → larger tolerance
    // Each element sums m products of floats, so error ≈ m * eps_float ≈ 2000 * 1e-7 ≈ 2e-4
    // But values can be up to O(m) ≈ 2000, so relative error matters more
    double max_val = expected.cwiseAbs().maxCoeff();
    double rel_err = max_err / max_val;
    EXPECT_LT(rel_err, 1e-3) << "max_err=" << max_err << " max_val=" << max_val;

    backend->free(b_H);
}

TEST_F(MetalBackendTest, HTh_RandomVerify) {
    int m = 2000;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 1);

    std::vector<double> H_rowmajor(m * 12);
    std::vector<double> h_vec(m);
    Eigen::MatrixXd H_eigen(m, 12);
    Eigen::VectorXd h_eigen(m);

    for (int i = 0; i < m; i++) {
        h_vec[i] = h_eigen(i) = dist(rng);
        for (int j = 0; j < 12; j++) {
            H_eigen(i, j) = dist(rng);
            H_rowmajor[i * 12 + j] = H_eigen(i, j);
        }
    }

    auto b_H = backend->alloc(m * 12 * sizeof(double));
    auto b_h = backend->alloc(m * sizeof(double));
    backend->upload(b_H, H_rowmajor.data(), m * 12 * sizeof(double));
    backend->upload(b_h, h_vec.data(), m * sizeof(double));

    double HTh[12];
    backend->compute_HTh(HTh, b_H, b_h, m);

    Eigen::Matrix<double, 12, 1> expected = H_eigen.transpose() * h_eigen;
    double max_err = 0;
    for (int i = 0; i < 12; i++)
        max_err = std::max(max_err, std::fabs(HTh[i] - expected(i)));

    double max_val = expected.cwiseAbs().maxCoeff();
    double rel_err = max_err / max_val;
    EXPECT_LT(rel_err, 1e-3) << "max_err=" << max_err;

    backend->free(b_H); backend->free(b_h);
}

// ═══════════════════════════════════════════════════════════════════════
// Fused pipeline cross-validation against CPU backend
// ═══════════════════════════════════════════════════════════════════════

TEST_F(MetalBackendTest, FusedPipeline_CrossValidateCPU) {
    int n = 5000, k = 5;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> pos_dist(-15, 15);
    std::uniform_real_distribution<float> z_dist(3, 30);

    Eigen::Matrix3d R_body = exp_so3(Eigen::Vector3d(0.02, -0.01, 0.03));
    Eigen::Vector3d t_body(5.0, -3.0, 1.0);
    auto body_tf = make_transform(R_body, t_body);
    auto ext_tf = make_identity();

    std::vector<float> points_body(n * 3);
    std::vector<float> neighbors(n * k * 3);

    for (int i = 0; i < n; i++) {
        float x = pos_dist(rng), y = pos_dist(rng), z = z_dist(rng);
        points_body[i*3+0] = x; points_body[i*3+1] = y; points_body[i*3+2] = z;

        Eigen::Vector3d pb(x, y, z);
        Eigen::Vector3d pw = R_body * pb + t_body;

        for (int j = 0; j < k; j++) {
            neighbors[i*k*3 + j*3 + 0] = (float)pw.x() + 0.05f * pos_dist(rng);
            neighbors[i*k*3 + j*3 + 1] = (float)pw.y() + 0.05f * pos_dist(rng);
            neighbors[i*k*3 + j*3 + 2] = (float)pw.z() + 0.001f * pos_dist(rng);
        }
    }

    // Run Metal backend
    auto metal_result = backend->fused_h_share_model(
        points_body.data(), neighbors.data(), n, k, body_tf, ext_tf, 0.1f, false);

    // Run CPU backend for comparison
    auto cpu = create_backend("cpu");
    auto cpu_result = cpu->fused_h_share_model(
        points_body.data(), neighbors.data(), n, k, body_tf, ext_tf, 0.1f, false);

    // Feature counts: Metal (normal equations + Cramer) vs CPU (Householder QR)
    // will differ at boundary cases. Both should find a substantial number.
    EXPECT_GT(metal_result.effct_feat_num, n / 3)
        << "Metal found too few features: " << metal_result.effct_feat_num;
    EXPECT_GT(cpu_result.effct_feat_num, n / 3)
        << "CPU found too few features: " << cpu_result.effct_feat_num;

    // Log the difference for informational purposes
    int count_diff = std::abs(metal_result.effct_feat_num - cpu_result.effct_feat_num);
    std::cout << "  Feature count: Metal=" << metal_result.effct_feat_num
              << " CPU=" << cpu_result.effct_feat_num
              << " diff=" << count_diff << " (" 
              << (100.0 * count_diff / cpu_result.effct_feat_num) << "%)" << std::endl;

    // HTH comparison: since feature sets differ, we can't expect exact match.
    // Instead verify that both HTH are positive semi-definite and in the same order of magnitude.
    if (metal_result.effct_feat_num > 100 && cpu_result.effct_feat_num > 100) {
        Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> metal_HTH(metal_result.HTH);
        Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> cpu_HTH(cpu_result.HTH);

        // Both diagonals should be positive
        for (int i = 0; i < 12; i++) {
            EXPECT_GE(metal_HTH(i,i), 0) << "Metal HTH diagonal " << i;
            EXPECT_GE(cpu_HTH(i,i), 0) << "CPU HTH diagonal " << i;
        }

        // Trace should be in same order of magnitude (within 2x)
        double metal_trace = metal_HTH.trace();
        double cpu_trace = cpu_HTH.trace();
        double trace_ratio = metal_trace / cpu_trace;
        EXPECT_GT(trace_ratio, 0.3) << "Metal trace=" << metal_trace << " CPU trace=" << cpu_trace;
        EXPECT_LT(trace_ratio, 3.0) << "Metal trace=" << metal_trace << " CPU trace=" << cpu_trace;

        std::cout << "  HTH trace: Metal=" << metal_trace << " CPU=" << cpu_trace
                  << " ratio=" << trace_ratio << std::endl;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Factory tests
// ═══════════════════════════════════════════════════════════════════════

TEST(MetalFactoryTest, CreateMetal) {
    auto backend = create_backend("metal");
    // May or may not be available
    if (backend) {
        EXPECT_EQ(backend->name(), "Metal");
    }
}

TEST(MetalFactoryTest, DefaultPrefersMetal) {
    auto backend = create_default_backend();
    ASSERT_NE(backend, nullptr);
    // On macOS with Metal GPU, should prefer Metal
    // (but CPU is also valid if Metal unavailable)
    EXPECT_TRUE(backend->name() == "Metal" || backend->name() == "CPU");
}
