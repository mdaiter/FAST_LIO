/**
 * Unit tests for the ComputeBackend abstraction (CPU reference backend).
 *
 * Tests all 7 kernels + the fused h_share_model pipeline against
 * known-correct values computed from the original FAST-LIO code.
 */

#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <cmath>
#include <random>
#include <vector>

#include "compute/cpu_backend.h"

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

// Rodrigues formula
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

class ComputeBackendTest : public ::testing::Test {
protected:
    std::unique_ptr<ComputeBackend> backend;

    void SetUp() override {
        backend = create_backend("cpu");
        ASSERT_NE(backend, nullptr);
        EXPECT_EQ(backend->name(), "CPU");
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Buffer management tests
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, BufferAllocFreeRoundtrip) {
    auto buf = backend->alloc(1024);
    ASSERT_NE(buf, INVALID_BUFFER);
    backend->free(buf);
}

TEST_F(ComputeBackendTest, BufferUploadDownload) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto buf = backend->alloc(data.size() * sizeof(float));

    ASSERT_TRUE(backend->upload(buf, data.data(), data.size() * sizeof(float)));

    std::vector<float> result(4);
    ASSERT_TRUE(backend->download(result.data(), buf, data.size() * sizeof(float)));

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(result[i], data[i]);
    }
    backend->free(buf);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 1: Batch point transformation
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, TransformIdentity) {
    int n = 3;
    std::vector<float> points = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<float> result(9);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    auto id = make_identity();
    backend->batch_transform_points(b_out, b_in, n, id, id);
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    for (int i = 0; i < 9; i++) {
        EXPECT_NEAR(result[i], points[i], 1e-5f);
    }
    backend->free(b_in);
    backend->free(b_out);
}

TEST_F(ComputeBackendTest, TransformTranslation) {
    int n = 1;
    std::vector<float> points = {1, 2, 3};
    std::vector<float> result(3);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    auto body = make_identity();
    body.t[0] = 10; body.t[1] = 20; body.t[2] = 30;
    auto ext = make_identity();

    backend->batch_transform_points(b_out, b_in, n, body, ext);
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    EXPECT_NEAR(result[0], 11.0f, 1e-5f);
    EXPECT_NEAR(result[1], 22.0f, 1e-5f);
    EXPECT_NEAR(result[2], 33.0f, 1e-5f);

    backend->free(b_in);
    backend->free(b_out);
}

TEST_F(ComputeBackendTest, TransformRotation) {
    // Rotate 90° around Z: (1,0,0) → (0,1,0)
    int n = 1;
    std::vector<float> points = {1, 0, 0};
    std::vector<float> result(3);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    Eigen::Matrix3d R = exp_so3(Eigen::Vector3d(0, 0, M_PI / 2));
    auto body = make_transform(R, Eigen::Vector3d::Zero());
    auto ext = make_identity();

    backend->batch_transform_points(b_out, b_in, n, body, ext);
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    EXPECT_NEAR(result[0], 0.0f, 1e-5f);
    EXPECT_NEAR(result[1], 1.0f, 1e-5f);
    EXPECT_NEAR(result[2], 0.0f, 1e-5f);

    backend->free(b_in);
    backend->free(b_out);
}

TEST_F(ComputeBackendTest, TransformWithExtrinsic) {
    // Test the two-stage transform: R_body * (R_ext * p + t_ext) + t_body
    int n = 1;
    std::vector<float> points = {1, 0, 0};
    std::vector<float> result(3);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    Eigen::Vector3d t_ext(0.1, 0.2, 0.3);
    auto ext = make_transform(Eigen::Matrix3d::Identity(), t_ext);
    auto body = make_identity();

    backend->batch_transform_points(b_out, b_in, n, body, ext);
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    EXPECT_NEAR(result[0], 1.1f, 1e-5f);
    EXPECT_NEAR(result[1], 0.2f, 1e-5f);
    EXPECT_NEAR(result[2], 0.3f, 1e-5f);

    backend->free(b_in);
    backend->free(b_out);
}

TEST_F(ComputeBackendTest, TransformBatch) {
    // Verify N > 1 works correctly
    int n = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10, 10);

    std::vector<float> points(n * 3);
    for (auto& p : points) p = dist(rng);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    Eigen::Matrix3d R = exp_so3(Eigen::Vector3d(0.1, 0.2, 0.3));
    Eigen::Vector3d t(1, 2, 3);
    auto body = make_transform(R, t);
    auto ext = make_identity();

    backend->batch_transform_points(b_out, b_in, n, body, ext);

    std::vector<float> result(n * 3);
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    // Verify against manual Eigen computation
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d p(points[i*3], points[i*3+1], points[i*3+2]);
        Eigen::Vector3d expected = R * p + t;
        EXPECT_NEAR(result[i*3+0], (float)expected.x(), 1e-4f) << "i=" << i;
        EXPECT_NEAR(result[i*3+1], (float)expected.y(), 1e-4f) << "i=" << i;
        EXPECT_NEAR(result[i*3+2], (float)expected.z(), 1e-4f) << "i=" << i;
    }

    backend->free(b_in);
    backend->free(b_out);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 2: Batch plane fitting
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, PlaneFitPerfectPlane) {
    // 5 points on z=2.0 plane
    int n = 1, k = 5;
    std::vector<float> neighbors = {
        1, 0, 2,    0, 1, 2,    -1, 0, 2,    0, -1, 2,    0.5, 0.5, 2
    };

    auto b_nb = backend->alloc(n * k * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * sizeof(PlaneCoeffs));
    backend->upload(b_nb, neighbors.data(), n * k * 3 * sizeof(float));

    backend->batch_plane_fit(b_planes, b_nb, n, k, 0.1f);

    PlaneCoeffs plane;
    backend->download(&plane, b_planes, sizeof(PlaneCoeffs));

    EXPECT_TRUE(plane.valid);
    // Normal should be (0, 0, ±1)
    EXPECT_NEAR(std::fabs(plane.c), 1.0f, 1e-3f);
    EXPECT_NEAR(plane.a, 0.0f, 1e-3f);
    EXPECT_NEAR(plane.b, 0.0f, 1e-3f);

    backend->free(b_nb);
    backend->free(b_planes);
}

TEST_F(ComputeBackendTest, PlaneFitNoisyReject) {
    // 5 points NOT on a plane (large residuals)
    int n = 1, k = 5;
    std::vector<float> neighbors = {
        0, 0, 0,    1, 0, 0,    0, 1, 0,    0, 0, 5,    1, 1, 10
    };

    auto b_nb = backend->alloc(n * k * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * sizeof(PlaneCoeffs));
    backend->upload(b_nb, neighbors.data(), n * k * 3 * sizeof(float));

    backend->batch_plane_fit(b_planes, b_nb, n, k, 0.01f);  // very tight threshold

    PlaneCoeffs plane;
    backend->download(&plane, b_planes, sizeof(PlaneCoeffs));

    EXPECT_FALSE(plane.valid);

    backend->free(b_nb);
    backend->free(b_planes);
}

TEST_F(ComputeBackendTest, PlaneFitBatch) {
    int n = 100, k = 5;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> xy_dist(-10, 10);

    std::vector<float> neighbors(n * k * 3);
    for (int i = 0; i < n; i++) {
        float z0 = xy_dist(rng);
        for (int j = 0; j < k; j++) {
            neighbors[i*k*3 + j*3 + 0] = xy_dist(rng);
            neighbors[i*k*3 + j*3 + 1] = xy_dist(rng);
            neighbors[i*k*3 + j*3 + 2] = z0;  // perfect plane at z=z0
        }
    }

    auto b_nb = backend->alloc(n * k * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * sizeof(PlaneCoeffs));
    backend->upload(b_nb, neighbors.data(), n * k * 3 * sizeof(float));

    backend->batch_plane_fit(b_planes, b_nb, n, k, 0.1f);

    std::vector<PlaneCoeffs> planes(n);
    backend->download(planes.data(), b_planes, n * sizeof(PlaneCoeffs));

    for (int i = 0; i < n; i++) {
        EXPECT_TRUE(planes[i].valid) << "plane " << i;
        EXPECT_NEAR(std::fabs(planes[i].c), 1.0f, 1e-2f) << "plane " << i;
    }

    backend->free(b_nb);
    backend->free(b_planes);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 3: Residual computation
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, ResidualComputeBasic) {
    int n = 1;
    // Point on the plane → small residual → valid
    // Plane: z = 0 → normal (0,0,1), d=0
    std::vector<float> points_world = {0, 0, 0.01f};  // near plane
    std::vector<float> points_body = {5, 5, 5};  // body point far from origin (large norm → lenient score)

    PlaneCoeffs plane = {0, 0, 1, 0, true};

    auto b_pw = backend->alloc(n * 3 * sizeof(float));
    auto b_pb = backend->alloc(n * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * sizeof(PlaneCoeffs));
    auto b_res = backend->alloc(n * sizeof(float));
    auto b_valid = backend->alloc(n * sizeof(uint8_t));

    backend->upload(b_pw, points_world.data(), n * 3 * sizeof(float));
    backend->upload(b_pb, points_body.data(), n * 3 * sizeof(float));
    backend->upload(b_planes, &plane, sizeof(PlaneCoeffs));

    backend->batch_compute_residuals(b_res, b_valid, b_pw, b_pb, b_planes, n);

    float residual;
    uint8_t valid;
    backend->download(&residual, b_res, sizeof(float));
    backend->download(&valid, b_valid, sizeof(uint8_t));

    EXPECT_NEAR(residual, 0.01f, 1e-5f);
    EXPECT_EQ(valid, 1);

    backend->free(b_pw); backend->free(b_pb); backend->free(b_planes);
    backend->free(b_res); backend->free(b_valid);
}

TEST_F(ComputeBackendTest, ResidualInvalidPlane) {
    int n = 1;
    std::vector<float> pw = {0, 0, 0};
    std::vector<float> pb = {0, 0, 0};
    PlaneCoeffs plane = {0, 0, 1, 0, false};  // invalid plane

    auto b_pw = backend->alloc(n * 3 * sizeof(float));
    auto b_pb = backend->alloc(n * 3 * sizeof(float));
    auto b_planes = backend->alloc(sizeof(PlaneCoeffs));
    auto b_res = backend->alloc(sizeof(float));
    auto b_valid = backend->alloc(sizeof(uint8_t));

    backend->upload(b_pw, pw.data(), n * 3 * sizeof(float));
    backend->upload(b_pb, pb.data(), n * 3 * sizeof(float));
    backend->upload(b_planes, &plane, sizeof(PlaneCoeffs));

    backend->batch_compute_residuals(b_res, b_valid, b_pw, b_pb, b_planes, n);

    uint8_t valid;
    backend->download(&valid, b_valid, sizeof(uint8_t));
    EXPECT_EQ(valid, 0);

    backend->free(b_pw); backend->free(b_pb); backend->free(b_planes);
    backend->free(b_res); backend->free(b_valid);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 4: Jacobian construction
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, JacobianBasic) {
    int m = 1;
    std::vector<float> pb = {1, 2, 3};
    std::vector<float> normals = {0, 0, 1};
    std::vector<float> dists = {0.5f};

    auto b_H = backend->alloc(m * 12 * sizeof(double));
    auto b_h = backend->alloc(m * sizeof(double));
    auto b_pb = backend->alloc(m * 3 * sizeof(float));
    auto b_n = backend->alloc(m * 3 * sizeof(float));
    auto b_d = backend->alloc(m * sizeof(float));

    backend->upload(b_pb, pb.data(), m * 3 * sizeof(float));
    backend->upload(b_n, normals.data(), m * 3 * sizeof(float));
    backend->upload(b_d, dists.data(), m * sizeof(float));

    Eigen::Matrix3d R_body = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_ext = Eigen::Matrix3d::Identity();
    double t_ext[3] = {0, 0, 0};

    backend->batch_build_jacobian(b_H, b_h, b_pb, b_n, b_d,
                                   m, R_body.data(), R_ext.data(), t_ext, false);

    std::vector<double> H(12);
    double h;
    backend->download(H.data(), b_H, 12 * sizeof(double));
    backend->download(&h, b_h, sizeof(double));

    // h = -dist
    EXPECT_NEAR(h, -0.5, 1e-10);

    // First 3 elements = normal
    EXPECT_NEAR(H[0], 0, 1e-10);
    EXPECT_NEAR(H[1], 0, 1e-10);
    EXPECT_NEAR(H[2], 1, 1e-10);

    // With identity R_body and R_ext, C = R^T * n = n = (0,0,1)
    // point_imu = R_ext * p + t_ext = (1,2,3)
    // A = skew(1,2,3) * (0,0,1) = (2, -1, 0)
    EXPECT_NEAR(H[3], 2, 1e-10);
    EXPECT_NEAR(H[4], -1, 1e-10);
    EXPECT_NEAR(H[5], 0, 1e-10);

    // extrinsic_est_en = false → last 6 zeros
    for (int i = 6; i < 12; i++) {
        EXPECT_NEAR(H[i], 0, 1e-10);
    }

    backend->free(b_H); backend->free(b_h);
    backend->free(b_pb); backend->free(b_n); backend->free(b_d);
}

TEST_F(ComputeBackendTest, JacobianWithExtrinsic) {
    int m = 1;
    std::vector<float> pb = {1, 0, 0};
    std::vector<float> normals = {1, 0, 0};
    std::vector<float> dists = {0.1f};

    auto b_H = backend->alloc(m * 12 * sizeof(double));
    auto b_h = backend->alloc(m * sizeof(double));
    auto b_pb = backend->alloc(m * 3 * sizeof(float));
    auto b_n = backend->alloc(m * 3 * sizeof(float));
    auto b_d = backend->alloc(m * sizeof(float));

    backend->upload(b_pb, pb.data(), m * 3 * sizeof(float));
    backend->upload(b_n, normals.data(), m * 3 * sizeof(float));
    backend->upload(b_d, dists.data(), m * sizeof(float));

    Eigen::Matrix3d R_body = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_ext = Eigen::Matrix3d::Identity();
    double t_ext[3] = {0, 0, 0};

    backend->batch_build_jacobian(b_H, b_h, b_pb, b_n, b_d,
                                   m, R_body.data(), R_ext.data(), t_ext, true);

    std::vector<double> H(12);
    backend->download(H.data(), b_H, 12 * sizeof(double));

    // With extrinsic enabled:
    // C = R^T * n = (1,0,0)
    // B = skew(p_body) * R_ext^T * C = skew(1,0,0) * (1,0,0) = (0,0,0) × (1,0,0) = 0
    // Actually skew(1,0,0) = [[0,0,0],[0,0,-1],[0,1,0]], so * (1,0,0) = (0,0,0)
    EXPECT_NEAR(H[6], 0, 1e-10);
    EXPECT_NEAR(H[7], 0, 1e-10);
    EXPECT_NEAR(H[8], 0, 1e-10);

    // C columns
    EXPECT_NEAR(H[9], 1, 1e-10);
    EXPECT_NEAR(H[10], 0, 1e-10);
    EXPECT_NEAR(H[11], 0, 1e-10);

    backend->free(b_H); backend->free(b_h);
    backend->free(b_pb); backend->free(b_n); backend->free(b_d);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 5: H^T * H
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, HTH_Basic) {
    int m = 3;
    // Simple 3x12 H matrix
    std::vector<double> H(m * 12, 0.0);
    // Row 0: [1, 0, 0, ...]
    H[0] = 1.0;
    // Row 1: [0, 1, 0, ...]
    H[1*12 + 1] = 1.0;
    // Row 2: [1, 1, 0, ...]
    H[2*12 + 0] = 1.0;
    H[2*12 + 1] = 1.0;

    auto b_H = backend->alloc(m * 12 * sizeof(double));
    backend->upload(b_H, H.data(), m * 12 * sizeof(double));

    double HTH[144];
    backend->compute_HTH(HTH, b_H, m);

    // H^T * H:
    // Column 0 of H: [1, 0, 1]
    // Column 1 of H: [0, 1, 1]
    // HTH[0,0] = 1+0+1 = 2
    // HTH[1,1] = 0+1+1 = 2
    // HTH[0,1] = 0+0+1 = 1
    Eigen::Map<Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> result(HTH);
    EXPECT_NEAR(result(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(result(1, 1), 2.0, 1e-10);
    EXPECT_NEAR(result(0, 1), 1.0, 1e-10);
    EXPECT_NEAR(result(1, 0), 1.0, 1e-10);

    backend->free(b_H);
}

TEST_F(ComputeBackendTest, HTH_RandomVerify) {
    int m = 500;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 1);

    Eigen::MatrixXd H_eigen(m, 12);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < 12; j++)
            H_eigen(i, j) = dist(rng);

    // Convert to row-major for our interface
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

    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            EXPECT_NEAR(result(i, j), expected(i, j), 1e-6)
                << "HTH(" << i << "," << j << ")";

    backend->free(b_H);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 6: H^T * h
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, HTh_RandomVerify) {
    int m = 500;
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
    for (int i = 0; i < 12; i++) {
        EXPECT_NEAR(HTh[i], expected(i), 1e-6) << "HTh[" << i << "]";
    }

    backend->free(b_H);
    backend->free(b_h);
}

// ═══════════════════════════════════════════════════════════════════════
// Kernel 7: Undistortion
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, UndistortIdentity) {
    // With identity transforms and zero motion, points should be unchanged
    int n = 3, num_seg = 1;
    std::vector<float> points = {1, 2, 3,  4, 5, 6,  7, 8, 9};
    std::vector<float> timestamps = {0, 0, 0};

    auto b_pts = backend->alloc(n * 3 * sizeof(float));
    auto b_ts = backend->alloc(n * sizeof(float));

    // 1 segment: identity rotation, zero vel/pos/acc/angvel
    double seg_R[9];
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>> seg_R_map(seg_R);
    seg_R_map = Eigen::Matrix3d::Identity();
    double seg_vel[3] = {0, 0, 0};
    double seg_pos[3] = {0, 0, 0};
    double seg_acc[3] = {0, 0, 0};
    double seg_angvel[3] = {0, 0, 0};
    double seg_t_start[1] = {0.0};

    auto b_seg_R = backend->alloc(9 * sizeof(double));
    auto b_seg_vel = backend->alloc(3 * sizeof(double));
    auto b_seg_pos = backend->alloc(3 * sizeof(double));
    auto b_seg_acc = backend->alloc(3 * sizeof(double));
    auto b_seg_angvel = backend->alloc(3 * sizeof(double));
    auto b_seg_t = backend->alloc(sizeof(double));

    backend->upload(b_pts, points.data(), n * 3 * sizeof(float));
    backend->upload(b_ts, timestamps.data(), n * sizeof(float));
    backend->upload(b_seg_R, seg_R, 9 * sizeof(double));
    backend->upload(b_seg_vel, seg_vel, 3 * sizeof(double));
    backend->upload(b_seg_pos, seg_pos, 3 * sizeof(double));
    backend->upload(b_seg_acc, seg_acc, 3 * sizeof(double));
    backend->upload(b_seg_angvel, seg_angvel, 3 * sizeof(double));
    backend->upload(b_seg_t, seg_t_start, sizeof(double));

    auto imu_end = make_identity();
    auto ext = make_identity();

    backend->batch_undistort_points(b_pts, b_ts, b_seg_R, b_seg_vel, b_seg_pos,
                                     b_seg_acc, b_seg_angvel, b_seg_t,
                                     n, num_seg, imu_end, ext);

    std::vector<float> result(n * 3);
    backend->download(result.data(), b_pts, n * 3 * sizeof(float));

    for (int i = 0; i < n * 3; i++) {
        EXPECT_NEAR(result[i], points[i], 1e-4f) << "i=" << i;
    }

    backend->free(b_pts); backend->free(b_ts);
    backend->free(b_seg_R); backend->free(b_seg_vel); backend->free(b_seg_pos);
    backend->free(b_seg_acc); backend->free(b_seg_angvel); backend->free(b_seg_t);
}

// ═══════════════════════════════════════════════════════════════════════
// Fused pipeline test
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, FusedPipelineBasic) {
    // Create points on a z=5 plane with known neighbors
    int n = 10, k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> xy_dist(-2, 2);

    std::vector<float> points_body(n * 3);
    std::vector<float> neighbors(n * k * 3);

    for (int i = 0; i < n; i++) {
        // Body points reasonably far from origin (for score calculation)
        float x = xy_dist(rng);
        float y = xy_dist(rng);
        float z = 5.0f;
        points_body[i*3+0] = x;
        points_body[i*3+1] = y;
        points_body[i*3+2] = z;

        // Neighbors: 5 points near world position (which = body with identity transform)
        for (int j = 0; j < k; j++) {
            neighbors[i*k*3 + j*3 + 0] = x + 0.1f * xy_dist(rng);
            neighbors[i*k*3 + j*3 + 1] = y + 0.1f * xy_dist(rng);
            neighbors[i*k*3 + j*3 + 2] = z + 0.001f * xy_dist(rng);  // tight z → good plane
        }
    }

    auto body = make_identity();
    auto ext = make_identity();

    auto result = backend->fused_h_share_model(
        points_body.data(), neighbors.data(),
        n, k, body, ext, 0.1f, false
    );

    // Most points should be valid (near-perfect planes, small residuals)
    EXPECT_GT(result.effct_feat_num, 0);
    EXPECT_LE(result.effct_feat_num, n);

    // Check sizes are consistent
    EXPECT_EQ((int)result.valid_points_body.size(), result.effct_feat_num * 3);
    EXPECT_EQ((int)result.valid_normals.size(), result.effct_feat_num * 3);
    EXPECT_EQ((int)result.valid_residuals.size(), result.effct_feat_num);

    // HTH should be 12x12, not all zeros if we have valid features
    if (result.effct_feat_num > 0) {
        double sum = 0;
        for (int i = 0; i < 144; i++) sum += std::fabs(result.HTH[i]);
        EXPECT_GT(sum, 0.0);
    }
}

TEST_F(ComputeBackendTest, FusedPipelineNoValidPoints) {
    // All points with very bad neighbors → no valid features
    int n = 5, k = 5;

    std::vector<float> points_body(n * 3);
    std::vector<float> neighbors(n * k * 3);

    // Points at origin
    for (int i = 0; i < n; i++) {
        points_body[i*3+0] = 0.001f;
        points_body[i*3+1] = 0.001f;
        points_body[i*3+2] = 0.001f;

        // Neighbors all over the place → bad plane fit
        for (int j = 0; j < k; j++) {
            neighbors[i*k*3 + j*3 + 0] = (float)(j * 10);
            neighbors[i*k*3 + j*3 + 1] = (float)(j * 20);
            neighbors[i*k*3 + j*3 + 2] = (float)(j * 30);
        }
    }

    auto body = make_identity();
    auto ext = make_identity();

    auto result = backend->fused_h_share_model(
        points_body.data(), neighbors.data(),
        n, k, body, ext, 0.001f, false  // very tight threshold
    );

    // Very likely 0 valid features
    EXPECT_EQ(result.effct_feat_num, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// Stress tests (larger data, cross-validate against manual Eigen math)
// ═══════════════════════════════════════════════════════════════════════

TEST_F(ComputeBackendTest, CrossValidateFusedPipeline_5000pts) {
    // Cross-validate the fused pipeline against step-by-step manual computation
    // exactly mirroring h_share_model() logic from laserMapping.cpp:650-752
    int n = 5000, k = 5;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> pos_dist(-15, 15);
    std::uniform_real_distribution<float> z_dist(3, 30);

    // Realistic transforms
    Eigen::Matrix3d R_body = exp_so3(Eigen::Vector3d(0.02, -0.01, 0.03));
    Eigen::Vector3d t_body(5.0, -3.0, 1.0);
    Eigen::Matrix3d R_ext = Eigen::Matrix3d::Identity();  // typical: identity or small rotation
    Eigen::Vector3d t_ext(0.1, 0.05, -0.02);

    auto body_tf = make_transform(R_body, t_body);
    auto ext_tf = make_transform(R_ext, t_ext);

    std::vector<float> points_body(n * 3);
    std::vector<float> neighbors(n * k * 3);

    for (int i = 0; i < n; i++) {
        float x = pos_dist(rng), y = pos_dist(rng), z = z_dist(rng);
        points_body[i*3+0] = x;
        points_body[i*3+1] = y;
        points_body[i*3+2] = z;

        // Compute world point for neighbor generation
        Eigen::Vector3d pb(x, y, z);
        Eigen::Vector3d pw = R_body * (R_ext * pb + t_ext) + t_body;

        // Neighbors near the world point, nearly coplanar
        for (int j = 0; j < k; j++) {
            neighbors[i*k*3 + j*3 + 0] = (float)pw.x() + 0.05f * pos_dist(rng);
            neighbors[i*k*3 + j*3 + 1] = (float)pw.y() + 0.05f * pos_dist(rng);
            neighbors[i*k*3 + j*3 + 2] = (float)pw.z() + 0.001f * pos_dist(rng);
        }
    }

    // Run through backend
    auto result = backend->fused_h_share_model(
        points_body.data(), neighbors.data(),
        n, k, body_tf, ext_tf, 0.1f, false
    );

    // Should have a substantial number of valid features (planes are nearly perfect)
    EXPECT_GT(result.effct_feat_num, n / 2)
        << "Expected most points to produce valid features";

    // Now manually compute the same thing step by step and cross-validate
    int manual_valid_count = 0;
    std::vector<Eigen::Vector3d> manual_normals;
    std::vector<float> manual_dists;
    std::vector<Eigen::Vector3d> manual_bodies;

    for (int i = 0; i < n; i++) {
        Eigen::Vector3d pb(points_body[i*3], points_body[i*3+1], points_body[i*3+2]);
        Eigen::Vector3d pw = R_body * (R_ext * pb + t_ext) + t_body;

        // Plane fit (same as esti_plane)
        Eigen::Matrix<float, 5, 3> A;
        Eigen::Matrix<float, 5, 1> b;
        b.setConstant(-1.0f);
        for (int j = 0; j < k; j++) {
            A(j, 0) = neighbors[i*k*3 + j*3 + 0];
            A(j, 1) = neighbors[i*k*3 + j*3 + 1];
            A(j, 2) = neighbors[i*k*3 + j*3 + 2];
        }
        Eigen::Vector3f normvec = A.colPivHouseholderQr().solve(b);
        float norm = normvec.norm();
        if (norm < 1e-10f) continue;

        Eigen::Vector4f pabcd;
        pabcd(0) = normvec(0) / norm;
        pabcd(1) = normvec(1) / norm;
        pabcd(2) = normvec(2) / norm;
        pabcd(3) = 1.0f / norm;

        // Check plane residuals
        bool plane_ok = true;
        for (int j = 0; j < k; j++) {
            float res = pabcd(0) * A(j, 0) + pabcd(1) * A(j, 1) + pabcd(2) * A(j, 2) + pabcd(3);
            if (std::fabs(res) > 0.1f) { plane_ok = false; break; }
        }
        if (!plane_ok) continue;

        // Residual + score (same as h_share_model lines 680-691)
        float pd2 = pabcd(0) * (float)pw.x() + pabcd(1) * (float)pw.y()
                   + pabcd(2) * (float)pw.z() + pabcd(3);
        float s = 1.0f - 0.9f * std::fabs(pd2) / std::sqrt((float)pb.norm());
        if (s > 0.9f) {
            manual_valid_count++;
            manual_normals.push_back(Eigen::Vector3d(pabcd(0), pabcd(1), pabcd(2)));
            manual_dists.push_back(pd2);
            manual_bodies.push_back(pb);
        }
    }

    // Valid counts should match exactly
    EXPECT_EQ(result.effct_feat_num, manual_valid_count)
        << "Backend and manual computation disagree on valid feature count";

    // If counts match, cross-validate the Jacobian computation
    if (result.effct_feat_num == manual_valid_count && manual_valid_count > 0) {
        // Rebuild H manually for the valid features
        Eigen::MatrixXd H_manual(manual_valid_count, 12);
        Eigen::VectorXd h_manual(manual_valid_count);

        Eigen::Matrix3d R_body_T = R_body.transpose();

        for (int i = 0; i < manual_valid_count; i++) {
            Eigen::Vector3d point_imu = R_ext * manual_bodies[i] + t_ext;
            Eigen::Vector3d C = R_body_T * manual_normals[i];
            Eigen::Matrix3d crossmat;
            crossmat << 0, -point_imu.z(), point_imu.y(),
                        point_imu.z(), 0, -point_imu.x(),
                        -point_imu.y(), point_imu.x(), 0;
            Eigen::Vector3d A_vec = crossmat * C;

            H_manual(i, 0) = manual_normals[i].x();
            H_manual(i, 1) = manual_normals[i].y();
            H_manual(i, 2) = manual_normals[i].z();
            H_manual(i, 3) = A_vec.x();
            H_manual(i, 4) = A_vec.y();
            H_manual(i, 5) = A_vec.z();
            for (int j = 6; j < 12; j++) H_manual(i, j) = 0.0;
            h_manual(i) = -(double)manual_dists[i];
        }

        Eigen::Matrix<double, 12, 12> HTH_manual = H_manual.transpose() * H_manual;
        Eigen::Matrix<double, 12, 1> HTh_manual = H_manual.transpose() * h_manual;

        // Compare HTH
        Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> HTH_backend(result.HTH);
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                EXPECT_NEAR(HTH_backend(i, j), HTH_manual(i, j), 1e-3)
                    << "HTH mismatch at (" << i << "," << j << ")";
            }
        }

        // Compare HTh
        for (int i = 0; i < 12; i++) {
            EXPECT_NEAR(result.HTh[i], HTh_manual(i), 1e-3)
                << "HTh mismatch at " << i;
        }
    }
}

TEST_F(ComputeBackendTest, TransformBatch_10000pts) {
    // Stress test: 10k points, verify every single one
    int n = 10000;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-50, 50);

    std::vector<float> points(n * 3);
    for (auto& p : points) p = dist(rng);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, points.data(), n * 3 * sizeof(float));

    Eigen::Matrix3d R_body = exp_so3(Eigen::Vector3d(-0.3, 0.15, 0.7));
    Eigen::Vector3d t_body(10, -5, 20);
    Eigen::Matrix3d R_ext = exp_so3(Eigen::Vector3d(0.01, -0.02, 0.005));
    Eigen::Vector3d t_ext(0.3, -0.1, 0.05);

    auto body = make_transform(R_body, t_body);
    auto ext = make_transform(R_ext, t_ext);

    backend->batch_transform_points(b_out, b_in, n, body, ext);

    std::vector<float> result(n * 3);
    backend->download(result.data(), b_out, n * 3 * sizeof(float));

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d p(points[i*3], points[i*3+1], points[i*3+2]);
        Eigen::Vector3d expected = R_body * (R_ext * p + t_ext) + t_body;
        for (int d = 0; d < 3; d++) {
            float diff = std::fabs(result[i*3+d] - (float)expected(d));
            if (diff > 1e-3f) {
                mismatches++;
                if (mismatches <= 5) {  // print first few
                    ADD_FAILURE() << "Point " << i << " dim " << d
                                  << ": got " << result[i*3+d]
                                  << " expected " << (float)expected(d)
                                  << " diff " << diff;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "Total mismatches: " << mismatches << " / " << n * 3;

    backend->free(b_in);
    backend->free(b_out);
}

TEST_F(ComputeBackendTest, HTH_LargeMatrix) {
    // 10k rows — this is the realistic ESKF scenario
    int m = 10000;
    std::mt19937 rng(77);
    std::normal_distribution<double> dist(0, 1);

    Eigen::MatrixXd H_eigen(m, 12);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < 12; j++)
            H_eigen(i, j) = dist(rng);

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
            max_err = std::max(max_err, std::fabs(result(i, j) - expected(i, j)));

    EXPECT_LT(max_err, 1e-4)
        << "Max HTH error: " << max_err << " (10k rows)";

    backend->free(b_H);
}

// ═══════════════════════════════════════════════════════════════════════
// Factory test
// ═══════════════════════════════════════════════════════════════════════

TEST(ComputeBackendFactoryTest, CreateCPU) {
    auto backend = create_backend("cpu");
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->name(), "CPU");
}

TEST(ComputeBackendFactoryTest, CreateDefault) {
    auto backend = create_default_backend();
    ASSERT_NE(backend, nullptr);
    // Should at least return a valid backend
    EXPECT_FALSE(backend->name().empty());
}

TEST(ComputeBackendFactoryTest, CreateUnknown) {
    auto backend = create_backend("imaginary_gpu");
    EXPECT_EQ(backend, nullptr);
}
