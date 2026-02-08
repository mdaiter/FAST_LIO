/**
 * Benchmarks for the ComputeBackend abstraction.
 *
 * Measures each kernel individually and the fused pipeline,
 * providing the baseline that GPU backends must beat.
 */

#include <benchmark/benchmark.h>
#include <Eigen/Eigen>
#include <random>
#include <vector>
#include <cmath>

#include "compute/cpu_backend.h"

using namespace fastlio::compute;

// ─── Helpers ─────────────────────────────────────────────────────────

static RigidTransform make_transform(double rx, double ry, double rz,
                                      double tx, double ty, double tz) {
    RigidTransform rt;
    Eigen::Vector3d w(rx, ry, rz);
    double theta = w.norm();
    Eigen::Matrix3d R;
    if (theta < 1e-10) {
        R = Eigen::Matrix3d::Identity();
    } else {
        Eigen::Vector3d axis = w / theta;
        Eigen::Matrix3d K;
        K << 0, -axis.z(), axis.y(),
             axis.z(), 0, -axis.x(),
             -axis.y(), axis.x(), 0;
        R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
    }
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rt.R) = R;
    rt.t[0] = tx; rt.t[1] = ty; rt.t[2] = tz;
    return rt;
}

// Generate realistic point cloud + neighbors for benchmarking
struct TestData {
    std::vector<float> points_body;   // N x 3
    std::vector<float> neighbors;     // N x k x 3
    RigidTransform body_to_world;
    RigidTransform lidar_to_imu;
    int n, k;

    TestData(int n_, int k_ = 5, uint32_t seed = 42) : n(n_), k(k_) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> xy(-20, 20);
        std::uniform_real_distribution<float> z(1, 50);

        points_body.resize(n * 3);
        neighbors.resize(n * k * 3);

        body_to_world = make_transform(0.01, 0.02, 0.03, 1.0, 2.0, 3.0);
        lidar_to_imu = make_transform(0.0, 0.0, 0.0, 0.1, 0.2, 0.05);

        // Precompute combined rotation/translation for neighbor placement.
        // The fused pipeline does: pw = R_b2w * (R_l2i * pb + t_l2i) + t_b2w
        // Neighbors must be near pw (the world-frame point) so that
        // the plane fitted to neighbors has a small residual at pw.
        Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::ColMajor>> R_b2w(body_to_world.R);
        Eigen::Vector3d t_b2w(body_to_world.t[0], body_to_world.t[1], body_to_world.t[2]);
        Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::ColMajor>> R_l2i(lidar_to_imu.R);
        Eigen::Vector3d t_l2i(lidar_to_imu.t[0], lidar_to_imu.t[1], lidar_to_imu.t[2]);

        for (int i = 0; i < n; i++) {
            float x = xy(rng), y = xy(rng), zv = z(rng);
            points_body[i*3+0] = x;
            points_body[i*3+1] = y;
            points_body[i*3+2] = zv;

            // Compute world-frame position
            Eigen::Vector3d pb(x, y, zv);
            Eigen::Vector3d pw = R_b2w * (R_l2i * pb + t_l2i) + t_b2w;

            // Neighbors: slightly perturbed copies near WORLD point (simulating nearby map points)
            for (int j = 0; j < k; j++) {
                neighbors[i*k*3 + j*3 + 0] = (float)pw.x() + 0.05f * xy(rng);
                neighbors[i*k*3 + j*3 + 1] = (float)pw.y() + 0.05f * xy(rng);
                neighbors[i*k*3 + j*3 + 2] = (float)pw.z() + 0.002f * xy(rng);  // nearly planar
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Individual kernel benchmarks
// ═══════════════════════════════════════════════════════════════════════

static void BM_Backend_TransformPoints(benchmark::State& state) {
    int n = state.range(0);
    auto backend = create_backend("cpu");
    TestData data(n);

    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, data.points_body.data(), n * 3 * sizeof(float));

    for (auto _ : state) {
        backend->batch_transform_points(b_out, b_in, n, data.body_to_world, data.lidar_to_imu);
    }
    state.SetItemsProcessed(state.iterations() * n);
    backend->free(b_in);
    backend->free(b_out);
}
BENCHMARK(BM_Backend_TransformPoints)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000)->Arg(10000);

static void BM_Backend_PlaneFit(benchmark::State& state) {
    int n = state.range(0);
    int k = 5;
    auto backend = create_backend("cpu");
    TestData data(n, k);

    auto b_nb = backend->alloc(n * k * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * sizeof(PlaneCoeffs));
    backend->upload(b_nb, data.neighbors.data(), n * k * 3 * sizeof(float));

    for (auto _ : state) {
        backend->batch_plane_fit(b_planes, b_nb, n, k, 0.1f);
    }
    state.SetItemsProcessed(state.iterations() * n);
    backend->free(b_nb);
    backend->free(b_planes);
}
BENCHMARK(BM_Backend_PlaneFit)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000);

static void BM_Backend_BuildJacobian(benchmark::State& state) {
    int m = state.range(0);
    auto backend = create_backend("cpu");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10, 10);

    std::vector<float> pb(m * 3), normals(m * 3), dists(m);
    for (int i = 0; i < m * 3; i++) { pb[i] = dist(rng); normals[i] = dist(rng); }
    for (int i = 0; i < m; i++) {
        float n = std::sqrt(normals[i*3]*normals[i*3] + normals[i*3+1]*normals[i*3+1] + normals[i*3+2]*normals[i*3+2]);
        normals[i*3] /= n; normals[i*3+1] /= n; normals[i*3+2] /= n;
        dists[i] = dist(rng) * 0.01f;
    }

    auto b_H = backend->alloc(m * 12 * sizeof(double));
    auto b_h = backend->alloc(m * sizeof(double));
    auto b_pb = backend->alloc(m * 3 * sizeof(float));
    auto b_n = backend->alloc(m * 3 * sizeof(float));
    auto b_d = backend->alloc(m * sizeof(float));
    backend->upload(b_pb, pb.data(), m * 3 * sizeof(float));
    backend->upload(b_n, normals.data(), m * 3 * sizeof(float));
    backend->upload(b_d, dists.data(), m * sizeof(float));

    auto tf = make_transform(0.01, 0.02, 0.03, 1, 2, 3);

    for (auto _ : state) {
        backend->batch_build_jacobian(b_H, b_h, b_pb, b_n, b_d,
                                       m, tf.R, tf.R, tf.t, false);
    }
    state.SetItemsProcessed(state.iterations() * m);
    backend->free(b_H); backend->free(b_h);
    backend->free(b_pb); backend->free(b_n); backend->free(b_d);
}
BENCHMARK(BM_Backend_BuildJacobian)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000);

static void BM_Backend_HTH(benchmark::State& state) {
    int m = state.range(0);
    auto backend = create_backend("cpu");

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 1);
    std::vector<double> H(m * 12);
    for (auto& v : H) v = dist(rng);

    auto b_H = backend->alloc(m * 12 * sizeof(double));
    backend->upload(b_H, H.data(), m * 12 * sizeof(double));

    double HTH[144];
    for (auto _ : state) {
        backend->compute_HTH(HTH, b_H, m);
        benchmark::DoNotOptimize(HTH);
    }
    state.SetItemsProcessed(state.iterations() * m);
    backend->free(b_H);
}
BENCHMARK(BM_Backend_HTH)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000)->Arg(10000);

// ═══════════════════════════════════════════════════════════════════════
// Fused pipeline benchmark (the main target)
// ═══════════════════════════════════════════════════════════════════════

static void BM_Backend_FusedPipeline(benchmark::State& state) {
    int n = state.range(0);
    int k = 5;
    auto backend = create_backend("cpu");
    TestData data(n, k);

    for (auto _ : state) {
        auto result = backend->fused_h_share_model(
            data.points_body.data(), data.neighbors.data(),
            n, k, data.body_to_world, data.lidar_to_imu,
            0.1f, false
        );
        benchmark::DoNotOptimize(result.effct_feat_num);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_Backend_FusedPipeline)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000)->Arg(10000);

BENCHMARK_MAIN();
