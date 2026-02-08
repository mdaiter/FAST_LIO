/**
 * Benchmark + Validation: Metal backend vs CPU backend
 * 
 * For each test size, runs BOTH backends on IDENTICAL data,
 * validates outputs match, then benchmarks.
 */

#include <benchmark/benchmark.h>
#include <Eigen/Eigen>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "compute/compute_backend.h"

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
        K << 0, -axis.z(), axis.y(), axis.z(), 0, -axis.x(), -axis.y(), axis.x(), 0;
        R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
    }
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rt.R) = R;
    rt.t[0] = tx; rt.t[1] = ty; rt.t[2] = tz;
    return rt;
}

struct TestData {
    std::vector<float> points_body;
    std::vector<float> neighbors;
    RigidTransform body_to_world;
    RigidTransform lidar_to_imu;
    int n, k;

    TestData(int n_, int k_ = 5) : n(n_), k(k_) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> xy(-20, 20);
        std::uniform_real_distribution<float> z(1, 50);

        points_body.resize(n * 3);
        neighbors.resize((size_t)n * k * 3);

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
            points_body[i*3+0] = x; points_body[i*3+1] = y; points_body[i*3+2] = zv;

            // Compute world-frame position
            Eigen::Vector3d pb(x, y, zv);
            Eigen::Vector3d pw = R_b2w * (R_l2i * pb + t_l2i) + t_b2w;

            for (int j = 0; j < k; j++) {
                size_t base = (size_t)i*k*3 + j*3;
                neighbors[base + 0] = (float)pw.x() + 0.05f * xy(rng);
                neighbors[base + 1] = (float)pw.y() + 0.05f * xy(rng);
                neighbors[base + 2] = (float)pw.z() + 0.002f * xy(rng);
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════
// VALIDATION: Run before benchmarks, print detailed comparison
// ═══════════════════════════════════════════════════════════════════════

static void validate_transform(int n) {
    auto cpu = create_backend("cpu");
    auto metal = create_backend("metal");
    if (!metal) { std::cout << "  [SKIP] Metal not available\n"; return; }

    TestData data(n);

    auto cpu_in = cpu->alloc(n * 3 * sizeof(float));
    auto cpu_out = cpu->alloc(n * 3 * sizeof(float));
    cpu->upload(cpu_in, data.points_body.data(), n * 3 * sizeof(float));
    cpu->batch_transform_points(cpu_out, cpu_in, n, data.body_to_world, data.lidar_to_imu);

    auto mtl_in = metal->alloc(n * 3 * sizeof(float));
    auto mtl_out = metal->alloc(n * 3 * sizeof(float));
    metal->upload(mtl_in, data.points_body.data(), n * 3 * sizeof(float));
    metal->batch_transform_points(mtl_out, mtl_in, n, data.body_to_world, data.lidar_to_imu);

    std::vector<float> cpu_pts(n * 3), mtl_pts(n * 3);
    cpu->download(cpu_pts.data(), cpu_out, n * 3 * sizeof(float));
    metal->download(mtl_pts.data(), mtl_out, n * 3 * sizeof(float));

    double max_err = 0;
    int mismatches = 0;
    for (int i = 0; i < n * 3; i++) {
        double err = std::fabs((double)cpu_pts[i] - (double)mtl_pts[i]);
        max_err = std::max(max_err, err);
        if (err > 1e-3) mismatches++;
    }

    std::cout << "  Transform " << n << " pts: max_err=" << max_err 
              << " mismatches=" << mismatches << "/" << n*3;
    if (mismatches == 0) std::cout << " ✓";
    else std::cout << " ✗";
    std::cout << "\n";

    cpu->free(cpu_in); cpu->free(cpu_out);
    metal->free(mtl_in); metal->free(mtl_out);
}

struct PlaneGPU { float a,b,c,d; uint32_t valid; };

static void validate_plane_fit(int n) {
    int k = 5;
    auto cpu = create_backend("cpu");
    auto metal = create_backend("metal");
    if (!metal) { std::cout << "  [SKIP] Metal not available\n"; return; }

    TestData data(n, k);

    auto cpu_nb = cpu->alloc((size_t)n * k * 3 * sizeof(float));
    auto cpu_planes = cpu->alloc(n * sizeof(PlaneCoeffs));
    cpu->upload(cpu_nb, data.neighbors.data(), (size_t)n * k * 3 * sizeof(float));
    cpu->batch_plane_fit(cpu_planes, cpu_nb, n, k, 0.1f);

    auto mtl_nb = metal->alloc((size_t)n * k * 3 * sizeof(float));
    auto mtl_planes = metal->alloc(n * sizeof(PlaneGPU));
    metal->upload(mtl_nb, data.neighbors.data(), (size_t)n * k * 3 * sizeof(float));
    metal->batch_plane_fit(mtl_planes, mtl_nb, n, k, 0.1f);

    std::vector<PlaneCoeffs> cpu_p(n);
    std::vector<PlaneGPU> mtl_p(n);
    cpu->download(cpu_p.data(), cpu_planes, n * sizeof(PlaneCoeffs));
    metal->download(mtl_p.data(), mtl_planes, n * sizeof(PlaneGPU));

    int cpu_valid = 0, mtl_valid = 0, both_valid = 0;
    int agree_on_validity = 0;
    double max_normal_err = 0;

    for (int i = 0; i < n; i++) {
        bool cv = cpu_p[i].valid;
        bool mv = (mtl_p[i].valid != 0);
        if (cv) cpu_valid++;
        if (mv) mtl_valid++;
        if (cv && mv) {
            both_valid++;
            // Compare normals (may be flipped)
            Eigen::Vector3f cn(cpu_p[i].a, cpu_p[i].b, cpu_p[i].c);
            Eigen::Vector3f mn(mtl_p[i].a, mtl_p[i].b, mtl_p[i].c);
            double dot = std::fabs(cn.dot(mn));
            double err = 1.0 - dot;
            max_normal_err = std::max(max_normal_err, err);
        }
        if (cv == mv) agree_on_validity++;
    }

    double agreement_pct = 100.0 * agree_on_validity / n;
    std::cout << "  PlaneFit " << n << " pts: CPU_valid=" << cpu_valid
              << " Metal_valid=" << mtl_valid << " both_valid=" << both_valid
              << " validity_agree=" << std::fixed << std::setprecision(1) << agreement_pct << "%"
              << " max_normal_err=" << std::scientific << std::setprecision(2) << max_normal_err;
    if (agreement_pct > 80 && max_normal_err < 0.01) std::cout << " ✓";
    else std::cout << " ⚠";
    std::cout << "\n" << std::defaultfloat;

    cpu->free(cpu_nb); cpu->free(cpu_planes);
    metal->free(mtl_nb); metal->free(mtl_planes);
}

static void validate_fused(int n) {
    int k = 5;
    auto cpu = create_backend("cpu");
    auto metal = create_backend("metal");
    if (!metal) { std::cout << "  [SKIP] Metal not available\n"; return; }

    TestData data(n, k);

    auto cpu_r = cpu->fused_h_share_model(data.points_body.data(), data.neighbors.data(),
        n, k, data.body_to_world, data.lidar_to_imu, 0.1f, false);
    auto mtl_r = metal->fused_h_share_model(data.points_body.data(), data.neighbors.data(),
        n, k, data.body_to_world, data.lidar_to_imu, 0.1f, false);

    // HTH comparison
    Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> cpu_HTH(cpu_r.HTH);
    Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::ColMajor>> mtl_HTH(mtl_r.HTH);

    double cpu_trace = cpu_HTH.trace();
    double mtl_trace = mtl_HTH.trace();
    double trace_ratio = (cpu_trace > 1e-10) ? mtl_trace / cpu_trace : 0;

    // HTh comparison
    double hth_max_cpu = 0, hth_max_mtl = 0;
    for (int i = 0; i < 12; i++) {
        hth_max_cpu = std::max(hth_max_cpu, std::fabs(cpu_r.HTh[i]));
        hth_max_mtl = std::max(hth_max_mtl, std::fabs(mtl_r.HTh[i]));
    }
    double hth_ratio = (hth_max_cpu > 1e-10) ? hth_max_mtl / hth_max_cpu : 0;

    std::cout << "  Fused " << n << " pts: CPU_feat=" << cpu_r.effct_feat_num
              << " Metal_feat=" << mtl_r.effct_feat_num
              << " HTH_trace_ratio=" << std::fixed << std::setprecision(3) << trace_ratio
              << " HTh_mag_ratio=" << std::setprecision(3) << hth_ratio;

    bool ok = (cpu_r.effct_feat_num > 0 && mtl_r.effct_feat_num > 0 &&
               trace_ratio > 0.3 && trace_ratio < 3.0);
    std::cout << (ok ? " ✓" : " ✗") << "\n" << std::defaultfloat;
}

// Run validation before benchmarks
static bool validation_done = false;
static void run_validation() {
    if (validation_done) return;
    validation_done = true;

    std::cout << "\n══════════════════════════════════════════════\n";
    std::cout << "OUTPUT VALIDATION (CPU vs Metal, same data)\n";
    std::cout << "══════════════════════════════════════════════\n";

    for (int n : {1000, 5000, 10000, 50000, 100000, 500000}) {
        std::cout << "\n─── N = " << n << " ───\n";
        validate_transform(n);
        validate_plane_fit(n);
        if (n <= 100000) validate_fused(n);  // fused is expensive at 500k
        else std::cout << "  Fused " << n << " pts: [skipped for time]\n";
    }
    std::cout << "\n══════════════════════════════════════════════\n\n";
}

// Hook: run validation on first benchmark
static void BM_Validation(benchmark::State& state) {
    run_validation();
    for (auto _ : state) {}
}
BENCHMARK(BM_Validation)->Iterations(1);

// ═══════════════════════════════════════════════════════════════════════
// Fused pipeline benchmarks
// ═══════════════════════════════════════════════════════════════════════

static void BM_FusedPipeline_CPU(benchmark::State& state) {
    int n = state.range(0);
    auto backend = create_backend("cpu");
    TestData data(n);
    for (auto _ : state) {
        auto r = backend->fused_h_share_model(data.points_body.data(), data.neighbors.data(),
            n, data.k, data.body_to_world, data.lidar_to_imu, 0.1f, false);
        benchmark::DoNotOptimize(r.effct_feat_num);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_FusedPipeline_CPU)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000)->Arg(100000);

static void BM_FusedPipeline_Metal(benchmark::State& state) {
    int n = state.range(0);
    auto backend = create_backend("metal");
    if (!backend) { state.SkipWithError("Metal not available"); return; }
    TestData data(n);
    for (auto _ : state) {
        auto r = backend->fused_h_share_model(data.points_body.data(), data.neighbors.data(),
            n, data.k, data.body_to_world, data.lidar_to_imu, 0.1f, false);
        benchmark::DoNotOptimize(r.effct_feat_num);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_FusedPipeline_Metal)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000)->Arg(100000);

// ═══════════════════════════════════════════════════════════════════════
// Plane fit (the bottleneck, isolated)
// ═══════════════════════════════════════════════════════════════════════

static void BM_PlaneFit_CPU(benchmark::State& state) {
    int n = state.range(0), k = 5;
    auto backend = create_backend("cpu");
    TestData data(n, k);
    auto b_nb = backend->alloc((size_t)n * k * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * sizeof(PlaneCoeffs));
    backend->upload(b_nb, data.neighbors.data(), (size_t)n * k * 3 * sizeof(float));
    for (auto _ : state) {
        backend->batch_plane_fit(b_planes, b_nb, n, k, 0.1f);
    }
    state.SetItemsProcessed(state.iterations() * n);
    backend->free(b_nb); backend->free(b_planes);
}
BENCHMARK(BM_PlaneFit_CPU)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000)->Arg(100000)->Arg(500000);

static void BM_PlaneFit_Metal(benchmark::State& state) {
    int n = state.range(0), k = 5;
    auto backend = create_backend("metal");
    if (!backend) { state.SkipWithError("Metal not available"); return; }
    TestData data(n, k);
    auto b_nb = backend->alloc((size_t)n * k * 3 * sizeof(float));
    auto b_planes = backend->alloc(n * 20);
    backend->upload(b_nb, data.neighbors.data(), (size_t)n * k * 3 * sizeof(float));
    for (auto _ : state) {
        backend->batch_plane_fit(b_planes, b_nb, n, k, 0.1f);
    }
    state.SetItemsProcessed(state.iterations() * n);
    backend->free(b_nb); backend->free(b_planes);
}
BENCHMARK(BM_PlaneFit_Metal)->Arg(1000)->Arg(5000)->Arg(10000)->Arg(50000)->Arg(100000)->Arg(500000);

// ═══════════════════════════════════════════════════════════════════════
// Transform kernel
// ═══════════════════════════════════════════════════════════════════════

static void BM_Transform_CPU(benchmark::State& state) {
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
    backend->free(b_in); backend->free(b_out);
}
BENCHMARK(BM_Transform_CPU)->Arg(1000)->Arg(10000)->Arg(50000)->Arg(100000)->Arg(500000);

static void BM_Transform_Metal(benchmark::State& state) {
    int n = state.range(0);
    auto backend = create_backend("metal");
    if (!backend) { state.SkipWithError("Metal not available"); return; }
    TestData data(n);
    auto b_in = backend->alloc(n * 3 * sizeof(float));
    auto b_out = backend->alloc(n * 3 * sizeof(float));
    backend->upload(b_in, data.points_body.data(), n * 3 * sizeof(float));
    for (auto _ : state) {
        backend->batch_transform_points(b_out, b_in, n, data.body_to_world, data.lidar_to_imu);
    }
    state.SetItemsProcessed(state.iterations() * n);
    backend->free(b_in); backend->free(b_out);
}
BENCHMARK(BM_Transform_Metal)->Arg(1000)->Arg(10000)->Arg(50000)->Arg(100000)->Arg(500000);

BENCHMARK_MAIN();
