/**
 * Benchmarks for computational kernels that are GPU conversion candidates.
 * These are the "embarrassingly parallel" operations from h_share_model().
 * 
 * Benchmarks:
 * 1. Point transformation (rotation + translation) — N independent
 * 2. Plane fitting (5x3 QR decomposition) — N independent  
 * 3. Jacobian H matrix construction — N independent
 * 4. H^T * H computation — parallel reduction
 * 5. Combined h_share_model inner loop (transform + knn + plane + jacobian)
 */

#include <benchmark/benchmark.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <vector>
#include <random>
#include <cmath>

#include "so3_math.h"

using PointType = pcl::PointXYZINormal;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

#define NUM_MATCH_POINTS 5

static PointType makePoint(float x, float y, float z) {
    PointType pt;
    pt.x = x; pt.y = y; pt.z = z;
    pt.intensity = 0; pt.normal_x = 0; pt.normal_y = 0; pt.normal_z = 0; pt.curvature = 0;
    return pt;
}

// ──────────────────────────────────────────────
// 1. Point transformation: R * p + t
// ──────────────────────────────────────────────

static void BM_PointTransform(benchmark::State& state) {
    int n = state.range(0);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50, 50);
    
    // Generate N points
    std::vector<Eigen::Vector3d> points(n);
    for (auto& p : points) p = Eigen::Vector3d(dist(rng), dist(rng), dist(rng));
    
    // Rotation + translation
    Eigen::Matrix3d R = Exp(Eigen::Vector3d(Eigen::Vector3d(0.1, 0.2, 0.3)));
    Eigen::Vector3d t(1.0, 2.0, 3.0);
    
    std::vector<Eigen::Vector3d> transformed(n);
    
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            transformed[i] = R * points[i] + t;
        }
        benchmark::DoNotOptimize(transformed.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_PointTransform)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000)->Arg(10000);

// ──────────────────────────────────────────────
// 2. Plane fitting: 5x3 QR decomposition
// ──────────────────────────────────────────────

static void BM_PlaneFitting(benchmark::State& state) {
    int n = state.range(0);  // number of independent plane fits
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10, 10);
    
    // Pre-generate point sets (5 points each)
    std::vector<std::array<Eigen::Vector3d, NUM_MATCH_POINTS>> point_sets(n);
    for (auto& ps : point_sets) {
        // Points roughly on a plane z = 1 + noise
        for (auto& p : ps) {
            p = Eigen::Vector3d(dist(rng), dist(rng), 1.0 + 0.01 * dist(rng));
        }
    }
    
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            Eigen::Matrix<double, NUM_MATCH_POINTS, 3> A;
            Eigen::Matrix<double, NUM_MATCH_POINTS, 1> b;
            b.setOnes();
            b *= -1.0;
            
            for (int j = 0; j < NUM_MATCH_POINTS; j++) {
                A(j, 0) = point_sets[i][j].x();
                A(j, 1) = point_sets[i][j].y();
                A(j, 2) = point_sets[i][j].z();
            }
            
            Eigen::Vector3d normvec = A.colPivHouseholderQr().solve(b);
            benchmark::DoNotOptimize(normvec.data());
        }
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_PlaneFitting)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000);

// ──────────────────────────────────────────────
// 3. Jacobian H row construction
//    Simulates the per-point Jacobian computation from h_share_model
// ──────────────────────────────────────────────

static void BM_JacobianConstruction(benchmark::State& state) {
    int n = state.range(0);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10, 10);
    
    // Pre-generate data
    Eigen::Matrix3d R_body = Exp(Eigen::Vector3d(Eigen::Vector3d(0.1, 0.2, 0.3)));
    Eigen::Matrix3d R_ext = Eigen::Matrix3d::Identity();
    
    std::vector<Eigen::Vector3d> points_body(n);
    std::vector<Eigen::Vector3d> normals(n);
    for (int i = 0; i < n; i++) {
        points_body[i] = Eigen::Vector3d(dist(rng), dist(rng), dist(rng));
        normals[i] = Eigen::Vector3d(dist(rng), dist(rng), dist(rng)).normalized();
    }
    
    Eigen::MatrixXd H(n, 12);
    
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            // This mirrors the Jacobian computation in h_share_model (lines 719-752)
            Eigen::Vector3d point_this = R_ext * points_body[i];
            Eigen::Matrix3d point_crossmat;
            point_crossmat << 0, -point_this.z(), point_this.y(),
                              point_this.z(), 0, -point_this.x(),
                              -point_this.y(), point_this.x(), 0;
            
            Eigen::Vector3d C = R_body.conjugate() * normals[i];
            Eigen::Vector3d A = point_crossmat * C;
            
            // H row: [A^T, normal^T, 0, 0, 0, 0] (simplified)
            H.block<1, 3>(i, 0) = A.transpose();
            H.block<1, 3>(i, 3) = normals[i].transpose();
            H.block<1, 6>(i, 6).setZero();
        }
        benchmark::DoNotOptimize(H.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_JacobianConstruction)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000);

// ──────────────────────────────────────────────
// 4. H^T * H computation (Nx12 -> 12x12)
//    This is the key reduction step in the Kalman update
// ──────────────────────────────────────────────

static void BM_HTH_Computation(benchmark::State& state) {
    int n = state.range(0);
    
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 1);
    
    Eigen::MatrixXd H(n, 12);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 12; j++)
            H(i, j) = dist(rng);
    
    Eigen::Matrix<double, 12, 12> HTH;
    
    for (auto _ : state) {
        HTH = H.transpose() * H;
        benchmark::DoNotOptimize(HTH.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_HTH_Computation)->Arg(500)->Arg(1000)->Arg(2000)->Arg(5000)->Arg(10000);

// ──────────────────────────────────────────────
// 5. Covariance update: P = (I - K*H) * P
//    23x23 matrix operations (stays on CPU, but baseline needed)
// ──────────────────────────────────────────────

static void BM_CovarianceUpdate(benchmark::State& state) {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 0.01);
    
    // Simulate 23x23 covariance
    Eigen::Matrix<double, 23, 23> P;
    P.setIdentity();
    for (int i = 0; i < 23; i++)
        for (int j = 0; j < 23; j++)
            P(i, j) += dist(rng);
    P = P * P.transpose();  // make positive definite
    
    // Kalman gain K: 23x12
    Eigen::Matrix<double, 23, 12> K;
    for (int i = 0; i < 23; i++)
        for (int j = 0; j < 12; j++)
            K(i, j) = dist(rng);
    
    // H: Nx12 but in the small-state path, only 12 cols matter
    Eigen::Matrix<double, 12, 23> H_x;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 23; j++)
            H_x(i, j) = dist(rng);
    
    for (auto _ : state) {
        Eigen::Matrix<double, 23, 23> L = Eigen::Matrix<double, 23, 23>::Identity() - K * H_x;
        Eigen::Matrix<double, 23, 23> P_new = L * P;
        benchmark::DoNotOptimize(P_new.data());
    }
}
BENCHMARK(BM_CovarianceUpdate);

// ──────────────────────────────────────────────
// 6. 23x23 matrix inversion (used in Kalman gain)
// ──────────────────────────────────────────────

static void BM_MatrixInverse23x23(benchmark::State& state) {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0, 0.01);
    
    Eigen::Matrix<double, 23, 23> P;
    P.setIdentity();
    for (int i = 0; i < 23; i++)
        for (int j = 0; j < 23; j++)
            P(i, j) += dist(rng);
    P = P * P.transpose();  // make positive definite
    
    for (auto _ : state) {
        Eigen::Matrix<double, 23, 23> P_inv = P.inverse();
        benchmark::DoNotOptimize(P_inv.data());
    }
}
BENCHMARK(BM_MatrixInverse23x23);

// ──────────────────────────────────────────────
// 7. SO(3) Exp computation (used in IMU undistortion)
// ──────────────────────────────────────────────

static void BM_SO3Exp(benchmark::State& state) {
    int n = state.range(0);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    
    std::vector<Eigen::Vector3d> vecs(n);
    for (auto& v : vecs) v = Eigen::Vector3d(dist(rng), dist(rng), dist(rng));
    
    std::vector<Eigen::Matrix3d> results(n);
    
    for (auto _ : state) {
        for (int i = 0; i < n; i++) {
            results[i] = Exp(vecs[i], 0.01);
        }
        benchmark::DoNotOptimize(results.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_SO3Exp)->Arg(100)->Arg(1000)->Arg(5000)->Arg(10000);
