/**
 * Benchmarks for ikd-Tree operations — establishes CPU baselines for GPU comparison.
 * 
 * IMPORTANT: KD_TREE must be heap-allocated (MANUAL_Q has ~80MB static array).
 * 
 * Benchmarks:
 * - Tree build from N points
 * - k-NN search (single query, batch queries)
 * - Incremental point insertion
 * - Box deletion
 */

#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <vector>
#include <random>
#include <memory>

#include "ikd_Tree.h"

using PointType = pcl::PointXYZINormal;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

static PointType makePoint(float x, float y, float z) {
    PointType pt;
    pt.x = x; pt.y = y; pt.z = z;
    pt.intensity = 0;
    pt.normal_x = 0; pt.normal_y = 0; pt.normal_z = 0;
    pt.curvature = 0;
    return pt;
}

static PointVector generateCloud(int n, unsigned seed = 42, float range = 100.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-range, range);
    PointVector cloud;
    cloud.reserve(n);
    for (int i = 0; i < n; i++) {
        cloud.push_back(makePoint(dist(rng), dist(rng), dist(rng)));
    }
    return cloud;
}

// ──────────────────────────────────────────────
// Build benchmark
// ──────────────────────────────────────────────

static void BM_TreeBuild(benchmark::State& state) {
    int n = state.range(0);
    PointVector cloud = generateCloud(n);
    
    for (auto _ : state) {
        auto tree = std::make_unique<KD_TREE<PointType>>();
        tree->Build(cloud);
        benchmark::DoNotOptimize(tree->size());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_TreeBuild)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(500000);

// ──────────────────────────────────────────────
// Single k-NN query benchmark
// ──────────────────────────────────────────────

static void BM_NearestSearch_SingleQuery(benchmark::State& state) {
    int tree_size = state.range(0);
    int k = state.range(1);
    
    PointVector cloud = generateCloud(tree_size);
    auto tree = std::make_unique<KD_TREE<PointType>>();
    tree->Build(cloud);
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-100, 100);
    
    PointVector result;
    std::vector<float> dists;
    
    for (auto _ : state) {
        PointType query = makePoint(dist(rng), dist(rng), dist(rng));
        tree->Nearest_Search(query, k, result, dists);
        benchmark::DoNotOptimize(result.data());
    }
}
BENCHMARK(BM_NearestSearch_SingleQuery)
    ->Args({10000, 5})
    ->Args({100000, 5})
    ->Args({500000, 5})
    ->Args({100000, 1})
    ->Args({100000, 10})
    ->Args({100000, 20});

// ──────────────────────────────────────────────
// Batch k-NN search (simulates h_share_model loop)
// ──────────────────────────────────────────────

static void BM_NearestSearch_BatchQuery(benchmark::State& state) {
    int tree_size = state.range(0);
    int num_queries = state.range(1);
    int k = 5;  // NUM_MATCH_POINTS
    
    PointVector cloud = generateCloud(tree_size);
    auto tree = std::make_unique<KD_TREE<PointType>>();
    tree->Build(cloud);
    
    PointVector queries = generateCloud(num_queries, 999);
    
    for (auto _ : state) {
        PointVector result;
        std::vector<float> dists;
        for (int i = 0; i < num_queries; i++) {
            tree->Nearest_Search(queries[i], k, result, dists);
        }
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * num_queries);
}
BENCHMARK(BM_NearestSearch_BatchQuery)
    ->Args({100000, 500})     // typical FAST-LIO scenario
    ->Args({100000, 1000})
    ->Args({100000, 2000})
    ->Args({500000, 1000});

// ──────────────────────────────────────────────
// Incremental insert benchmark
// ──────────────────────────────────────────────

static void BM_AddPoints(benchmark::State& state) {
    int tree_size = state.range(0);
    int add_count = state.range(1);
    bool downsample = state.range(2);
    
    PointVector base_cloud = generateCloud(tree_size);
    PointVector add_cloud = generateCloud(add_count, 999);
    
    for (auto _ : state) {
        state.PauseTiming();
        auto tree = std::make_unique<KD_TREE<PointType>>();
        tree->Build(base_cloud);
        state.ResumeTiming();
        
        tree->Add_Points(add_cloud, downsample);
        benchmark::DoNotOptimize(tree->size());
    }
    state.SetItemsProcessed(state.iterations() * add_count);
}
BENCHMARK(BM_AddPoints)
    ->Args({100000, 500, 0})    // no downsample
    ->Args({100000, 500, 1})    // with downsample
    ->Args({100000, 1000, 0})
    ->Args({100000, 2000, 0});

// ──────────────────────────────────────────────
// Box delete benchmark
// ──────────────────────────────────────────────

static void BM_DeletePointBoxes(benchmark::State& state) {
    int tree_size = state.range(0);
    PointVector cloud = generateCloud(tree_size);
    
    BoxPointType box;
    box.vertex_min[0] = -25; box.vertex_min[1] = -25; box.vertex_min[2] = -25;
    box.vertex_max[0] = 25;  box.vertex_max[1] = 25;  box.vertex_max[2] = 25;
    std::vector<BoxPointType> boxes = {box};
    
    for (auto _ : state) {
        state.PauseTiming();
        auto tree = std::make_unique<KD_TREE<PointType>>();
        tree->Build(cloud);
        state.ResumeTiming();
        
        int deleted = tree->Delete_Point_Boxes(boxes);
        benchmark::DoNotOptimize(deleted);
    }
}
BENCHMARK(BM_DeletePointBoxes)->Arg(10000)->Arg(100000)->Arg(500000);
