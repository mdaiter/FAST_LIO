/**
 * Unit tests for ikd-Tree: incremental k-d tree used as the map structure.
 * This is the single biggest computational hotspot (~40-60% of frame time).
 * 
 * IMPORTANT: KD_TREE must be heap-allocated because MANUAL_Q contains a
 * static array of 1M Operation_Logger_Type entries (~80MB), which overflows
 * the stack if placed there.
 * 
 * Tests verify:
 * - Build from point cloud
 * - k-NN search correctness vs brute force
 * - Incremental point insertion  
 * - Box-based deletion
 * - Tree size/validity tracking
 * - Radius search correctness
 */

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <memory>

#include "ikd_Tree.h"

using PointType = pcl::PointXYZINormal;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using TreePtr = std::unique_ptr<KD_TREE<PointType>>;

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

PointType makePoint(float x, float y, float z) {
    PointType pt;
    pt.x = x; pt.y = y; pt.z = z;
    pt.intensity = 0;
    pt.normal_x = 0; pt.normal_y = 0; pt.normal_z = 0;
    pt.curvature = 0;
    return pt;
}

float pointDist(const PointType& a, const PointType& b) {
    return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z);
}

PointVector generateRandomCloud(int n, std::mt19937& rng, float range = 100.0f) {
    std::uniform_real_distribution<float> dist(-range, range);
    PointVector cloud;
    cloud.reserve(n);
    for (int i = 0; i < n; i++) {
        cloud.push_back(makePoint(dist(rng), dist(rng), dist(rng)));
    }
    return cloud;
}

// Brute-force k-NN for correctness verification
void bruteForceKNN(const PointVector& cloud, const PointType& query, int k,
                   PointVector& neighbors, std::vector<float>& distances) {
    struct Pair { float dist; int idx; };
    std::vector<Pair> all;
    all.reserve(cloud.size());
    for (int i = 0; i < (int)cloud.size(); i++) {
        all.push_back({pointDist(cloud[i], query), i});
    }
    std::sort(all.begin(), all.end(), [](const Pair& a, const Pair& b){ return a.dist < b.dist; });
    
    neighbors.clear();
    distances.clear();
    int found = std::min(k, (int)all.size());
    for (int i = 0; i < found; i++) {
        neighbors.push_back(cloud[all[i].idx]);
        distances.push_back(all[i].dist);
    }
}

TreePtr makeTree() {
    return std::make_unique<KD_TREE<PointType>>();
}

TreePtr makeTree(float delete_param, float balance_param, float box_length) {
    return std::make_unique<KD_TREE<PointType>>(delete_param, balance_param, box_length);
}

class IKDTreeTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};
};

// ──────────────────────────────────────────────
// Build tests
// ──────────────────────────────────────────────

TEST_F(IKDTreeTest, BuildSinglePoint) {
    auto tree = makeTree();
    PointVector cloud = {makePoint(1, 2, 3)};
    tree->Build(cloud);
    EXPECT_EQ(tree->size(), 1);
    EXPECT_EQ(tree->validnum(), 1);
}

TEST_F(IKDTreeTest, BuildSmallCloud) {
    auto tree = makeTree();
    PointVector cloud = generateRandomCloud(100, rng);
    tree->Build(cloud);
    EXPECT_EQ(tree->size(), 100);
    EXPECT_EQ(tree->validnum(), 100);
}

TEST_F(IKDTreeTest, BuildMediumCloud) {
    auto tree = makeTree();
    PointVector cloud = generateRandomCloud(10000, rng);
    tree->Build(cloud);
    EXPECT_EQ(tree->size(), 10000);
}

// ──────────────────────────────────────────────
// k-NN Search correctness tests
// ──────────────────────────────────────────────

TEST_F(IKDTreeTest, NearestSearchK1) {
    auto tree = makeTree();
    PointVector cloud = generateRandomCloud(1000, rng);
    tree->Build(cloud);
    
    PointType query = makePoint(0, 0, 0);
    PointVector result;
    std::vector<float> dists;
    tree->Nearest_Search(query, 1, result, dists);
    
    EXPECT_EQ(result.size(), 1u);
    
    // Verify against brute force
    PointVector bf_result;
    std::vector<float> bf_dists;
    bruteForceKNN(cloud, query, 1, bf_result, bf_dists);
    
    EXPECT_NEAR(dists[0], bf_dists[0], 1e-5);
}

TEST_F(IKDTreeTest, NearestSearchK5CorrectVsBruteForce) {
    auto tree = makeTree();
    PointVector cloud = generateRandomCloud(5000, rng);
    tree->Build(cloud);
    
    // Test multiple query points
    for (int trial = 0; trial < 20; trial++) {
        std::uniform_real_distribution<float> dist(-100, 100);
        PointType query = makePoint(dist(rng), dist(rng), dist(rng));
        
        PointVector tree_result;
        std::vector<float> tree_dists;
        tree->Nearest_Search(query, 5, tree_result, tree_dists);
        
        PointVector bf_result;
        std::vector<float> bf_dists;
        bruteForceKNN(cloud, query, 5, bf_result, bf_dists);
        
        ASSERT_EQ(tree_result.size(), 5u) << "Trial " << trial;
        
        // Distances should match (sorted)
        std::sort(tree_dists.begin(), tree_dists.end());
        for (int i = 0; i < 5; i++) {
            EXPECT_NEAR(tree_dists[i], bf_dists[i], 1e-4)
                << "Trial " << trial << " neighbor " << i;
        }
    }
}

TEST_F(IKDTreeTest, NearestSearchLargeK) {
    auto tree = makeTree();
    PointVector cloud = generateRandomCloud(100, rng, 10.0f);
    tree->Build(cloud);
    
    PointType query = makePoint(0, 0, 0);
    PointVector result;
    std::vector<float> dists;
    tree->Nearest_Search(query, 20, result, dists);
    
    EXPECT_EQ(result.size(), 20u);
    
    // Verify sorted distances match brute force
    PointVector bf_result;
    std::vector<float> bf_dists;
    bruteForceKNN(cloud, query, 20, bf_result, bf_dists);
    
    std::sort(dists.begin(), dists.end());
    for (int i = 0; i < 20; i++) {
        EXPECT_NEAR(dists[i], bf_dists[i], 1e-4) << "neighbor " << i;
    }
}

TEST_F(IKDTreeTest, NearestSearchWithMaxDist) {
    auto tree = makeTree();
    // Create a cluster at origin and one far away
    PointVector cloud;
    for (int i = 0; i < 50; i++) {
        cloud.push_back(makePoint(i * 0.01f, 0, 0));  // near origin
    }
    for (int i = 0; i < 50; i++) {
        cloud.push_back(makePoint(1000 + i * 0.01f, 0, 0));  // far away
    }
    tree->Build(cloud);
    
    PointType query = makePoint(0, 0, 0);
    PointVector result;
    std::vector<float> dists;
    tree->Nearest_Search(query, 100, result, dists, 1.0f);  // max_dist = 1.0
    
    // Should only find the nearby cluster
    for (const auto& d : dists) {
        EXPECT_LE(d, 1.0f);
    }
}

// ──────────────────────────────────────────────
// Incremental Insert tests
// ──────────────────────────────────────────────

TEST_F(IKDTreeTest, AddPointsNoDownsample) {
    auto tree = makeTree();
    PointVector initial = generateRandomCloud(100, rng);
    tree->Build(initial);
    
    PointVector new_points = generateRandomCloud(50, rng);
    tree->Add_Points(new_points, false);
    
    EXPECT_EQ(tree->size(), 150);
}

TEST_F(IKDTreeTest, AddPointsWithDownsample) {
    auto tree = makeTree(0.5, 0.6, 0.5);  // downsample_size = 0.5
    PointVector initial = generateRandomCloud(100, rng, 5.0f);
    tree->Build(initial);
    
    int initial_size = tree->size();
    
    // Add duplicate-ish points (within same voxel)
    PointVector new_points;
    for (auto& pt : initial) {
        PointType p = pt;
        p.x += 0.01f;
        new_points.push_back(p);
    }
    tree->Add_Points(new_points, true);
    
    // With downsampling, size should not double
    EXPECT_LT(tree->size(), initial_size * 2);
}

TEST_F(IKDTreeTest, AddPointsThenSearch) {
    auto tree = makeTree();
    PointVector initial = generateRandomCloud(1000, rng);
    tree->Build(initial);
    
    // Add a point very close to query — it should be found as nearest
    PointType special = makePoint(0.001f, 0.001f, 0.001f);
    PointVector to_add = {special};
    tree->Add_Points(to_add, false);
    
    PointType query = makePoint(0, 0, 0);
    PointVector result;
    std::vector<float> dists;
    tree->Nearest_Search(query, 1, result, dists);
    
    float expected_dist = pointDist(special, query);
    EXPECT_NEAR(dists[0], expected_dist, 1e-5);
}

// ──────────────────────────────────────────────
// Box Delete tests
// ──────────────────────────────────────────────

TEST_F(IKDTreeTest, DeleteByBox) {
    auto tree = makeTree();
    
    // Create points spread evenly
    PointVector cloud;
    for (int i = 0; i < 1000; i++) {
        cloud.push_back(makePoint(i * 0.1f, 0, 0));
    }
    tree->Build(cloud);
    EXPECT_EQ(tree->size(), 1000);
    
    // Delete points in box [0, 10] x [-1, 1] x [-1, 1] (should delete ~100 points)
    BoxPointType box;
    box.vertex_min[0] = 0; box.vertex_min[1] = -1; box.vertex_min[2] = -1;
    box.vertex_max[0] = 10; box.vertex_max[1] = 1; box.vertex_max[2] = 1;
    
    std::vector<BoxPointType> boxes = {box};
    int deleted = tree->Delete_Point_Boxes(boxes);
    
    EXPECT_GT(deleted, 0);
    EXPECT_LT(tree->validnum(), 1000);
}

// ──────────────────────────────────────────────
// Box Search tests
// ──────────────────────────────────────────────

TEST_F(IKDTreeTest, BoxSearch) {
    auto tree = makeTree();
    PointVector cloud;
    for (int x = 0; x < 10; x++)
        for (int y = 0; y < 10; y++)
            for (int z = 0; z < 10; z++)
                cloud.push_back(makePoint(x, y, z));
    tree->Build(cloud);
    
    BoxPointType box;
    box.vertex_min[0] = 2; box.vertex_min[1] = 2; box.vertex_min[2] = 2;
    box.vertex_max[0] = 5; box.vertex_max[1] = 5; box.vertex_max[2] = 5;
    
    PointVector result;
    tree->Box_Search(box, result);
    
    for (const auto& pt : result) {
        EXPECT_GE(pt.x, 2.0f);
        EXPECT_LE(pt.x, 5.0f);
        EXPECT_GE(pt.y, 2.0f);
        EXPECT_LE(pt.y, 5.0f);
        EXPECT_GE(pt.z, 2.0f);
        EXPECT_LE(pt.z, 5.0f);
    }
    EXPECT_GT(result.size(), 0u);
}

// ──────────────────────────────────────────────
// Radius Search tests
// ──────────────────────────────────────────────

TEST_F(IKDTreeTest, RadiusSearch) {
    auto tree = makeTree();
    PointVector cloud = generateRandomCloud(5000, rng, 50.0f);
    tree->Build(cloud);
    
    PointType query = makePoint(0, 0, 0);
    float radius = 10.0f;
    
    PointVector result;
    tree->Radius_Search(query, radius, result);
    
    // All returned points should be within radius
    for (const auto& pt : result) {
        float d = std::sqrt(pointDist(pt, query));
        EXPECT_LE(d, radius + 1e-4);
    }
    
    // Brute force count for verification
    int bf_count = 0;
    for (const auto& pt : cloud) {
        if (std::sqrt(pointDist(pt, query)) <= radius) bf_count++;
    }
    EXPECT_EQ((int)result.size(), bf_count);
}

// ──────────────────────────────────────────────
// Tree range tests
// ──────────────────────────────────────────────

TEST_F(IKDTreeTest, TreeRangeCoversAllPoints) {
    auto tree = makeTree();
    PointVector cloud = generateRandomCloud(1000, rng, 50.0f);
    tree->Build(cloud);
    
    BoxPointType range = tree->tree_range();
    
    for (const auto& pt : cloud) {
        EXPECT_GE(pt.x, range.vertex_min[0]);
        EXPECT_LE(pt.x, range.vertex_max[0]);
        EXPECT_GE(pt.y, range.vertex_min[1]);
        EXPECT_LE(pt.y, range.vertex_max[1]);
        EXPECT_GE(pt.z, range.vertex_min[2]);
        EXPECT_LE(pt.z, range.vertex_max[2]);
    }
}
