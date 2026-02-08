/**
 * Unit tests for plane estimation functions (esti_plane, esti_normvector).
 * These are called once per point per EKF iteration — a key GPU kernel candidate.
 * 
 * We test independently of ROS by redefining the necessary types locally.
 */

#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cmath>
#include <random>
#include <vector>

// Replicate the types from common_lib.h without ROS deps
typedef pcl::PointXYZINormal PointType;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

#define NUM_MATCH_POINTS 5

// Include so3_math.h for Exp/Log used by some helpers
#include "so3_math.h"

// ──────────────────────────────────────────────
// Inline the esti_plane function from common_lib.h
// (to avoid pulling in ROS headers)
// ──────────────────────────────────────────────
template<typename T>
bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

// ──────────────────────────────────────────────
// Helper: create points on a known plane
// ──────────────────────────────────────────────
PointVector makePointsOnPlane(const Eigen::Vector3d& normal, double d, 
                               int n, std::mt19937& rng, double noise = 0.0) {
    // Plane: normal . x = -d (matching esti_plane convention: Ax+By+Cz+D=0, D > 0)
    // Create two tangent vectors
    Eigen::Vector3d t1, t2;
    if (std::abs(normal.x()) < 0.9) {
        t1 = normal.cross(Eigen::Vector3d::UnitX()).normalized();
    } else {
        t1 = normal.cross(Eigen::Vector3d::UnitY()).normalized();
    }
    t2 = normal.cross(t1).normalized();
    
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    std::normal_distribution<double> noise_dist(0.0, noise);
    
    PointVector points;
    for (int i = 0; i < n; i++) {
        double s = dist(rng), t = dist(rng);
        Eigen::Vector3d base = -d * normal;  // point on plane closest to origin
        Eigen::Vector3d p = base + s * t1 + t * t2;
        
        if (noise > 0) {
            p += noise_dist(rng) * normal;
        }
        
        PointType pt;
        pt.x = p.x();
        pt.y = p.y();
        pt.z = p.z();
        points.push_back(pt);
    }
    return points;
}

class PlaneEstimationTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};
};

// ──────────────────────────────────────────────
// esti_plane tests
// ──────────────────────────────────────────────

TEST_F(PlaneEstimationTest, PerfectXYPlane) {
    // z = 1 plane: normal = (0,0,1), d = -1
    // Convention: 0*x + 0*y + 1*z + (-1) = 0 → z = 1
    PointVector points;
    double z_val = 1.0;
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < NUM_MATCH_POINTS; i++) {
        PointType pt;
        pt.x = dist(rng);
        pt.y = dist(rng);
        pt.z = z_val;
        points.push_back(pt);
    }
    
    Eigen::Vector4d result;
    bool success = esti_plane(result, points, 0.1);
    
    EXPECT_TRUE(success);
    // Normal should be [0, 0, ±1] (normalized)
    EXPECT_NEAR(std::abs(result(2)), 1.0, 1e-6);
    EXPECT_NEAR(std::abs(result(0)), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(result(1)), 0.0, 1e-6);
}

TEST_F(PlaneEstimationTest, TiltedPlane) {
    Eigen::Vector3d normal = Eigen::Vector3d(1, 1, 1).normalized();
    double d = 2.0;
    PointVector points = makePointsOnPlane(normal, d, NUM_MATCH_POINTS, rng);
    
    Eigen::Vector4d result;
    bool success = esti_plane(result, points, 0.1);
    
    EXPECT_TRUE(success);
    // Check that estimated normal is parallel to true normal
    Eigen::Vector3d est_normal(result(0), result(1), result(2));
    double dot = std::abs(est_normal.normalized().dot(normal));
    EXPECT_NEAR(dot, 1.0, 1e-6);
}

TEST_F(PlaneEstimationTest, NoisyPlaneRejects) {
    // With extreme noise, should reject (return false) with tight threshold
    Eigen::Vector3d normal = Eigen::Vector3d(0, 0, 1).normalized();
    double d = 1.0;
    PointVector points = makePointsOnPlane(normal, d, NUM_MATCH_POINTS, rng, 10.0);
    
    Eigen::Vector4d result;
    bool success = esti_plane(result, points, 0.001);
    
    // Very noisy points with very tight threshold should fail
    EXPECT_FALSE(success);
}

TEST_F(PlaneEstimationTest, NoisyPlaneAccepts) {
    // With slight noise and generous threshold, should accept
    Eigen::Vector3d normal = Eigen::Vector3d(0, 1, 0).normalized();
    double d = 3.0;
    PointVector points = makePointsOnPlane(normal, d, NUM_MATCH_POINTS, rng, 0.001);
    
    Eigen::Vector4d result;
    bool success = esti_plane(result, points, 0.1);
    
    EXPECT_TRUE(success);
}

TEST_F(PlaneEstimationTest, PlaneDistanceCorrect) {
    // For plane z = 5: normal = (0,0,1), D = -5
    // esti_plane solves A/D*x + B/D*y + C/D*z = -1
    // So normvec = [0, 0, -1/5], then normalized and D = 1/norm
    PointVector points;
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    for (int i = 0; i < NUM_MATCH_POINTS; i++) {
        PointType pt;
        pt.x = dist(rng);
        pt.y = dist(rng);
        pt.z = 5.0;
        points.push_back(pt);
    }
    
    Eigen::Vector4d result;
    bool success = esti_plane(result, points, 0.1);
    EXPECT_TRUE(success);
    
    // Distance from origin to plane = |D| / |normal| = |result(3)| since normal is normalized
    EXPECT_NEAR(std::abs(result(3)), 5.0, 1e-6);
}

TEST_F(PlaneEstimationTest, MultipleRandomPlanes) {
    // Test many random planes for robustness
    std::normal_distribution<double> ndist(0.0, 1.0);
    
    for (int i = 0; i < 50; i++) {
        Eigen::Vector3d normal(ndist(rng), ndist(rng), ndist(rng));
        normal.normalize();
        double d = std::abs(ndist(rng)) + 0.1;  // positive distance
        
        PointVector points = makePointsOnPlane(normal, d, NUM_MATCH_POINTS, rng, 0.0);
        
        Eigen::Vector4d result;
        bool success = esti_plane(result, points, 0.1);
        
        ASSERT_TRUE(success) << "Failed for plane " << i 
                             << " normal=" << normal.transpose() << " d=" << d;
        
        // Verify all points satisfy the plane equation
        for (int j = 0; j < NUM_MATCH_POINTS; j++) {
            double residual = result(0) * points[j].x + result(1) * points[j].y 
                            + result(2) * points[j].z + result(3);
            EXPECT_NEAR(residual, 0.0, 1e-4) << "Plane " << i << " point " << j;
        }
    }
}
