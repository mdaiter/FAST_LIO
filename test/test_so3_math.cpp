/**
 * Unit tests for SO(3) math operations: Exp, Log, skew_sym_mat, RotMtoEuler
 * These are the core rotation math primitives used throughout FAST-LIO.
 * 
 * Tests verify:
 * - Exp produces valid rotation matrices (orthogonal, det=1)
 * - Log inverts Exp (roundtrip property)
 * - Small angle approximation behavior
 * - skew_sym_mat antisymmetry
 * - Euler angle conversion consistency
 */

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <random>

// Include the SO(3) math header directly
#include "so3_math.h"

class SO3MathTest : public ::testing::Test {
protected:
    std::mt19937 rng{42};  // fixed seed for reproducibility
    
    // Generate random 3-vector with given max magnitude
    Eigen::Vector3d randomVec(double max_norm = M_PI) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        Eigen::Vector3d v(dist(rng), dist(rng), dist(rng));
        v.normalize();
        std::uniform_real_distribution<double> mag(0.01, max_norm);
        return v * mag(rng);
    }
    
    // Check if a matrix is a valid rotation matrix
    void assertValidRotation(const Eigen::Matrix3d& R, const std::string& msg = "") {
        // R^T * R = I
        Eigen::Matrix3d err = R.transpose() * R - Eigen::Matrix3d::Identity();
        EXPECT_LT(err.norm(), 1e-10) << "R^T*R != I " << msg;
        // det(R) = 1
        EXPECT_NEAR(R.determinant(), 1.0, 1e-10) << "det(R) != 1 " << msg;
    }
};

// ──────────────────────────────────────────────
// skew_sym_mat tests
// ──────────────────────────────────────────────

TEST_F(SO3MathTest, SkewSymMatIsAntisymmetric) {
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    auto K = skew_sym_mat(v);
    
    // K should be antisymmetric: K + K^T = 0
    Eigen::Matrix3d sum = K + K.transpose();
    EXPECT_LT(sum.norm(), 1e-15);
}

TEST_F(SO3MathTest, SkewSymMatCrossProduct) {
    // K(v) * u = v x u
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    Eigen::Vector3d u(4.0, 5.0, 6.0);
    auto K = skew_sym_mat(v);
    
    Eigen::Vector3d cross_product = v.cross(u);
    Eigen::Vector3d K_u = K * u;
    
    EXPECT_LT((cross_product - K_u).norm(), 1e-15);
}

TEST_F(SO3MathTest, SkewSymMatZeroVector) {
    Eigen::Vector3d v(0.0, 0.0, 0.0);
    auto K = skew_sym_mat(v);
    EXPECT_LT(K.norm(), 1e-15);
}

// ──────────────────────────────────────────────
// Exp tests
// ──────────────────────────────────────────────

TEST_F(SO3MathTest, ExpZeroVectorIsIdentity) {
    Eigen::Matrix3d R = Exp(Eigen::Vector3d(0.0, 0.0, 0.0));
    Eigen::Matrix3d err = R - Eigen::Matrix3d::Identity();
    EXPECT_LT(err.norm(), 1e-10);
}

TEST_F(SO3MathTest, ExpProducesValidRotation) {
    for (int i = 0; i < 100; i++) {
        Eigen::Vector3d v = randomVec();
        Eigen::Matrix3d R = Exp(Eigen::Vector3d(v));
        assertValidRotation(R, "trial " + std::to_string(i));
    }
}

TEST_F(SO3MathTest, ExpSmallAngle) {
    // For very small angles, Exp(v) ≈ I + skew(v)
    Eigen::Vector3d v(1e-9, 2e-9, 3e-9);
    Eigen::Matrix3d R = Exp(Eigen::Vector3d(v));
    Eigen::Matrix3d approx = Eigen::Matrix3d::Identity() + skew_sym_mat(v);
    EXPECT_LT((R - approx).norm(), 1e-8);
}

TEST_F(SO3MathTest, ExpRotationX90) {
    // Exp([pi/2, 0, 0]) should rotate 90° around X
    double angle = M_PI / 2.0;
    Eigen::Matrix3d R = Exp(Eigen::Vector3d(Eigen::Vector3d(angle, 0, 0)));
    
    // Should map [0,1,0] -> [0,0,1]
    Eigen::Vector3d result = R * Eigen::Vector3d(0, 1, 0);
    EXPECT_NEAR(result(0), 0.0, 1e-10);
    EXPECT_NEAR(result(1), 0.0, 1e-10);
    EXPECT_NEAR(result(2), 1.0, 1e-10);
}

TEST_F(SO3MathTest, ExpWithAngVelAndDt) {
    // Exp(ang_vel, dt) should equal Exp(ang_vel * dt)
    Eigen::Vector3d ang_vel(0.5, 1.0, -0.3);
    double dt = 0.01;
    
    Eigen::Matrix3d R1 = Exp(ang_vel, dt);
    Eigen::Matrix3d R2 = Exp(Eigen::Vector3d(ang_vel * dt));
    
    EXPECT_LT((R1 - R2).norm(), 1e-10);
}

TEST_F(SO3MathTest, ExpThreeScalarsMatchesVector) {
    double v1 = 0.3, v2 = -0.5, v3 = 0.7;
    Eigen::Matrix3d R1 = Exp(v1, v2, v3);
    Eigen::Matrix3d R2 = Exp(Eigen::Vector3d(Eigen::Vector3d(v1, v2, v3)));
    EXPECT_LT((R1 - R2).norm(), 1e-10);
}

// ──────────────────────────────────────────────
// Log tests
// ──────────────────────────────────────────────

TEST_F(SO3MathTest, LogIdentityIsZero) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Vector3d v = Log(I);
    EXPECT_LT(v.norm(), 1e-10);
}

TEST_F(SO3MathTest, ExpLogRoundtrip) {
    for (int i = 0; i < 100; i++) {
        Eigen::Vector3d v_orig = randomVec(M_PI * 0.95);  // stay away from ±π
        Eigen::Matrix3d R = Exp(Eigen::Vector3d(v_orig));
        Eigen::Vector3d v_recovered = Log(R);
        
        // The recovered angle-axis should produce the same rotation
        Eigen::Matrix3d R_recovered = Exp(Eigen::Vector3d(v_recovered));
        EXPECT_LT((R - R_recovered).norm(), 1e-8)
            << "Roundtrip failed for v = " << v_orig.transpose();
    }
}

TEST_F(SO3MathTest, LogExpRoundtripSmallAngle) {
    Eigen::Vector3d v(1e-5, 2e-5, -3e-5);
    Eigen::Matrix3d R = Exp(Eigen::Vector3d(v));
    Eigen::Vector3d v2 = Log(R);
    EXPECT_LT((v - v2).norm(), 1e-9);
}

// ──────────────────────────────────────────────
// RotMtoEuler tests
// ──────────────────────────────────────────────

TEST_F(SO3MathTest, RotMtoEulerIdentity) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Vector3d euler = RotMtoEuler(I);
    EXPECT_LT(euler.norm(), 1e-10);
}

TEST_F(SO3MathTest, RotMtoEulerConsistency) {
    // Create rotation from known Euler angles, convert back
    double roll = 0.3, pitch = 0.2, yaw = 0.5;
    
    // Build R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Eigen::Matrix3d Rx, Ry, Rz;
    Rx = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    Ry = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
    Rz = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d R = Rz * Ry * Rx;
    
    Eigen::Vector3d euler = RotMtoEuler(R);
    EXPECT_NEAR(euler(0), roll, 1e-10);
    EXPECT_NEAR(euler(1), pitch, 1e-10);
    EXPECT_NEAR(euler(2), yaw, 1e-10);
}

// ──────────────────────────────────────────────
// Algebraic property tests (important for GPU correctness)
// ──────────────────────────────────────────────

TEST_F(SO3MathTest, ExpAdditivityForSameAxis) {
    // For same axis: Exp(a*v) * Exp(b*v) = Exp((a+b)*v) 
    Eigen::Vector3d axis = Eigen::Vector3d(1, 2, 3).normalized();
    double a = 0.3, b = 0.5;
    
    Eigen::Matrix3d R1 = Exp(Eigen::Vector3d(a * axis)) * Exp(Eigen::Vector3d(b * axis));
    Eigen::Matrix3d R2 = Exp(Eigen::Vector3d((a + b) * axis));
    
    EXPECT_LT((R1 - R2).norm(), 1e-10);
}

TEST_F(SO3MathTest, ExpInverseIsNegative) {
    // Exp(v)^{-1} = Exp(-v)
    for (int i = 0; i < 50; i++) {
        Eigen::Vector3d v = randomVec();
        Eigen::Matrix3d R = Exp(Eigen::Vector3d(v));
        Eigen::Matrix3d R_inv = Exp(Eigen::Vector3d(-v));
        Eigen::Matrix3d product = R * R_inv;
        
        EXPECT_LT((product - Eigen::Matrix3d::Identity()).norm(), 1e-10);
    }
}

TEST_F(SO3MathTest, BCHFirstOrderApprox) {
    // For small angles: Exp(a) * Exp(b) ≈ Exp(a + b) (first order BCH)
    Eigen::Vector3d a(0.001, 0.002, -0.001);
    Eigen::Vector3d b(-0.002, 0.001, 0.003);
    
    Eigen::Matrix3d R_product = Exp(Eigen::Vector3d(a)) * Exp(Eigen::Vector3d(b));
    Eigen::Matrix3d R_sum = Exp(Eigen::Vector3d(a + b));
    
    // Should be very close for small angles
    EXPECT_LT((R_product - R_sum).norm(), 1e-5);
}
