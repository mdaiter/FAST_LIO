/**
 * Metal Compute Shaders for FAST-LIO GPU Acceleration
 *
 * 7 kernels matching the ComputeBackend interface. Float32 throughout;
 * double conversion happens on the CPU side for HTH/HTh final reduction.
 */

#include <metal_stdlib>
using namespace metal;

// ─── Shared types ────────────────────────────────────────────────────

struct TransformParams {
    float R_body[9];  // 3x3 col-major
    float t_body[3];
    float R_ext[9];   // 3x3 col-major
    float t_ext[3];
    uint  n;
};

struct PlaneFitParams {
    uint  n;
    uint  k;
    float threshold;
};

struct PlaneCoeffsGPU {
    float a, b, c, d;
    uint  valid;  // 0 or 1
};

struct ResidualParams { uint n; };

struct JacobianParams {
    float R_body[9];
    float R_ext[9];
    float t_ext[3];
    uint  m;
    uint  extrinsic_est_en;
};

struct HTHParams { uint m; };

struct UndistortParams {
    float R_end[9];
    float t_end[3];
    float R_ext[9];
    float t_ext[3];
    uint  n;
    uint  num_segments;
};

// ─── Linear-algebra helpers ──────────────────────────────────────────

// M*v  (column-major 3x3)
inline float3 mat3_mul(float m0, float m1, float m2,
                       float m3, float m4, float m5,
                       float m6, float m7, float m8, float3 v) {
    return float3(m0*v.x + m3*v.y + m6*v.z,
                  m1*v.x + m4*v.y + m7*v.z,
                  m2*v.x + m5*v.y + m8*v.z);
}

// M^T*v (column-major 3x3)
inline float3 mat3_tmul(float m0, float m1, float m2,
                        float m3, float m4, float m5,
                        float m6, float m7, float m8, float3 v) {
    return float3(m0*v.x + m1*v.y + m2*v.z,
                  m3*v.x + m4*v.y + m5*v.z,
                  m6*v.x + m7*v.y + m8*v.z);
}

// Address-space overloads
inline float3 mat3x3_mul(device const float* M, float3 v) {
    return mat3_mul(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}
inline float3 mat3x3_tmul(device const float* M, float3 v) {
    return mat3_tmul(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}
inline float3 mat3x3_mul(const constant float* M, float3 v) {
    return mat3_mul(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}
inline float3 mat3x3_tmul(const constant float* M, float3 v) {
    return mat3_tmul(M[0],M[1],M[2],M[3],M[4],M[5],M[6],M[7],M[8], v);
}

// a × v
inline float3 cross_mul(float3 a, float3 v) {
    return float3(a.y*v.z - a.z*v.y,
                  a.z*v.x - a.x*v.z,
                  a.x*v.y - a.y*v.x);
}

// Load column-major 3x3 from flat array into float3x3
inline float3x3 load_mat3(device const float* m) {
    return float3x3(float3(m[0],m[1],m[2]),
                    float3(m[3],m[4],m[5]),
                    float3(m[6],m[7],m[8]));
}
inline float3x3 load_mat3_const(const constant float* m) {
    return float3x3(float3(m[0],m[1],m[2]),
                    float3(m[3],m[4],m[5]),
                    float3(m[6],m[7],m[8]));
}

// Rodrigues: exp(w) for rotation vector w
inline float3x3 exp_so3(float3 w) {
    float theta = length(w);
    if (theta < 1e-6f) return float3x3(1,0,0, 0,1,0, 0,0,1);
    float3 ax = w / theta;
    float s = sin(theta), c = cos(theta), t = 1.0f - c;
    return float3x3(
        t*ax.x*ax.x + c,       t*ax.x*ax.y + s*ax.z, t*ax.x*ax.z - s*ax.y,
        t*ax.x*ax.y - s*ax.z,  t*ax.y*ax.y + c,      t*ax.y*ax.z + s*ax.x,
        t*ax.x*ax.z + s*ax.y,  t*ax.y*ax.z - s*ax.x, t*ax.z*ax.z + c);
}

// Mark a plane result as invalid
inline PlaneCoeffsGPU invalid_plane() {
    PlaneCoeffsGPU p; p.a=0; p.b=0; p.c=0; p.d=0; p.valid=0; return p;
}

// ─── Kernel 1: Batch point transformation ────────────────────────────
//   p_world = R_body * (R_ext * p_body + t_ext) + t_body

kernel void transform_points(
    device const float*      points_body  [[buffer(0)]],
    device       float*      points_world [[buffer(1)]],
    constant TransformParams& params      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    float3 pb = float3(points_body[tid*3], points_body[tid*3+1], points_body[tid*3+2]);
    float3 p_imu = mat3x3_mul(params.R_ext, pb)
                 + float3(params.t_ext[0], params.t_ext[1], params.t_ext[2]);
    float3 pw = mat3x3_mul(params.R_body, p_imu)
              + float3(params.t_body[0], params.t_body[1], params.t_body[2]);

    points_world[tid*3+0] = pw.x;
    points_world[tid*3+1] = pw.y;
    points_world[tid*3+2] = pw.z;
}

// ─── Kernel 2: Batch plane fitting ───────────────────────────────────
//   Normal equations (A^T*A) x = A^T*b  with coordinate pre-scaling
//   and Cholesky decomposition for numerical stability in float32.

kernel void plane_fit(
    device const float*          neighbors [[buffer(0)]],
    device       PlaneCoeffsGPU* planes    [[buffer(1)]],
    constant PlaneFitParams&     params    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    uint k = params.k;
    device const float* pts = neighbors + tid * k * 3;

    // Pre-scale coordinates to unit magnitude for float32 stability
    float max_abs = 0;
    for (uint j = 0; j < k; j++) {
        max_abs = max(max_abs, abs(pts[j*3+0]));
        max_abs = max(max_abs, abs(pts[j*3+1]));
        max_abs = max(max_abs, abs(pts[j*3+2]));
    }
    float scale = (max_abs > 1e-6f) ? (1.0f / max_abs) : 1.0f;

    // Build A^T*A (symmetric) and A^T*b on scaled coordinates
    float ata00=0, ata01=0, ata02=0, ata11=0, ata12=0, ata22=0;
    float atb0=0, atb1=0, atb2=0;
    for (uint j = 0; j < k; j++) {
        float x = pts[j*3+0]*scale, y = pts[j*3+1]*scale, z = pts[j*3+2]*scale;
        ata00 += x*x; ata01 += x*y; ata02 += x*z;
        ata11 += y*y; ata12 += y*z; ata22 += z*z;
        atb0 -= x; atb1 -= y; atb2 -= z;
    }

    // Cholesky: A^T*A = L*L^T
    float l00 = sqrt(ata00);
    if (l00 < 1e-7f) { planes[tid] = invalid_plane(); return; }
    float l00i = 1.0f / l00;
    float l10 = ata01 * l00i;
    float l20 = ata02 * l00i;

    float d1 = ata11 - l10*l10;
    if (d1 < 1e-14f) { planes[tid] = invalid_plane(); return; }
    float l11 = sqrt(d1);
    float l11i = 1.0f / l11;
    float l21 = (ata12 - l20*l10) * l11i;

    float d2 = ata22 - l20*l20 - l21*l21;
    if (d2 < 1e-14f) { planes[tid] = invalid_plane(); return; }
    float l22 = sqrt(d2);
    float l22i = 1.0f / l22;

    // Forward substitution: L*y = atb
    float y0 = atb0 * l00i;
    float y1 = (atb1 - l10*y0) * l11i;
    float y2 = (atb2 - l20*y0 - l21*y1) * l22i;

    // Back substitution: L^T*x = y
    float x2 = y2 * l22i;
    float x1 = (y1 - l21*x2) * l11i;
    float x0 = (y0 - l10*x1 - l20*x2) * l00i;

    // Unscale: normvec_orig = scale*(x0,x1,x2), d_orig = 1
    // After normalizing by ||normvec_orig|| = scale*||x||:
    //   a,b,c = x/||x||;  d = 1/(scale*||x||)
    float norm = sqrt(x0*x0 + x1*x1 + x2*x2);
    if (norm < 1e-10f) { planes[tid] = invalid_plane(); return; }

    PlaneCoeffsGPU result;
    result.a = x0 / norm;
    result.b = x1 / norm;
    result.c = x2 / norm;
    result.d = 1.0f / (norm * scale);

    // Validate: all neighbor residuals must be below threshold
    uint valid = 1;
    for (uint j = 0; j < k; j++) {
        float res = result.a*pts[j*3+0] + result.b*pts[j*3+1]
                  + result.c*pts[j*3+2] + result.d;
        if (abs(res) > params.threshold) { valid = 0; break; }
    }
    result.valid = valid;
    planes[tid] = result;
}

// ─── Kernel 3: Batch residual computation + validity filter ──────────
//   score = 1 - 0.9 * |residual| / sqrt(||p_body||);  valid if > 0.9

kernel void compute_residuals(
    device const float*          points_world [[buffer(0)]],
    device const float*          points_body  [[buffer(1)]],
    device const PlaneCoeffsGPU* planes       [[buffer(2)]],
    device       float*          residuals    [[buffer(3)]],
    device       uint8_t*        valid_mask   [[buffer(4)]],
    constant ResidualParams&     params       [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    if (planes[tid].valid == 0) {
        residuals[tid] = 0.0f;
        valid_mask[tid] = 0;
        return;
    }

    float3 pw = float3(points_world[tid*3], points_world[tid*3+1], points_world[tid*3+2]);
    float3 pb = float3(points_body[tid*3],  points_body[tid*3+1],  points_body[tid*3+2]);

    float pd2 = planes[tid].a*pw.x + planes[tid].b*pw.y
              + planes[tid].c*pw.z + planes[tid].d;
    float s = 1.0f - 0.9f * abs(pd2) / sqrt(length(pb));

    if (s > 0.9f) {
        residuals[tid] = pd2;
        valid_mask[tid] = 1;
    } else {
        residuals[tid] = 0.0f;
        valid_mask[tid] = 0;
    }
}

// ─── Kernel 4: Batch Jacobian construction ───────────────────────────
//   H row = [normal^T, A^T, B^T, C^T]  (1x12 float, converted to double on CPU)

kernel void build_jacobian(
    device const float*      points_body [[buffer(0)]],
    device const float*      normals     [[buffer(1)]],
    device const float*      plane_dists [[buffer(2)]],
    device       float*      H           [[buffer(3)]],
    device       float*      h           [[buffer(4)]],
    constant JacobianParams& params      [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.m) return;

    float3 pb   = float3(points_body[tid*3], points_body[tid*3+1], points_body[tid*3+2]);
    float3 norm = float3(normals[tid*3],     normals[tid*3+1],     normals[tid*3+2]);

    float3 point_imu = mat3x3_mul(params.R_ext, pb)
                     + float3(params.t_ext[0], params.t_ext[1], params.t_ext[2]);
    float3 C = mat3x3_tmul(params.R_body, norm);
    float3 A = cross_mul(point_imu, C);

    device float* row = H + tid * 12;
    row[0] = norm.x; row[1] = norm.y; row[2] = norm.z;
    row[3] = A.x;    row[4] = A.y;    row[5] = A.z;

    if (params.extrinsic_est_en) {
        float3 B = cross_mul(pb, mat3x3_tmul(params.R_ext, C));
        row[6]=B.x; row[7]=B.y; row[8]=B.z;
        row[9]=C.x; row[10]=C.y; row[11]=C.z;
    } else {
        row[6]=0; row[7]=0; row[8]=0;
        row[9]=0; row[10]=0; row[11]=0;
    }

    h[tid] = -plane_dists[tid];
}

// ─── Kernel 5: H^T*H partial reduction ──────────────────────────────
//   Each threadgroup reduces its rows into a 78-element upper triangle.
//   CPU does final reduction across threadgroups.

#define HTH_GROUP_SIZE 256
#define HTH_UPPER_SIZE 78   // 12*13/2

kernel void hth_partial(
    device const float* H        [[buffer(0)]],
    device       float* partials [[buffer(1)]],
    constant HTHParams& params   [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup float shared[HTH_GROUP_SIZE][12];

    if (tid < params.m) {
        for (uint c = 0; c < 12; c++) shared[lid][c] = H[tid*12+c];
    } else {
        for (uint c = 0; c < 12; c++) shared[lid][c] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threads 0..77 each compute one upper-triangle element
    if (lid < HTH_UPPER_SIZE) {
        uint idx = lid, i = 0, j = 0, cumsum = 0;
        for (uint r = 0; r < 12; r++) {
            uint row_len = 12 - r;
            if (idx < cumsum + row_len) { i = r; j = r + (idx - cumsum); break; }
            cumsum += row_len;
        }
        float sum = 0;
        for (uint t = 0; t < HTH_GROUP_SIZE; t++) sum += shared[t][i] * shared[t][j];
        partials[gid * HTH_UPPER_SIZE + lid] = sum;
    }
}

// ─── Kernel 6: H^T*h partial reduction ──────────────────────────────

kernel void hth_partial_vec(
    device const float* H        [[buffer(0)]],
    device const float* h_vec    [[buffer(1)]],
    device       float* partials [[buffer(2)]],
    constant HTHParams& params   [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup float shared_h[HTH_GROUP_SIZE];
    threadgroup float shared_H[HTH_GROUP_SIZE][12];

    if (tid < params.m) {
        shared_h[lid] = h_vec[tid];
        for (uint c = 0; c < 12; c++) shared_H[lid][c] = H[tid*12+c];
    } else {
        shared_h[lid] = 0;
        for (uint c = 0; c < 12; c++) shared_H[lid][c] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < 12) {
        float sum = 0;
        for (uint t = 0; t < HTH_GROUP_SIZE; t++) sum += shared_H[t][lid] * shared_h[t];
        partials[gid * 12 + lid] = sum;
    }
}

// ─── Fused superkernel: transform + plane_fit + residual + jacobian ──
//   Runs kernels 1-4 in a single dispatch with no intermediate buffers.
//   Output: N x 12 float H rows + N float h values (invalid → zero rows).
//   This eliminates 3 kernel launches + CPU compaction roundtrip.

struct FusedParams {
    float R_body[9];
    float t_body[3];
    float R_ext[9];
    float t_ext[3];
    uint  n;
    uint  k;
    float plane_threshold;
    uint  extrinsic_est_en;
};

// Inline: plane fit from neighbor pointer, returns (normal, d, valid)
inline void fit_plane(device const float* pts, uint k, float threshold,
                      thread float3& normal, thread float& d, thread bool& valid) {
    valid = false;
    normal = float3(0); d = 0;

    float max_abs = 0;
    for (uint j = 0; j < k; j++) {
        max_abs = max(max_abs, abs(pts[j*3+0]));
        max_abs = max(max_abs, abs(pts[j*3+1]));
        max_abs = max(max_abs, abs(pts[j*3+2]));
    }
    float scale = (max_abs > 1e-6f) ? (1.0f / max_abs) : 1.0f;

    float ata00=0,ata01=0,ata02=0,ata11=0,ata12=0,ata22=0;
    float atb0=0,atb1=0,atb2=0;
    for (uint j = 0; j < k; j++) {
        float x=pts[j*3+0]*scale, y=pts[j*3+1]*scale, z=pts[j*3+2]*scale;
        ata00+=x*x; ata01+=x*y; ata02+=x*z;
        ata11+=y*y; ata12+=y*z; ata22+=z*z;
        atb0-=x; atb1-=y; atb2-=z;
    }

    float l00=sqrt(ata00);
    if (l00<1e-7f) return;
    float l00i=1.0f/l00, l10=ata01*l00i, l20=ata02*l00i;
    float d1=ata11-l10*l10;
    if (d1<1e-14f) return;
    float l11=sqrt(d1), l11i=1.0f/l11, l21=(ata12-l20*l10)*l11i;
    float d2=ata22-l20*l20-l21*l21;
    if (d2<1e-14f) return;
    float l22=sqrt(d2), l22i=1.0f/l22;

    float y0=atb0*l00i;
    float y1=(atb1-l10*y0)*l11i;
    float y2=(atb2-l20*y0-l21*y1)*l22i;
    float x2=y2*l22i, x1=(y1-l21*x2)*l11i, x0=(y0-l10*x1-l20*x2)*l00i;

    float norm=sqrt(x0*x0+x1*x1+x2*x2);
    if (norm<1e-10f) return;

    normal = float3(x0,x1,x2)/norm;
    d = 1.0f/(norm*scale);

    for (uint j = 0; j < k; j++) {
        float res = normal.x*pts[j*3+0]+normal.y*pts[j*3+1]+normal.z*pts[j*3+2]+d;
        if (abs(res)>threshold) return;
    }
    valid = true;
}

kernel void fused_h_share(
    device const float*  points_body [[buffer(0)]],
    device const float*  neighbors   [[buffer(1)]],
    device       float*  H_out       [[buffer(2)]],  // N x 12
    device       float*  h_out       [[buffer(3)]],  // N
    constant FusedParams& params     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n) {
        // Zero out for threads beyond n (padding for threadgroup alignment)
        return;
    }

    device float* row = H_out + tid * 12;

    // ── Transform: pw = R_body * (R_ext * pb + t_ext) + t_body ──
    float3 pb = float3(points_body[tid*3], points_body[tid*3+1], points_body[tid*3+2]);
    float3 p_imu = mat3x3_mul(params.R_ext, pb)
                 + float3(params.t_ext[0], params.t_ext[1], params.t_ext[2]);
    float3 pw = mat3x3_mul(params.R_body, p_imu)
              + float3(params.t_body[0], params.t_body[1], params.t_body[2]);

    // ── Plane fit ──
    float3 normal; float d; bool plane_valid;
    fit_plane(neighbors + tid * params.k * 3, params.k, params.plane_threshold,
              normal, d, plane_valid);

    if (!plane_valid) {
        for (uint c = 0; c < 12; c++) row[c] = 0;
        h_out[tid] = 0;
        return;
    }

    // ── Residual + validity scoring ──
    float pd2 = normal.x*pw.x + normal.y*pw.y + normal.z*pw.z + d;
    float s = 1.0f - 0.9f * abs(pd2) / sqrt(length(pb));

    if (s <= 0.9f) {
        for (uint c = 0; c < 12; c++) row[c] = 0;
        h_out[tid] = 0;
        return;
    }

    // ── Jacobian: H row = [normal, A, B, C] ──
    float3 C = mat3x3_tmul(params.R_body, normal);
    float3 A = cross_mul(p_imu, C);

    row[0]=normal.x; row[1]=normal.y; row[2]=normal.z;
    row[3]=A.x; row[4]=A.y; row[5]=A.z;

    if (params.extrinsic_est_en) {
        float3 B = cross_mul(pb, mat3x3_tmul(params.R_ext, C));
        row[6]=B.x; row[7]=B.y; row[8]=B.z;
        row[9]=C.x; row[10]=C.y; row[11]=C.z;
    } else {
        row[6]=0; row[7]=0; row[8]=0;
        row[9]=0; row[10]=0; row[11]=0;
    }

    h_out[tid] = -pd2;
}

// ─── Combined HTH + HTh partial reduction ────────────────────────────
//   Fuses kernels 5+6: loads H and h once, produces both HTH (78 upper
//   triangle) and HTh (12) partials per threadgroup.

kernel void hth_combined_partial(
    device const float* H             [[buffer(0)]],
    device const float* h_vec         [[buffer(1)]],
    device       float* partials_hth  [[buffer(2)]],  // num_groups x 78
    device       float* partials_hth_vec [[buffer(3)]],  // num_groups x 12
    constant HTHParams& params        [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup float shared_H[HTH_GROUP_SIZE][12];
    threadgroup float shared_h[HTH_GROUP_SIZE];

    if (tid < params.m) {
        for (uint c = 0; c < 12; c++) shared_H[lid][c] = H[tid*12+c];
        shared_h[lid] = h_vec[tid];
    } else {
        for (uint c = 0; c < 12; c++) shared_H[lid][c] = 0;
        shared_h[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threads 0..77: HTH upper triangle
    if (lid < HTH_UPPER_SIZE) {
        uint idx = lid, i = 0, j = 0, cumsum = 0;
        for (uint r = 0; r < 12; r++) {
            uint row_len = 12 - r;
            if (idx < cumsum + row_len) { i = r; j = r + (idx - cumsum); break; }
            cumsum += row_len;
        }
        float sum = 0;
        for (uint t = 0; t < HTH_GROUP_SIZE; t++) sum += shared_H[t][i] * shared_H[t][j];
        partials_hth[gid * HTH_UPPER_SIZE + lid] = sum;
    }

    // Threads 0..11: HTh vector (runs in parallel with HTH on threads 0-11)
    if (lid < 12) {
        float sum = 0;
        for (uint t = 0; t < HTH_GROUP_SIZE; t++) sum += shared_H[t][lid] * shared_h[t];
        partials_hth_vec[gid * 12 + lid] = sum;
    }
}

// ─── Kernel 7: Batch point undistortion (IMU motion compensation) ────
//   P_comp = R_ext^T * (R_end^T * (R_i * (R_ext*P + t_ext) + T_ei) - t_ext)

kernel void undistort_points(
    device       float*       points      [[buffer(0)]],
    device const float*       timestamps  [[buffer(1)]],
    device const float*       seg_R       [[buffer(2)]],
    device const float*       seg_vel     [[buffer(3)]],
    device const float*       seg_pos     [[buffer(4)]],
    device const float*       seg_acc     [[buffer(5)]],
    device const float*       seg_angvel  [[buffer(6)]],
    device const float*       seg_t_start [[buffer(7)]],
    constant UndistortParams& params      [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.n) return;

    float t = timestamps[tid];

    // Find segment (linear scan — typically < 20 segments)
    int seg = (int)params.num_segments - 1;
    for (int s = seg; s >= 0; s--) {
        if (t >= seg_t_start[s]) { seg = s; break; }
    }
    float dt = t - seg_t_start[seg];

    float3x3 R_i = load_mat3(seg_R + seg*9)
                  * exp_so3(float3(seg_angvel[seg*3], seg_angvel[seg*3+1], seg_angvel[seg*3+2]) * dt);

    float3 pos = float3(seg_pos[seg*3], seg_pos[seg*3+1], seg_pos[seg*3+2]);
    float3 vel = float3(seg_vel[seg*3], seg_vel[seg*3+1], seg_vel[seg*3+2]);
    float3 acc = float3(seg_acc[seg*3], seg_acc[seg*3+1], seg_acc[seg*3+2]);
    float3 T_ei = pos + vel*dt + 0.5f*acc*dt*dt
                - float3(params.t_end[0], params.t_end[1], params.t_end[2]);

    float3x3 R_end_T = transpose(load_mat3_const(params.R_end));
    float3x3 R_ext   = load_mat3_const(params.R_ext);
    float3   t_ext   = float3(params.t_ext[0], params.t_ext[1], params.t_ext[2]);

    float3 P_i = float3(points[tid*3], points[tid*3+1], points[tid*3+2]);
    float3 P_comp = transpose(R_ext) * (R_end_T * (R_i * (R_ext*P_i + t_ext) + T_ei) - t_ext);

    points[tid*3+0] = P_comp.x;
    points[tid*3+1] = P_comp.y;
    points[tid*3+2] = P_comp.z;
}
