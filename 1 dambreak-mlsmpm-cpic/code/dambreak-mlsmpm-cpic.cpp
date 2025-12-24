#define _CRT_SECURE_NO_WARNINGS
#include "taichi.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <omp.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

using namespace taichi;



// ==========================================
// 1. Definitions and constants
// ==========================================
using Vec = Vector3;
using Mat = Matrix3;

const int n = 64;           // Grid resolution
const real dt = 2e-4_f;     // Physical time step size
const real frame_dt = 1.0_f / 60.0_f; // Frame rate
const real dx = 1.0_f / n;  // Grid spacing
const real inv_dx = 1.0_f / dx;

// Fluid parameters
const real particle_mass = 1.0_f;
const real vol = 1.0_f; 
const real K = 50.0_f; 
const real gamma_val = 7.0_f; 

struct Particle {
    Vec x, v;
    Mat C;       
    real J;      
    int c;       
    int type;    // 0: Water
    int tag;     // CPIC tag: 1 or -1
    
    // Reconstructed SDF Information
    real sdf;    // Signed distance (negative inside, positive outside)
    Vec normal;  // Reconstructed surface normal
    bool near_body; // Denote particle affinity

    Particle(Vec x, int type = 0, int color = 0xFFFFFF) :
        x(x), v(Vec(0)), C(Mat(0)), J(1.0_f), type(type), c(color), tag(0), 
        sdf(1e10), normal(Vec(0,1,0)), near_body(false) {}
};

std::vector<Particle> particles;
Vector4 grid[n][n][n]; 
real grid_dist[n][n][n]; // Minimum unsigned distance to the rigid body surface
int  grid_tag[n][n][n];    // Grid tag (T_ir)  
bool grid_affinity[n][n][n]; // Grid affinity (A_ir)  
// In the original paper, an A_ir/T_ir pair is assigned to each oriented rigid surface r and each grid node i; here, this is simplified.




// ==========================================
// 2. Rigid body motion
// ==========================================
const Vec obs_size(0.05f, 0.1f, 0.05f); // Half length, width, height
const Vec motion_center(0.5f, 0.1f, 0.5f); // Revolution orbit center
const real motion_radius = 0.2f; // Revolution radius
const real orbit_speed = 15.0f;   // Revolution angular velocity (rad/s)
const real self_rotate_speed = 15.0f; // Self-rotation angular velocity (rad/s)
 
struct ObstacleState {
    Vec pos;      
    Vec v_linear; 
    Vec omega;    
    Mat R;        
};

// Define triangle primitive
struct RigidTriangle {
    Vec v[3]; // Three vertices in world coordinates
    Vec n;    // Surface normal
};


struct RigidParticle {
    Vec p;             // Position of sampled particle
    Vec n;             // Normal of sampled particle
    RigidTriangle tri; // Triangle primitive  
};

bool is_point_in_triangle(const Vec& p, const RigidTriangle& tri) {
    // Use Barycentric Technique to determine if point is inside triangle (https://blackpawn.com/texts/pointinpoly/)
    Vec v0 = tri.v[2] - tri.v[0];
    Vec v1 = tri.v[1] - tri.v[0];
    Vec v2 = p - tri.v[0];

    real dot00 = dot(v0, v0);
    real dot01 = dot(v0, v1);
    real dot02 = dot(v0, v2);
    real dot11 = dot(v1, v1);
    real dot12 = dot(v1, v2);

    real invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    real u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    real v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    const real eps = -1e-4f;
    return (u >= eps) && (v >= eps) && (u + v <= 1.0f - eps);
}

// Sample rigid particles in triangle surfaces
void sample_triangle(const Vec& a, const Vec& b, const Vec& c, 
                     std::vector<RigidParticle>& samples, real spacing) {
    Vec e1 = b - a;
    Vec e2 = c - a;
    Vec normal = normalized(cross(e1, e2));
    
    real len_e1 = e1.length();
    real len_e2 = e2.length();
    
    int steps_1 = std::max(1, (int)std::ceil(len_e1 / spacing));
    int steps_2 = std::max(1, (int)std::ceil(len_e2 / spacing));

    for (int i = 0; i <= steps_1; i++) {
        for (int j = 0; j <= steps_2; j++) {
            real u = (real)i / steps_1;
            real v = (real)j / steps_2;
            
            if (u + v <= 1.0f + 1e-4f) {
                Vec p = a + u * e1 + v * e2;
                
                RigidTriangle t;
                t.v[0] = a; t.v[1] = b; t.v[2] = c;
                t.n = normal;
                
                samples.push_back({p, normal, t});
            }
        }
    }
}

void get_obstacle_state(real time, ObstacleState &state) {
    real theta_orbit = orbit_speed * time;
    state.pos.x = motion_center.x + motion_radius * std::cos(theta_orbit);
    state.pos.z = motion_center.z + motion_radius * std::sin(theta_orbit);
    state.pos.y = motion_center.y + dx;  // Location of center of rigid body

    state.v_linear.x = -motion_radius * orbit_speed * std::sin(theta_orbit);
    state.v_linear.z =  motion_radius * orbit_speed * std::cos(theta_orbit);
    state.v_linear.y = 0.0f;

    real theta_self = self_rotate_speed * time;
    real c = std::cos(theta_self);
    real s = std::sin(theta_self);
    state.R = Matrix3(1.0_f);
    state.R(0, 0) = c; state.R(0, 2) = s;
    state.R(2, 0) = -s; state.R(2, 2) = c;
    state.omega = Vec(0, self_rotate_speed, 0); // Counterclockwise rotation
}

bool is_valid_idx(const Vector3i& idx) {
    return idx.x >= 0 && idx.x < n &&
           idx.y >= 0 && idx.y < n &&
           idx.z >= 0 && idx.z < n;
}




// ==========================================
// 3. Grid CDF realization
// ==========================================
void update_grid_dist(const ObstacleState &obs) {
    // 1. Reset grid
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                grid_dist[i][j][k] = 1e10_f;
                grid_affinity[i][j][k] = false;
            }
        }
    }

    // 2. Decompose the rigid body into 12 triangles 
    std::vector<RigidParticle> rigid_samples;
    real sample_step = dx * 0.5f;  
    
    // 8 vertices of the rigid body
    Vec vertices[8];
    for(int i=0; i<8; i++) {
        Vec local_v(
            (i & 1) ? obs_size.x : -obs_size.x,
            (i & 2) ? obs_size.y : -obs_size.y,
            (i & 4) ? obs_size.z : -obs_size.z
        );
        vertices[i] = obs.R * local_v + obs.pos; // Transform to world coordinates
    }

    int faces[6][4] = {
        {0, 4, 6, 2}, // x- face: (-1, 0, 0)
        {1, 3, 7, 5}, // x+ face: (+1, 0, 0)
        {0, 1, 5, 4}, // y- face: (0, -1, 0)
        {2, 6, 7, 3}, // y+ face: (0, +1, 0)
        {0, 2, 3, 1}, // z- face: (0, 0, -1)
        {4, 5, 7, 6}  // z+ face: (0, 0, +1)
    };

    // Generate triangles and sample
    for (int i = 0; i < 6; i++) {
        Vec v0 = vertices[faces[i][0]];
        Vec v1 = vertices[faces[i][1]];
        Vec v2 = vertices[faces[i][2]];
        Vec v3 = vertices[faces[i][3]];

        // Split each rectangular face into two triangles: (v0, v1, v2) and (v0, v2, v3)
        sample_triangle(v0, v1, v2, rigid_samples, sample_step);
        sample_triangle(v0, v2, v3, rigid_samples, sample_step);
    }

    // 3. Distance Splatting 
    for (const auto &rp : rigid_samples) {
        Vector3i base_node = (rp.p * inv_dx - Vec(0.5f)).cast<int>();

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    Vector3i node_idx = base_node + Vector3i(i, j, k);
                    if (!is_valid_idx(node_idx)) continue;  

                    Vec x_i = node_idx.cast<real>() * dx;
                    
                    // Point-Plane Distance  
                    real signed_dist = dot(x_i - rp.p, rp.n);
                    real abs_dist = std::abs(signed_dist);

                    // Projected Point on Plane
                    Vec proj_point = x_i - signed_dist * rp.n;

                    if (is_point_in_triangle(proj_point, rp.tri)) {
                        real current_min_dist = grid_dist[node_idx.x][node_idx.y][node_idx.z];

                        if (abs_dist < current_min_dist) {
                            grid_dist[node_idx.x][node_idx.y][node_idx.z] = abs_dist;
                            grid_tag[node_idx.x][node_idx.y][node_idx.z] = (signed_dist > 0 ? 1 : -1);
                        grid_affinity[node_idx.x][node_idx.y][node_idx.z] = true;
                        }
                    }
                }
            }
        }
    }
}



// ==========================================
// Particle Distance & Normal Reconstruction
// ==========================================
void reconstruct_particle_sdf(std::vector<Particle>& particles) {
    #pragma omp parallel for
    for (int i = 0; i < (int)particles.size(); i++) {
        auto& p = particles[i];
        
        // Reset state
        p.sdf = 10.0f;  
        p.normal = Vec(0, 0, 0); 
        p.near_body = false;

        // 1. Compute shape function weights and gradients
        Vector3i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();

        Vec w[3] = { 
            Vec(0.5) * sqr(Vec(1.5) - fx), 
            Vec(0.75) - sqr(fx - Vec(1.0)), 
            Vec(0.5) * sqr(fx - Vec(0.5)) 
        };

        // Quadratic B-Spline derivative dw  
        // d/dx (0.5*(1.5-x)^2) = x - 1.5
        // d/dx (0.75-(x-1)^2)  = -2*(x-1)
        // d/dx (0.5*(x-0.5)^2) = x - 0.5
        Vec dw[3] = {
            fx - Vec(1.5f),
            -2.0f * (fx - Vec(1.0f)),
            fx - Vec(0.5f)
        };

        real phi_val = 0.0f;       // Signed distance obtained from interpolation
        Vec phi_grad = Vec(0.0f);  // Interpolated distance field gradient (unnormalized normal)
        real weight_sum = 0.0f;    // Weight sum, used to check if in valid region

        // 2. Traverse neighborhood grid nodes
        for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
                for (int iz = 0; iz < 3; iz++) {
                    Vector3i node_idx = base_coord + Vector3i(ix, iy, iz);
                    
                    if (!is_valid_idx(node_idx)) continue;

                    // Reconstruct using only affinity nodes
                    if (grid_affinity[node_idx.x][node_idx.y][node_idx.z]) {
                        // Obtain Signed Distance
                        real grid_phi = grid_dist[node_idx.x][node_idx.y][node_idx.z] * (real)grid_tag[node_idx.x][node_idx.y][node_idx.z];

                        // Compute shape function weight N_i(xp)
                        real weight = w[ix].x * w[iy].y * w[iz].z;

                        // Compute shape function gradient \nabla N_i(xp)
                        // Gradient N = [dw.x * w.y * w.z, w.x * dw.y * w.z, w.x * w.y * dw.z] * inv_dx
                        Vec grad_weight(
                            dw[ix].x * w[iy].y * w[iz].z,
                            w[ix].x * dw[iy].y * w[iz].z,
                            w[ix].x * w[iy].y * dw[iz].z
                        );
                        grad_weight *= inv_dx; 

                        phi_val += weight * grid_phi;
                        phi_grad += grad_weight * grid_phi;
                        weight_sum += weight;
                    }
                }
            }
        }

        // 3. Result
        // If weight_sum > 0, then the particle lies within the coverage of the rigid body’s CDF  
        if (weight_sum > 1e-6f) {
            p.sdf = phi_val; 
            real len_sq = dot(phi_grad, phi_grad);
            if (len_sq > 1e-12f) {
                p.normal = phi_grad * (1.0f / std::sqrt(len_sq));
            }
            p.near_body = true;

            // Initialize Tag here
            // If the particle hasn't obtained Tag (tag == 0), then assign the tag according to the sign of the interpolated SDF
            if (p.tag == 0) {
                p.tag = (p.sdf >= 0) ? 1 : -1;
            }
        } else {
            // The situation where all nodes in p’s kernel lose affinities with r
            p.near_body = false;  
            p.tag = 0;           // Reset Tag, indicating complete forgetting
        }
    }
}




// ==========================================
// 4. Physical simulation with CPIC
// ==========================================
void advance(real dt, real current_time) {
    std::memset(grid, 0, sizeof(grid));

    ObstacleState obs;
    get_obstacle_state(current_time, obs);
    Mat R_inv = transposed(obs.R); 
    update_grid_dist(obs);
    reconstruct_particle_sdf(particles);

    // [P2G] 
    #pragma omp parallel for
    for (int i = 0; i < (int)particles.size(); i++) {
        auto& p = particles[i];
        Vector3i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        Vec w[3] = { Vec(0.5)*sqr(Vec(1.5)-fx), Vec(0.75)-sqr(fx-Vec(1.0)), Vec(0.5)*sqr(fx-Vec(0.5))};
 
        real effective_J = std::max(p.J, 0.1_f); 
        real pressure = K * (std::pow(effective_J, -gamma_val) - 1.0_f); 
        Mat stress = Mat(-pressure); 
        Mat affine = -(dt * vol) * (4 * inv_dx * inv_dx) * effective_J * stress + particle_mass * p.C; 

        for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
                for (int iz = 0; iz < 3; iz++) {
                    int idx_x = base_coord.x + ix, idx_y = base_coord.y + iy, idx_z = base_coord.z + iz;
                    if (idx_x < 0 || idx_x >= n || idx_y < 0 || idx_y >= n || idx_z < 0 || idx_z >= n) continue;

                    // CPIC compatibility check 
                    // Skip the transfer if a grid node is near the rigid body and lies on the opposite side from the particle 
                    if (grid_affinity[idx_x][idx_y][idx_z] && p.tag != grid_tag[idx_x][idx_y][idx_z]) continue;

                    Vec dpos = (Vec(ix, iy, iz) - fx) * dx;
                    Vector4 contrib(p.v * particle_mass + affine * dpos, particle_mass);
                    real weight = w[ix].x * w[iy].y * w[iz].z;

                    #pragma omp atomic
                    grid[idx_x][idx_y][idx_z].x += weight * contrib.x;
                    #pragma omp atomic
                    grid[idx_x][idx_y][idx_z].y += weight * contrib.y;
                    #pragma omp atomic
                    grid[idx_x][idx_y][idx_z].z += weight * contrib.z;
                    #pragma omp atomic
                    grid[idx_x][idx_y][idx_z].w += weight * contrib.w;
                }
            }
        }
    }

    // [Grid update] Boundary and collision handling
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                auto& g = grid[i][j][k];
                
                if (g.w > 1e-10_f) {
                    // 1. Gravity
                    g.x /= g.w; g.y /= g.w; g.z /= g.w;
                    g.y -= 9.8_f * dt;

                    // 2. Boundary condition
                    int b = 2;
                    if (i < b && g.x < 0) g.x = 0; if (i > n-b && g.x > 0) g.x = 0;
                    if (j < b && g.y < 0) g.y = 0; if (j > n-b && g.y > 0) g.y = 0;
                    if (k < b && g.z < 0) g.z = 0; if (k > n-b && g.z > 0) g.z = 0;

                    // ---------------------------------------------------------
                    // 3. Rigid body collision handling  
                    // ---------------------------------------------------------
                    real d = grid_dist[i][j][k];
                    real mu = 0.5f; // Dynamic friction coefficient

                    // [First-level filtering] Use the affinity to quickly skip grid nodes far from the rigid body
                    // Only process nodes within small distance (0.5dx) from the surface
                    if (grid_affinity[i][j][k] && d < dx * 0.5f) { 
                        Vec world_pos = Vec(i, j, k) * dx;
                        Vec rel_pos = world_pos - obs.pos;
                        Vec local_pos = R_inv * rel_pos; 

                        // [Second-level filtering] Verify that the node is inside the geometric bounds of the OBB
                        // Slightly relax the boundary (+dx) to prevent missed detections caused by numerical errors
                        if (std::abs(local_pos.x) < obs_size.x + dx &&
                            std::abs(local_pos.y) < obs_size.y + dx &&
                            std::abs(local_pos.z) < obs_size.z + dx) {

                            // A. Analytical Normal Calculation
                            // Find the face that is closest to this point
                            Vec d_min = Vec(obs_size.x - std::abs(local_pos.x),
                                            obs_size.y - std::abs(local_pos.y),
                                            obs_size.z - std::abs(local_pos.z));
                            
                            Vec local_n(0);
                            // Locate the minimum component of d_min, to determine the normal axis
                            if (d_min.x < d_min.y && d_min.x < d_min.z) {
                                local_n.x = (local_pos.x > 0) ? 1 : -1;
                            } else if (d_min.y < d_min.z) {
                                local_n.y = (local_pos.y > 0) ? 1 : -1;
                            } else {
                                local_n.z = (local_pos.z > 0) ? 1 : -1;
                            }

                            Vec world_n = obs.R * local_n;

                            // B. Velocity Projection
                            Vec v_body = obs.v_linear + cross(obs.omega, rel_pos);
                            Vec grid_v(g.x, g.y, g.z);
                            Vec v_rel = grid_v - v_body;
                            real v_n_proj = dot(v_rel, world_n);
 
                            if (v_n_proj < 0) {
                                Vec v_t = v_rel - world_n * v_n_proj;
                                real v_t_len = v_t.length();
                                Vec v_new_rel(0);
                                
                                if (v_t_len > 1e-6f) {
                                    real remaining_len = std::max(0.0f, v_t_len + mu * v_n_proj);
                                    v_new_rel = v_t * (remaining_len / v_t_len);
                                } 
                                
                                Vec v_new = v_new_rel + v_body;

                                g.x = v_new.x; 
                                g.y = v_new.y; 
                                g.z = v_new.z;
                            }
                        }
                    }
                }   
            }
        }
    }

    // [G2P]  
    #pragma omp parallel for
    for (int i = 0; i < (int)particles.size(); i++) {
        auto& p = particles[i];
        Vector3i base_coord = (p.x * inv_dx - Vec(0.5f)).cast<int>();
        Vec fx = p.x * inv_dx - base_coord.cast<real>();
        Vec w[3] = { Vec(0.5)*sqr(Vec(1.5)-fx), Vec(0.75)-sqr(fx-Vec(1.0)), Vec(0.5)*sqr(fx-Vec(0.5)) };
        
        // Preserve particle velocity as Ghost Velocity
        Vec v_ghost = p.v; 

        p.v = Vec(0); 
        p.C = Mat(0);

        for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
                for (int iz = 0; iz < 3; iz++) {
                    int idx_x = base_coord.x + ix, idx_y = base_coord.y + iy, idx_z = base_coord.z + iz;
                    if (idx_x < 0 || idx_x >= n || idx_y < 0 || idx_y >= n || idx_z < 0 || idx_z >= n) continue;

                    real weight = w[ix].x * w[iy].y * w[iz].z;
                    
                    // Check compatibility
                    bool is_compatible = true;
                    if (grid_affinity[idx_x][idx_y][idx_z]) {
                        if (p.tag != grid_tag[idx_x][idx_y][idx_z]) {
                            is_compatible = false;
                        }
                    }

                    Vec v_node_effective;

                    if (is_compatible) {
                        // A: Compatible node -> Read the updated grid velocity v_i^{n+1} 
                        auto& g = grid[idx_x][idx_y][idx_z];
                        v_node_effective = Vec(g.x, g.y, g.z);
                    } else {
                        // B: Imcompatible node -> use Ghost Velocity 
                        v_node_effective = v_ghost;
                    }

                    p.v += weight * v_node_effective;
                    p.C += 4 * inv_dx * Mat::outer_product(weight * v_node_effective, (Vec(ix, iy, iz) - fx));
                }
            }
        }

        p.x += dt * p.v;
        p.J *= (1.0_f + dt * (p.C[0][0] + p.C[1][1] + p.C[2][2]));
        if (p.type == 0) { 
            p.J = std::max(0.6_f, std::min(p.J, 20.0_f)); 
        }

        if (p.near_body && p.sdf < 0) {
            // Penalty Force: F = -k * depth * normal
            real penetration = -p.sdf;
            Vec penalty_force = p.normal * (2000.0f * penetration); // k = 2000
            
            p.v += dt * penalty_force / particle_mass;
            real alpha = 0.5f; 
            p.x += p.normal * (penetration * alpha);
        }

        // Boundary condition
        p.x.x = taichi::clamp(p.x.x, 1.0_f * dx, (n - 1.0_f) * dx);
        p.x.y = taichi::clamp(p.x.y, 1.0_f * dx, (n - 1.0_f) * dx);
        p.x.z = taichi::clamp(p.x.z, 1.0_f * dx, (n - 1.0_f) * dx);
    }
}

// ==========================================
// 5. Tool functions and main loop
// ==========================================
void add_box(Vec center, Vec size, int type, int color) {
    real step = dx * 0.5f; 
    for (real x = center.x - size.x; x < center.x + size.x; x += step) {
        for (real y = center.y - size.y; y < center.y + size.y; y += step) {
            for (real z = center.z - size.z; z < center.z + size.z; z += step) {
                particles.push_back(Particle(Vec(x, y, z), type, color));
            }
        }
    }
}

void save_ply(int frame, real current_time) {
    char buffer[100];
    sprintf(buffer, "ply_rotation/output_%04d.ply", frame);
    FILE* f = fopen(buffer, "w");
    if (!f) return;

    ObstacleState obs;
    get_obstacle_state(current_time, obs);
    
    std::vector<Particle> obs_viz;
    real step = dx * 0.5f; 
    for (real x = -obs_size.x; x <= obs_size.x; x += step) {
        for (real y = -obs_size.y; y <= obs_size.y; y += step) {
            for (real z = -obs_size.z; z <= obs_size.z; z += step) {
                Vec p_world = obs.R * Vec(x, y, z) + obs.pos;
                obs_viz.push_back(Particle(p_world, 1, 0xFFFF00));
            }
        }
    }

    fprintf(f, "ply\nformat ascii 1.0\nelement vertex %d\n", (int)(particles.size() + obs_viz.size()));
    fprintf(f, "property float x\nproperty float y\nproperty float z\n");
    fprintf(f, "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n");

    for (const auto& p : particles) fprintf(f, "%f %f %f %d %d %d\n", p.x.x, p.x.y, p.x.z, (p.c>>16)&0xFF, (p.c>>8)&0xFF, p.c&0xFF);
    for (const auto& p : obs_viz) fprintf(f, "%f %f %f %d %d %d\n", p.x.x, p.x.y, p.x.z, (p.c>>16)&0xFF, (p.c>>8)&0xFF, p.c&0xFF);
    fclose(f);
}

int main() {
#ifdef _WIN32
    _mkdir("ply_rotation");
#else
    mkdir("ply_rotation", 0777);
#endif
    // Create a water box
    add_box(Vec(0.2, 0.5, 0.2), Vec(0.15, 0.45, 0.15), 0, 0x00FF00);
    
    int frame = 0; real current_time = 0;
    while (frame < 300) {
        int substeps = (int)(frame_dt / dt);
        for (int i = 0; i < substeps; i++) {
            advance(dt, current_time);
            current_time += dt;
        }
        save_ply(frame++, current_time);
        printf("Frame %d done.\n", frame);
    }
    return 0;
}




// Limitations:
// 1. Only handles the single rigid body case and does not address simultaneous motion of multiple rigid bodies or rigid–rigid collisions.
// 2. Does not include two-way coupling between the fluid and the rigid body.
// 3. The rigid body is limited to a simple box shape and its motion is simplified to a circular orbit and self-rotation.







/* ==========================================
   Cross-Platform Compilation Commands (ensure that taichi.h is in the same directory)
==========================================

1. Windows (MinGW/GCC):
   g++ dambreak-mlsmpm-cpic.cpp -o dambreak-mlsmpm-cpic.exe -std=c++14 -O3 -lgdi32 -fopenmp

2. Linux (Ubuntu/Debian etc.):
   # Requires: sudo apt-get install libx11-dev
   g++ dambreak-mlsmpm-cpic.cpp -o dambreak-mlsmpm-cpic -std=c++14 -O3 -lX11 -lpthread -fopenmp

3. macOS (Clang):
   # Requires: brew install libomp
   g++ dambreak-mlsmpm-cpic.cpp -o dambreak-mlsmpm-cpic -std=c++14 -O3 -Xpreprocessor -fopenmp -lomp -framework Cocoa -framework CoreGraphics




.\dambreak-mlsmpm-cpic.exe   

   */

 