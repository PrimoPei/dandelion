#include <array>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <spdlog/spdlog.h>

#include "rasterizer.h"
#include "triangle.h"
#include "../utils/math.hpp"

using Eigen::Matrix4f;
using Eigen::Vector2i;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::fill;
using std::tuple;

void Rasterizer::worker_thread()
{
    while (!Context::rasterizer_finish) {
        VertexShaderPayload payload;
        Triangle            triangle;
        {
            if (Context::vertex_finish && Context::vertex_shader_output_queue.empty()) {
                Context::rasterizer_finish = true;
                return;
            }
            if (Context::vertex_shader_output_queue.size() < 3) {
                continue;
            }
            std::unique_lock<std::mutex> lock(Context::vertex_queue_mutex);
            if (Context::vertex_shader_output_queue.size() < 3) {
                continue;
            }
            for (size_t vertex_count = 0; vertex_count < 3; vertex_count++) {
                payload = Context::vertex_shader_output_queue.front();
                Context::vertex_shader_output_queue.pop();
                if (vertex_count == 0) {
                    triangle.world_pos[0]    = payload.world_position;
                    triangle.viewport_pos[0] = payload.viewport_position;
                    triangle.normal[0]       = payload.normal;
                } else if (vertex_count == 1) {
                    triangle.world_pos[1]    = payload.world_position;
                    triangle.viewport_pos[1] = payload.viewport_position;
                    triangle.normal[1]       = payload.normal;
                } else {
                    triangle.world_pos[2]    = payload.world_position;
                    triangle.viewport_pos[2] = payload.viewport_position;
                    triangle.normal[2]       = payload.normal;
                }
            }
        }
        rasterize_triangle(triangle);
    }
}

float sign(Eigen::Vector2f p1, Eigen::Vector2f p2, Eigen::Vector2f p3)
{
    return (p1.x() - p3.x()) * (p2.y() - p3.y()) - (p2.x() - p3.x()) * (p1.y() - p3.y());
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，判断(x,y)是否在三角形的内部
bool Rasterizer::inside_triangle(int x, int y, const Vector4f* vertices)
{
    Vector3f v[3];
    for (int i = 0; i < 3; i++) v[i] = {vertices[i].x(), vertices[i].y(), 1.0};

    Vector3f p(float(x), float(y), 1.0f);
    float s1 = sign(p.head<2>(), v[0].head<2>(), v[1].head<2>());
    float s2 = sign(p.head<2>(), v[1].head<2>(), v[2].head<2>());
    float s3 = sign(p.head<2>(), v[2].head<2>(), v[0].head<2>());


    return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
}

// 给定坐标(x,y)以及三角形的三个顶点坐标，计算(x,y)对应的重心坐标[alpha, beta, gamma]
tuple<float, float, float> Rasterizer::compute_barycentric_2d(float x, float y, const Vector4f* v)
{
    float c1 = 0.f, c2 = 0.f, c3 = 0.f;

    Eigen::Vector2f p(x, y);

    float total = sign(v[1].head<2>(), v[2].head<2>(), v[0].head<2>());

    if (std::abs(total) < 1e-9f)
    {
        return {1.0f, 0.0f, 0.0f};
    }

    c1 = sign(v[1].head<2>(), v[2].head<2>(), p) / total;
    c2 = sign(v[2].head<2>(), v[0].head<2>(), p) / total;
    c3 = sign(v[0].head<2>(), v[1].head<2>(), p) / total;
    return {c1, c2, c3};
}

// 对顶点的某一属性插值
Vector3f Rasterizer::interpolate(
    float alpha, float beta, float gamma, const Eigen::Vector3f& vert1,
    const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, const Eigen::Vector3f& weight,
    const float& Z
)
{
    Vector3f interpolated_res;
    for (int i = 0; i < 3; i++) {
        interpolated_res[i] = alpha * vert1[i] / weight[0] + beta * vert2[i] / weight[1]
                            + gamma * vert3[i] / weight[2];
    }
    interpolated_res *= Z;
    return interpolated_res;
}

// 对当前三角形进行光栅化
void Rasterizer::rasterize_triangle(Triangle& t)
{
    float minx = std::min({t.viewport_pos[0].x(), t.viewport_pos[1].x(), t.viewport_pos[2].x()});
    float maxx = std::max({t.viewport_pos[0].x(), t.viewport_pos[1].x(), t.viewport_pos[2].x()});
    float miny = std::min({t.viewport_pos[0].y(), t.viewport_pos[1].y(), t.viewport_pos[2].y()});
    float maxy = std::max({t.viewport_pos[0].y(), t.viewport_pos[1].y(), t.viewport_pos[2].y()});

    int x_start = std::max(0, (int)std::floor(minx));
    int x_end   = std::min(Uniforms::width - 1, (int)std::ceil(maxx));
    int y_start = std::max(0, (int)std::floor(miny));
    int y_end   = std::min(Uniforms::height - 1, (int)std::ceil(maxy));

    if (t.viewport_pos[0].w() < 1e-5f ||
        t.viewport_pos[1].w() < 1e-5f ||
        t.viewport_pos[2].w() < 1e-5f)
    {
        return;
    }


    for (int y = y_start; y <= y_end; y++) {
        for (int x = x_start; x <= x_end; x++) {
            // if current pixel is in current triange:
            if (!inside_triangle(x, y, t.viewport_pos)) {
                continue;
            }
            // 1. interpolate depth(use projection correction algorithm)
            auto [alpha, beta, gamma] = compute_barycentric_2d(float(x), float(y), t.viewport_pos);
            float zt = 1.0f / (alpha / t.viewport_pos[0].w() + beta / t.viewport_pos[1].w() + gamma / t.viewport_pos[2].w());
            float it = (alpha * t.viewport_pos[0].z() / t.viewport_pos[0].w() +
                                    beta * t.viewport_pos[1].z() / t.viewport_pos[1].w() +
                                    gamma * t.viewport_pos[2].z() / t.viewport_pos[2].w()) * zt;
            
            int index = (Uniforms::height - 1 - y) * Uniforms::width + x; 
            if (it < Context::frame_buffer.depth_buffer[index]) {
                Context::frame_buffer.depth_buffer[index] = it;

                // 2. interpolate vertex positon & normal(use function:interpolate())
                FragmentShaderPayload payload;
                payload.x = x;
                payload.y = y;
                payload.depth = it;
                Eigen::Vector3f weights(t.viewport_pos[0].w(), t.viewport_pos[1].w(), t.viewport_pos[2].w());
                payload.world_pos = interpolate(alpha, beta, gamma, t.world_pos[0].head<3>(), t.world_pos[1].head<3>(), t.world_pos[2].head<3>(), weights, zt);
                payload.world_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], weights, zt);
                payload.world_normal = payload.world_normal.normalized();

                // 3. push primitive into fragment queue
                std::unique_lock<std::mutex> lock(Context::rasterizer_queue_mutex);
                Context::rasterizer_output_queue.push(payload);
            }
        }
    }
}
