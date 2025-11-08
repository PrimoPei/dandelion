#include "ray.h"

#include <cmath>
#include <array>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include "../utils/math.hpp"

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::numeric_limits;
using std::optional;
using std::size_t;

constexpr float infinity = 1e5f;
constexpr float eps      = 1e-5f;

Intersection::Intersection() : t(numeric_limits<float>::infinity()), face_index(0)
{
}

Ray generate_ray(int width, int height, int x, int y, Camera& camera, float depth)
{
    // The ratio between the specified plane (width x height)'s depth and the image plane's depth.
    Vector2f pos((float)x + 0.5f, (float)y + 0.5f);
    Vector2f center((float)width / 2.0f, (float)height / 2.0f);
    Matrix4f inv_view = camera.view().inverse();
    float fov_y_rad = radians(camera.fov_y_degrees);
    float image_plane_height = 2.0f * std::tan(fov_y_rad * 0.5f);
    float ratio = image_plane_height / (float)height;

    Vector4f view_pos((pos.x() - center.x()) * ratio, -(pos.y() - center.y()) * ratio, -depth, 1.0f);

    // Transfer the view-space position to world space.
    Vector3f world_pos = (inv_view * view_pos).head<3>();

    return {camera.position, (world_pos - camera.position).normalized()};

}

optional<Intersection> ray_triangle_intersect(const Ray& ray, const GL::Mesh& mesh, size_t index)
{
    Vector3f v0 = mesh.vertex(mesh.face(index)[0]);
    Vector3f v1 = mesh.vertex(mesh.face(index)[1]);
    Vector3f v2 = mesh.vertex(mesh.face(index)[2]);

    Vector3f e1 = v1 - v0;
    Vector3f e2 = v2 - v0;


    Vector3f x = ray.direction.cross(e2);
    float det = e1.dot(x);

    if (std::abs(det) < eps) {
        return std::nullopt;
    }

    float inv_det = 1.0f / det;

    Vector3f a = ray.origin - v0;

    float u = (a.dot(x)) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return std::nullopt;
    }

    Vector3f y = a.cross(e1);
    float v = (ray.direction.dot(y)) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        return std::nullopt;
    }

    float dis = (e2.dot(y)) * inv_det;

    if (dis < eps) {
        return std::nullopt;
    }

    Intersection result;
    result.t = dis;
    result.face_index = index;

    result.barycentric_coord = Vector3f(1.0f - u - v, u, v);

    Vector3f n0 = mesh.normal(mesh.face(index)[0]);
    Vector3f n1 = mesh.normal(mesh.face(index)[1]);
    Vector3f n2 = mesh.normal(mesh.face(index)[2]);
    result.normal = (result.barycentric_coord[0] * n0 + result.barycentric_coord[1] * n1 + result.barycentric_coord[2] * n2).normalized();

    if (result.t - infinity < -eps) {
        return result;
    } else {
        return std::nullopt;
    }

}

optional<Intersection> naive_intersect(const Ray& ray, const GL::Mesh& mesh, const Matrix4f model)
{
    Eigen::Matrix4f inv_model = model.inverse();
    Ray model_ray;
    model_ray.origin = (inv_model * ray.origin.homogeneous()).head<3>();
    model_ray.direction = (inv_model.topLeftCorner<3, 3>() * ray.direction).normalized();


    optional<Intersection> result = std::nullopt;
    float closest_t = std::numeric_limits<float>::infinity();


    for (size_t i = 0; i < mesh.faces.count(); i++)
    {
        optional<Intersection> intersection = ray_triangle_intersect(model_ray, mesh, i);
        if (intersection && intersection->t < closest_t)
        {
            closest_t = intersection->t;
            result = intersection;
        }
    }


    if (result)
    {
        Eigen::Matrix3f inv_trans_model = inv_model.transpose().block<3, 3>(0, 0);
        result->normal = (inv_trans_model * result->normal).normalized();
    }

    if (result->t - infinity < -eps) {
        return result;
    }
    return std::nullopt;
}
