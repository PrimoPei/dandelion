#include "rasterizer_renderer.h"
#include "../utils/math.hpp"
#include <cstdio>

#ifdef _WIN32
    #undef min
    #undef max
#endif

using Eigen::Vector3f;
using Eigen::Vector4f;

// vertex shader
VertexShaderPayload vertex_shader(const VertexShaderPayload& payload)
{
    VertexShaderPayload output_payload = payload;

    // Vertex position transformation
    Vector4f world_pos = Uniforms::inv_trans_M.inverse().transpose() * payload.world_position;
    output_payload.world_position = world_pos;

    // Viewport transformation
    Vector4f clip_pos = Uniforms::MVP * payload.world_position;
    Vector4f viewport_pos;
    viewport_pos.x() = (clip_pos.x() / clip_pos.w() + 1) * 0.5 * Uniforms::width;
    viewport_pos.y() = (clip_pos.y() / clip_pos.w() + 1) * 0.5 * Uniforms::height;
    viewport_pos.z() = clip_pos.z() / clip_pos.w();
    viewport_pos.w() = clip_pos.w();

    output_payload.viewport_position = viewport_pos;

    // Vertex normal transformation
    Eigen::Matrix3f m = Uniforms::inv_trans_M.topLeftCorner<3, 3>();
    Vector3f world_norm   = (m * payload.normal).normalized();
    output_payload.normal = world_norm;

    return output_payload;
}

Vector3f phong_fragment_shader(
    const FragmentShaderPayload& payload, const GL::Material& material,
    const std::list<Light>& lights, const Camera& camera
)
{

    Vector3f result = {0, 0, 0};

    // ka,kd,ks can be got from material.ambient,material.diffuse,material.specular

    // set ambient light intensity
    result += material.ambient;


    Vector3f norm = payload.world_normal.normalized();
    // View Direction
    Vector3f view_dir = (camera.position - payload.world_pos.head<3>()).normalized();

    for (const auto& light: lights) {
        // Light Direction
        Vector3f light_dir = (light.position - payload.world_pos.head<3>());
        float light_dis = light_dir.norm();
        light_dir = light_dir.normalized();

        // Half Vector
        Vector3f half_vec = (light_dir + view_dir).normalized();

        //Light Attenuation
        float attenuation = 1.0f / (light_dis * light_dis);

        //Diffuse
        Vector3f  diffuse = material.diffuse * light.intensity * std::max(0.0f, norm.dot(light_dir)) * attenuation;

        result += diffuse;
        //Specular
        Vector3f specular = material.specular * light.intensity * std::pow(std::max(0.0f, norm.dot(half_vec)), material.shininess) * attenuation;
        result += specular;
    }

    // set rendering result max threshold to 255

    result *= 255.f;
    result.x() = std::min(255.0f, result.x());
    result.y() = std::min(255.0f, result.y());
    result.z() = std::min(255.0f, result.z());

    return result;

}
