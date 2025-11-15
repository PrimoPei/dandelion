#include "bvh.h"

#include <cassert>
#include <iostream>
#include <optional>

#include <Eigen/Geometry>
#include "formatter.hpp"
#include <spdlog/spdlog.h>

#include "math.hpp"

using Eigen::Vector3f;
using std::optional;
using std::vector;

BVHNode::BVHNode() : left(nullptr), right(nullptr), face_idx(0)
{
}

BVH::BVH(const GL::Mesh& mesh) : root(nullptr), mesh(mesh)
{
}

// 建立bvh，将需要建立BVH的图元索引初始化
void BVH::build()
{
    if (mesh.faces.count() == 0) {
        root = nullptr;
        return;
    }

    primitives.resize(mesh.faces.count());
    for (size_t i = 0; i < mesh.faces.count(); i++) primitives[i] = i;

    root = recursively_build(primitives);
    return;
}

// 删除bvh
void BVH::recursively_delete(BVHNode* node)
{
    if (node == nullptr)
        return;
    recursively_delete(node->left);
    recursively_delete(node->right);
    delete node;
    node = nullptr;
}

// 统计BVH树建立的节点个数
size_t BVH::count_nodes(BVHNode* node)
{
    if (node == nullptr)
        return 0;
    else
        return count_nodes(node->left) + count_nodes(node->right) + 1;
}

// 递归建立BVH
BVHNode* BVH::recursively_build(vector<size_t> faces_idx)
{
    BVHNode* node = new BVHNode();

    AABB aabb;
    for (size_t i = 0; i < faces_idx.size(); i++) {
        aabb = union_AABB(aabb, get_aabb(mesh, faces_idx[i]));
    }
    node->aabb = aabb;
    // if faces_idx.size()==1: return node;
     if (faces_idx.size() == 1) {
            node->left = nullptr;
            node->right = nullptr;
            node->face_idx = faces_idx[0];
            return node;
        }
    // if faces_idx.size()==2: recursively_build() & union_AABB(node->left->aabb,
    // node->right->aabb);
    if (faces_idx.size() == 2) {
        node->left = recursively_build({faces_idx[0]});
        node->right = recursively_build({faces_idx[1]});
        return node;
    }
    //else:
    // choose the longest dimension among x,y,z
    int dim = aabb.max_extent();
    std::sort(faces_idx.begin(), faces_idx.end(), [&](size_t f1, size_t f2) {
        Vector3f c1 = get_aabb(mesh, f1).centroid();
        Vector3f c2 = get_aabb(mesh, f2).centroid();
        return c1[dim] < c2[dim];
    });

    // devide the primitives into two along the longest dimension
    auto mid = faces_idx.begin() + faces_idx.size() / 2;
    vector<size_t> l(faces_idx.begin(), mid);
    vector<size_t> r(mid, faces_idx.end());

    // recursively_build() & union_AABB(node->left->aabb, node->right->aabb)
    node->left = recursively_build(l);
    node->right = recursively_build(r);

    return node;
}

// 使用BVH求交
optional<Intersection> BVH::intersect(
    const Ray& ray, [[maybe_unused]] const GL::Mesh& mesh, const Eigen::Matrix4f obj_model
)
{
    model = obj_model;
    optional<Intersection> isect;
    if (!root) {
        isect = std::nullopt;
        return isect;
    }
    isect = ray_node_intersect(root, ray);
    return isect;
}

// 发射的射线与当前节点求交，并递归获取最终的求交结果
optional<Intersection> BVH::ray_node_intersect(BVHNode* node, const Ray& ray) const
{
    // The node intersection is performed in the model coordinate system.
    // Therefore, the ray needs to be transformed into the model coordinate system.
    // The intersection attributes returned are all in the model coordinate system.
    // Therefore, They are need to be converted to the world coordinate system.
    // If the model shrinks, the value of t will also change.
    // The change of t can be solved by intersection point changing simultaneously
    Eigen::Matrix4f inv_model = model.inverse();
    Ray model_ray;
    model_ray.origin = (inv_model * ray.origin.homogeneous()).head<3>();
    model_ray.direction = (inv_model.topLeftCorner<3, 3>() * ray.direction).normalized();

    Vector3f inv_dir(
            1.0f / model_ray.direction.x(),
            1.0f / model_ray.direction.y(),
            1.0f / model_ray.direction.z()
    );
    std::array<int, 3> dir_is_neg = {
            (inv_dir.x() < 0),
            (inv_dir.y() < 0),
            (inv_dir.z() < 0)
    };

    if (!node->aabb.intersect(model_ray, inv_dir, dir_is_neg)) {
        return std::nullopt;
    }

    if (node->left == nullptr && node->right == nullptr) {

        return ray_triangle_intersect(model_ray, mesh, node->face_idx);
    }

    auto hit_l = ray_node_intersect(node->left, ray);
    auto hit_r = ray_node_intersect(node->right, ray);


    if (hit_l.has_value() && hit_r.has_value()) {
        return (hit_l->t < hit_r->t) ? hit_l : hit_r;
    }
    if (hit_l.has_value()) {
        return hit_l;
    }
    if (hit_r.has_value()) {
        return hit_r;
    }
    return std::nullopt;







    // 2. 为 AABB::intersect 预计算优化参数


    // 3. --- BVH 核心逻辑 ---
    // (1) 检查射线是否击中当前节点的 AABB (在模型空间)

}
