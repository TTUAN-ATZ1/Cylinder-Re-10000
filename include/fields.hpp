#pragma once

#include "parameters.hpp"
#include <vector>

// Noi suy Bien cong Bouzidi (Immersed Curved Boundary)
struct BouzidiLink
{
    int i, j; // Toa do nut chat long sat bien
    int k;    // Huong dong dam vao vat the
    float q;  // Khoang cach chuan hoa tu nut chat long toi bien cong [0, 1]
};

// Truong du lieu toan cuc
extern std::vector<BouzidiLink> bouzidi_links;
extern std::vector<float> f;
extern std::vector<float> f_post;
extern std::vector<float> rho;
extern std::vector<float> ux;
extern std::vector<float> uy;
extern std::vector<float> P;
extern std::vector<float> Cp;
extern std::vector<float> vort;
extern std::vector<int> solid;
extern std::vector<float> alpha_closure; // Dong hoc dong ho/Agent dong vien dong

// Ham can bang Maxwell-Boltzmann bac 2 chuan D2Q9
inline float feq(int k, float r, float u, float v)
{
    float eu = (float)ex[k] * u + (float)ey[k] * v;
    float uv = u * u + v * v;
    return r * w[k] * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * uv);
}
