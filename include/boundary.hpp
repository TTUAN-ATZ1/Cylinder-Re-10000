#pragma once

#include "parameters.hpp"
#include "fields.hpp"

inline void stream_and_bc(float &Cd, float &Cl)
{
    const float U0 = g_cfg.U0;

    // Lan truyen (Pull streaming song song)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            if (solid[idx(i, j)] == 1)
                continue;
            for (int k = 0; k < 9; ++k)
            {
                int prev_i = i - ex[k];
                int prev_j = j - ey[k];
                if (prev_i >= 0 && prev_i < NX && prev_j >= 0 && prev_j < NY)
                {
                    f[fidx(i, j, k)] = f_post[fidx(prev_i, prev_j, k)];
                }
            }
        }
    }

    // Bien cong Bouzidi & Do luc dong hoc Newton Ladd MEA
    float Fx = 0.0f;
    float Fy = 0.0f;

    int num_links = static_cast<int>(bouzidi_links.size());
#pragma omp parallel for reduction(+ : Fx, Fy)
    for (int link_idx = 0; link_idx < num_links; ++link_idx)
    {
        const auto &link = bouzidi_links[link_idx];
        int i = link.i;
        int j = link.j;
        int k = link.k;
        int k_opp = opp[k];
        float q = link.q;

        const float fk = f_post[fidx(i, j, k)];
        float f_refl = 0.0f;

        if (q < 0.5f)
        {
            // Nhanh q < 0.5: Noi suy huong k giua nut (i, j) va nut lui phia sau (ip, jp)
            int ip = i - ex[k];
            int jp = j - ey[k];
            if (ip >= 0 && ip < NX && jp >= 0 && jp < NY && solid[idx(ip, jp)] == 0)
            {
                f_refl = 2.0f * q * fk + (1.0f - 2.0f * q) * f_post[fidx(ip, jp, k)];
            }
            else
            {
                f_refl = fk; // Fallback: Halfway bounce-back an toan khi nut lui bi vuong solid
            }
        }
        else
        {
            // Nhanh q >= 0.5: Noi suy tai chinh nut (i, j) giua huong k va huong k_opp
            f_refl = (1.0f / (2.0f * q)) * fk + ((2.0f * q - 1.0f) / (2.0f * q)) * f_post[fidx(i, j, k_opp)];
        }

        f[fidx(i, j, k_opp)] = f_refl;

        // Trao doi dong luong Ladd MEA
        float mom = fk + f_refl;
        Fx += mom * static_cast<float>(ex[k]);
        Fy += mom * static_cast<float>(ey[k]);
    }

    // He so luc nang Cl, luc can Cd
    Cd = (2.0f * Fx) / (1.0f * U0 * U0 * B);
    Cl = (2.0f * Fy) / (1.0f * U0 * U0 * B);

    // Free-slip bien tren va duoi (Specular reflection triet tieu phan xa song am)
#pragma omp parallel for
    for (int i = 0; i < NX; ++i)
    {
        // Day: j = 0 (phan chieu 4->2, 7->6, 8->5)
        f[fidx(i, 0, 2)] = f[fidx(i, 0, 4)];
        f[fidx(i, 0, 5)] = f[fidx(i, 0, 8)];
        f[fidx(i, 0, 6)] = f[fidx(i, 0, 7)];

        // Dinh: j = NY - 1 (phan chieu 2->4, 5->8, 6->7)
        f[fidx(i, NY - 1, 4)] = f[fidx(i, NY - 1, 2)];
        f[fidx(i, NY - 1, 7)] = f[fidx(i, NY - 1, 6)];
        f[fidx(i, NY - 1, 8)] = f[fidx(i, NY - 1, 5)];
    }

    // Zou-He van toc Inlet (i = 0)
#pragma omp parallel for
    for (int j = 1; j < NY - 1; ++j)
    {
        float u_in = U0;
        float r_in = (f[fidx(0, j, 0)] + f[fidx(0, j, 2)] + f[fidx(0, j, 4)] +
                      2.0f * (f[fidx(0, j, 3)] + f[fidx(0, j, 6)] + f[fidx(0, j, 7)])) /
                     (1.0f - u_in);
        f[fidx(0, j, 1)] = f[fidx(0, j, 3)] + (2.0f / 3.0f) * r_in * u_in;
        f[fidx(0, j, 5)] = f[fidx(0, j, 7)] - 0.5f * (f[fidx(0, j, 2)] - f[fidx(0, j, 4)]) + (1.0f / 6.0f) * r_in * u_in;
        f[fidx(0, j, 8)] = f[fidx(0, j, 6)] + 0.5f * (f[fidx(0, j, 2)] - f[fidx(0, j, 4)]) + (1.0f / 6.0f) * r_in * u_in;
    }

    // Non-Reflecting Guo Outlet (i = NX - 1)
    float u_mean = 0.0f;
#pragma omp parallel for reduction(+ : u_mean)
    for (int j = 1; j < NY - 1; ++j)
    {
        float rn = 0.0f, sx = 0.0f;
        for (int k = 0; k < 9; ++k)
        {
            float v = f[fidx(NX - 2, j, k)];
            rn += v;
            sx += v * static_cast<float>(ex[k]);
        }
        u_mean += sx / rn;
    }
    u_mean /= static_cast<float>(NY - 2);

    // Tro khang song am phang huong ra: rho_b = 1.0 + sqrt(3) * (u_mean - U0)
    const float rho_b = 1.0f + 1.7320508f * (u_mean - U0);

#pragma omp parallel for
    for (int j = 1; j < NY - 1; ++j)
    {
        float rn = 0.0f, sx = 0.0f, sy = 0.0f;
        for (int k = 0; k < 9; ++k)
        {
            float val = f[fidx(NX - 2, j, k)];
            rn += val;
            sx += val * static_cast<float>(ex[k]);
            sy += val * static_cast<float>(ey[k]);
        }
        float un = sx / rn;
        float vn = sy / rn;

        for (int k = 0; k < 9; ++k)
        {
            f[fidx(NX - 1, j, k)] = feq(k, rho_b, un, vn) + (f[fidx(NX - 2, j, k)] - feq(k, rn, un, vn));
        }
    }

    // 4 goc bien
    for (int k = 0; k < 9; ++k)
    {
        f[fidx(0, 0, k)] = feq(k, 1.0f, U0, 0.0f);
        f[fidx(0, NY - 1, k)] = feq(k, 1.0f, U0, 0.0f);
        f[fidx(NX - 1, 0, k)] = f[fidx(NX - 2, 1, k)];
        f[fidx(NX - 1, NY - 1, k)] = f[fidx(NX - 2, NY - 2, k)];
    }
}
