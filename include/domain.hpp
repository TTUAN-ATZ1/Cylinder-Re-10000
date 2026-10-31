#pragma once

#include "parameters.hpp"
#include "fields.hpp"
#include <cmath>

inline void init_domain()
{
    const float U0 = g_cfg.U0;

    // Khoi tao dong the (Potential Flow) xung quanh tru
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            int id = idx(i, j);
            float dx_c = (float)i - xc;
            float dy_c = (float)j - yc;
            float dist = std::sqrt(dx_c * dx_c + dy_c * dy_c);

            if (dist <= radius)
            {
                solid[id] = 1;
                ux[id] = 0.0f;
                uy[id] = 0.0f;
                rho[id] = 1.0f;
            }
            else
            {
                solid[id] = 0;
                float theta = std::atan2(dy_c, dx_c);
                float r2 = dist * dist;
                float R2 = radius * radius;
                ux[id] = U0 * (1.0f - (R2 / r2) * std::cos(2.0f * theta));
                uy[id] = -U0 * (R2 / r2) * std::sin(2.0f * theta);
                rho[id] = 1.0f;
            }

            P[id] = rho[id] * cs2;

            for (int k = 0; k < 9; ++k)
            {
                float eq = feq(k, rho[id], ux[id], uy[id]);
                f[fidx(i, j, k)] = eq;
                f_post[fidx(i, j, k)] = eq;
            }
        }
    }

    // Hang Inlet tai i = 0
    for (int j = 0; j < NY; ++j)
    {
        ux[idx(0, j)] = U0;
        uy[idx(0, j)] = 0.0f;
        rho[idx(0, j)] = 1.0f;
        P[idx(0, j)] = P_inf;
        for (int k = 0; k < 9; ++k)
        {
            float eq = feq(k, 1.0f, U0, 0.0f);
            f[fidx(0, j, k)] = eq;
            f_post[fidx(0, j, k)] = eq;
        }
    }

    // Tien xu ly lien ket bien cong Bouzidi
    bouzidi_links.clear();
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            if (solid[idx(i, j)] == 0)
            {
                for (int k = 1; k < 9; ++k)
                {
                    int ni = i + ex[k];
                    int nj = j + ey[k];
                    if (ni >= 0 && ni < NX && nj >= 0 && nj < NY)
                    {
                        if (solid[idx(ni, nj)] == 1)
                        {
                            float xf = (float)i;
                            float yf = (float)j;
                            float ex_k = (float)ex[k];
                            float ey_k = (float)ey[k];

                            float A = ex_k * ex_k + ey_k * ey_k;
                            float B_poly = 2.0f * ((xf - xc) * ex_k + (yf - yc) * ey_k);
                            float C_poly = (xf - xc) * (xf - xc) + (yf - yc) * (yf - yc) - radius * radius;

                            float delta = B_poly * B_poly - 4.0f * A * C_poly;
                            if (delta >= 0.0f)
                            {
                                float q = (-B_poly - std::sqrt(delta)) / (2.0f * A);
                                if (q < 0.001f)
                                    q = 0.001f;
                                if (q > 0.999f)
                                    q = 0.999f;
                                bouzidi_links.push_back({i, j, k, q});
                            }
                        }
                    }
                }
            }
        }
    }
}
