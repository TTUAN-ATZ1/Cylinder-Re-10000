#pragma once

#include "parameters.hpp"
#include "fields.hpp"
#include <cmath>
#include <string>
#include <sstream>

// Va cham LBM D2Q9 (Ho tro LBGK, LES Smagorinsky SGS, hoac Agent Closure)
inline void collide()
{
    const float tau_0 = g_cfg.tau;
    const float omega_0 = g_cfg.omega;
    const std::string &model = g_cfg.model;
    const float Cs2_factor = 18.0f * 1.41421356f * (g_cfg.smago_Cs * g_cfg.smago_Cs); // 18 * sqrt(2) * Cs^2

#pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            int id = idx(i, j);
            if (solid[id] == 1)
                continue;

            float r = rho[id];
            float u = ux[id];
            float v = uy[id];

            float feqs[9];
            for (int k = 0; k < 9; ++k)
            {
                feqs[k] = feq(k, r, u, v);
            }

            float omega_eff = omega_0;

            if (model == "smago")
            {
                // Tinh ung suat phi can bang Pi_neq
                float pixx = 0.0f, piyy = 0.0f, pixy = 0.0f;
                for (int k = 0; k < 9; ++k)
                {
                    float fneq = f[fidx(i, j, k)] - feqs[k];
                    pixx += fneq * static_cast<float>(ex[k] * ex[k]);
                    piyy += fneq * static_cast<float>(ey[k] * ey[k]);
                    pixy += fneq * static_cast<float>(ex[k] * ey[k]);
                }
                float Q = pixx * pixx + piyy * piyy + 2.0f * pixy * pixy;
                float S_mag = std::sqrt(Q) / r;
                float tau_eff = 0.5f * (std::sqrt(tau_0 * tau_0 + Cs2_factor * S_mag) + tau_0);
                omega_eff = 1.0f / tau_eff;
            }
            else if (model == "agent")
            {
                // Mo hinh Zonal MARL: Ket hop LES Smagorinsky nen voi Agent Closure cuc bo
                float pixx = 0.0f, piyy = 0.0f, pixy = 0.0f;
                for (int k = 0; k < 9; ++k)
                {
                    float fneq = f[fidx(i, j, k)] - feqs[k];
                    pixx += fneq * static_cast<float>(ex[k] * ex[k]);
                    piyy += fneq * static_cast<float>(ey[k] * ey[k]);
                    pixy += fneq * static_cast<float>(ex[k] * ey[k]);
                }
                float Q = pixx * pixx + piyy * piyy + 2.0f * pixy * pixy;
                float S_mag = std::sqrt(Q) / r;
                float tau_sgs = 0.5f * (std::sqrt(tau_0 * tau_0 + Cs2_factor * S_mag) - tau_0);

                // Trong vung ROI, Agent dieu bien alpha(x, y) [-0.70, +0.80] de bat dung vat ly tach dong
                float alpha = (!alpha_closure.empty()) ? alpha_closure[id] : 0.0f;
                float tau_eff = tau_0 + tau_sgs * (1.0f + alpha);
                if (tau_eff < 0.50005f)
                    tau_eff = 0.50005f;
                omega_eff = 1.0f / tau_eff;
            }

            for (int k = 0; k < 9; ++k)
            {
                int fid = fidx(i, j, k);
                f_post[fid] = f[fid] - omega_eff * (f[fid] - feqs[k]);
            }

            // Lop xop tieu am phi phan xa gan Inlet (Acoustic Absorbing Sponge Layer cho x in [0, 40])
            // Nuot tron 99% song am nguoc dong truoc khi dap vao bien Zou-He cung, triet tieu song dung
            if (i < 40)
            {
                float factor = static_cast<float>(40 - i) * 0.025f; // (40 - i) / 40.0f
                float sigma = 0.35f * factor * factor;
                for (int k = 0; k < 9; ++k)
                {
                    int fid = fidx(i, j, k);
                    float f_inflow = feq(k, 1.0f, g_cfg.U0, 0.0f);
                    f_post[fid] = (1.0f - sigma) * f_post[fid] + sigma * f_inflow;
                }
            }
        }
    }
}

// Cap nhat bien vi mo (rho, ux, uy, P)
inline void update_macroscopic()
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            int id = idx(i, j);
            if (solid[id] == 1)
            {
                rho[id] = 1.0f;
                ux[id] = 0.0f;
                uy[id] = 0.0f;
                P[id] = P_inf;
                continue;
            }

            float r = 0.0f, sx = 0.0f, sy = 0.0f;
            for (int k = 0; k < 9; ++k)
            {
                float val = f[fidx(i, j, k)];
                r += val;
                sx += val * static_cast<float>(ex[k]);
                sy += val * static_cast<float>(ey[k]);
            }
            rho[id] = r;
            ux[id] = sx / r;
            uy[id] = sy / r;
            P[id] = r * cs2;
        }
    }
}

// Tinh truong do xoay khong thu nguyen omega* va he so ap suat Cp
inline void compute_vorticity_and_cp()
{
    const float U0 = g_cfg.U0;
    const float q_dyn = g_cfg.q_dyn;

#pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            int id = idx(i, j);
            if (solid[id] == 1)
            {
                vort[id] = 0.0f;
                Cp[id] = 0.0f;
                continue;
            }

            // Dao ham dvy/dx
            float dv_dx;
            if (i == 0)
                dv_dx = (uy[idx(1, j)] - uy[idx(0, j)]);
            else if (i == NX - 1)
                dv_dx = (uy[idx(NX - 1, j)] - uy[idx(NX - 2, j)]);
            else
                dv_dx = 0.5f * (uy[idx(i + 1, j)] - uy[idx(i - 1, j)]);

            // Dao ham dux/dy
            float du_dy;
            if (j == 0)
                du_dy = (ux[idx(i, 1)] - ux[idx(i, 0)]);
            else if (j == NY - 1)
                du_dy = (ux[idx(i, NY - 1)] - ux[idx(i, NY - 2)]);
            else
                du_dy = 0.5f * (ux[idx(i, j + 1)] - ux[idx(i, j - 1)]);

            // Xoay khong thu nguyen omega* = (dv/dx - du/dy) * (D / U0)
            float omega_raw = dv_dx - du_dy;
            vort[id] = omega_raw * (B / U0);

            // He so ap suat khong thu nguyen Cp = (P - P_inf) / (0.5 * rho * U0^2)
            Cp[id] = (P[id] - P_inf) / q_dyn;
        }
    }
}

// Kiem tra do on dinh so hoc (Phat hien NaN / Inf / Blow-up)
inline bool check_numerical_stability(int step, float Cd, float Cl, std::string &diag_msg)
{
    // Kiem tra he so luc
    if (std::isnan(Cd) || std::isnan(Cl) || std::isinf(Cd) || std::isinf(Cl))
    {
        std::ostringstream ss;
        ss << "He so khi dong bi vo dinh: Cd=" << Cd << ", Cl=" << Cl;
        diag_msg = ss.str();
        return false;
    }

    // Kiem tra mot vai diem quan trong: diem truoc tru, sau tru, va dinh tru
    int test_points[] = {
        idx(static_cast<int>(xc - 2.0f * radius), static_cast<int>(yc)),
        idx(static_cast<int>(xc + 2.0f * radius), static_cast<int>(yc)),
        idx(static_cast<int>(xc), static_cast<int>(yc + 2.0f * radius)),
        idx(NX / 2, NY / 2)
    };

    for (int id : test_points)
    {
        float r = rho[id];
        float u = ux[id];
        float v = uy[id];

        if (std::isnan(r) || std::isnan(u) || std::isnan(v) ||
            std::isinf(r) || std::isinf(u) || std::isinf(v))
        {
            std::ostringstream ss;
            ss << "Phat hien gia tri NaN/Inf tai nut vi mo! rho=" << r << ", u=" << u << ", v=" << v;
            diag_msg = ss.str();
            return false;
        }

        // Kiem tra mat do thoat ly vat ly (0.1 < rho < 3.0) hoac van toc qua lon (u > 0.6)
        if (r < 0.1f || r > 3.0f || (u * u + v * v) > 0.36f)
        {
            std::ostringstream ss;
            ss << "Trang thai luu chat thoat ly gioi han vat ly: rho=" << r
               << ", |u|=" << std::sqrt(u * u + v * v) << " (gioi han am thanh cs=0.577)";
            diag_msg = ss.str();
            return false;
        }
    }

    return true;
}
