#pragma once

#include "parameters.hpp"
#include "fields.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

// ==============================================================================
// VUNG HOAT DONG MARL (ACTIVE REGION OF INTEREST - ROI)
// ==============================================================================
// Tru dat tai (xc = 315, yc = 250.05), duong kinh D = 30 cells (radius = 15)
// - X: Tu (xc - 1D) = 285 den (xc + 4D) = 435  --> Chieu rong W_ROI = 150 cells
// - Y: Tu (yc - 2D) = 190 den (yc + 2D) = 310  --> Chieu cao H_ROI = 120 cells
// Tong cong: 150 x 120 = 18,000 agents cuc bo
// ==============================================================================

constexpr int ROI_X_START = 285;
constexpr int ROI_X_END   = 435; // 150 cells
constexpr int ROI_Y_START = 190;
constexpr int ROI_Y_END   = 310; // 120 cells

constexpr int ROI_WIDTH   = ROI_X_END - ROI_X_START;  // 150
constexpr int ROI_HEIGHT  = ROI_Y_END - ROI_Y_START;  // 120
constexpr int ROI_AGENTS  = ROI_WIDTH * ROI_HEIGHT;   // 18,000

constexpr int BASE_CHANNELS  = 8;                              // 8 dac trung vi mo & pho da thang do
constexpr int STACK_FRAMES   = 3;                              // 3 khung thoi gian (t, t-1, t-2) de nhan biet pha xoay phi(t)
constexpr int NUM_CHANNELS   = BASE_CHANNELS * STACK_FRAMES;   // 24 kenh tong hop (24 x 18,000 = 432,000 floats)

struct MarlROI
{
    // Bo dem chua 8 dac trung tuc thoi Frame t: [8, H_ROI, W_ROI]
    std::vector<float> current_obs;

    // Bo dem chua tensor quan sat Frame Stacking: [24, H_ROI, W_ROI]
    // Flatten 1D: size = 24 * 120 * 150 = 432,000 floats
    std::vector<float> obs_buffer;
    bool history_initialized = false;

    MarlROI()
    {
        current_obs.resize(BASE_CHANNELS * ROI_AGENTS, 0.0f);
        obs_buffer.resize(NUM_CHANNELS * ROI_AGENTS, 0.0f);
        history_initialized = false;
        if (alpha_closure.size() != NX * NY)
        {
            alpha_closure.assign(NX * NY, 0.0f);
        }
    }

    // Dat lai bo nho lich su khung thoi gian khi bat dau Episode moi
    void reset_history()
    {
        history_initialized = false;
        obs_buffer.assign(NUM_CHANNELS * ROI_AGENTS, 0.0f);
    }

    // Trich xuat 8 kenh vi mo da thang do & Frame Stacking tu luoi LBM
    void extract_observation()
    {
        const float U0 = g_cfg.U0;
        const float q_dyn = g_cfg.q_dyn;

#pragma omp parallel for collapse(2)
        for (int ry = 0; ry < ROI_HEIGHT; ++ry)
        {
            for (int rx = 0; rx < ROI_WIDTH; ++rx)
            {
                int gx = ROI_X_START + rx;
                int gy = ROI_Y_START + ry;
                int gid = idx(gx, gy);
                int agent_id = ry * ROI_WIDTH + rx;

                // Neu la vat the (solid), dac trung = 0
                if (solid[gid] == 1)
                {
                    for (int c = 0; c < BASE_CHANNELS; ++c)
                    {
                        current_obs[c * ROI_AGENTS + agent_id] = 0.0f;
                    }
                    continue;
                }

                float r = rho[gid];
                float u = ux[gid];
                float v = uy[gid];

                // 1. Kenh 0: Do xoay omega* chuan hoa
                float omega = vort[gid];
                float omega_norm = omega * 0.1f; // Scale hop ly quanh [-1, 1]

                // 2. Kenh 1: Do lon van toc tuong doi |u| / U0
                float vel_mag = std::sqrt(u * u + v * v) / U0;

                // 3. Kenh 2: He so ap suat Cp
                float cp_val = Cp[gid];

                // 4-6. Kenh 3-5: Ung suat phi can bang Pi_neq / rho
                float pixx = 0.0f, piyy = 0.0f, pixy = 0.0f;
                for (int k = 0; k < 9; ++k)
                {
                    float eq = feq(k, r, u, v);
                    float fneq = f[fidx(gx, gy, k)] - eq;
                    pixx += fneq * static_cast<float>(ex[k] * ex[k]);
                    piyy += fneq * static_cast<float>(ey[k] * ey[k]);
                    pixy += fneq * static_cast<float>(ex[k] * ey[k]);
                }
                float pi_xx_norm = (pixx / r) * 10.0f;
                float pi_yy_norm = (piyy / r) * 10.0f;
                float pi_xy_norm = (pixy / r) * 10.0f;

                // 7. Kenh 6: Enstrophy vi mo cuc bo E = 0.5 * omega^2
                float enstrophy = 0.5f * omega * omega * 0.05f;

                // 8. Kenh 7: Loc da thang do 3x3 va Do lech pho Duoi-luoi (Subgrid Vorticity Deficit)
                // Test filter 3x3 kieu Germano:
                float w_N  = vort[idx(gx, std::min(gy + 1, NY - 1))];
                float w_S  = vort[idx(gx, std::max(gy - 1, 0))];
                float w_E  = vort[idx(std::min(gx + 1, NX - 1), gy)];
                float w_W  = vort[idx(std::max(gx - 1, 0), gy)];
                float w_NE = vort[idx(std::min(gx + 1, NX - 1), std::min(gy + 1, NY - 1))];
                float w_NW = vort[idx(std::max(gx - 1, 0), std::min(gy + 1, NY - 1))];
                float w_SE = vort[idx(std::min(gx + 1, NX - 1), std::max(gy - 1, 0))];
                float w_SW = vort[idx(std::max(gx - 1, 0), std::max(gy - 1, 0))];

                float omega_filtered = (4.0f * omega + 2.0f * (w_N + w_S + w_E + w_W) + (w_NE + w_NW + w_SE + w_SW)) / 16.0f;
                float subgrid_deficit = (omega - omega_filtered) * 0.2f;

                current_obs[0 * ROI_AGENTS + agent_id] = std::clamp(omega_norm, -3.0f, 3.0f);
                current_obs[1 * ROI_AGENTS + agent_id] = std::clamp(vel_mag, 0.0f, 3.0f);
                current_obs[2 * ROI_AGENTS + agent_id] = std::clamp(cp_val, -3.0f, 2.0f);
                current_obs[3 * ROI_AGENTS + agent_id] = std::clamp(pi_xx_norm, -3.0f, 3.0f);
                current_obs[4 * ROI_AGENTS + agent_id] = std::clamp(pi_yy_norm, -3.0f, 3.0f);
                current_obs[5 * ROI_AGENTS + agent_id] = std::clamp(pi_xy_norm, -3.0f, 3.0f);
                current_obs[6 * ROI_AGENTS + agent_id] = std::clamp(enstrophy, 0.0f, 3.0f);
                current_obs[7 * ROI_AGENTS + agent_id] = std::clamp(subgrid_deficit, -3.0f, 3.0f);
            }
        }

        // Frame Stacking: Day khung thoi gian (t-2 <- t-1, t-1 <- t, t <- current_obs)
        const int FRAME_SIZE = BASE_CHANNELS * ROI_AGENTS; // 8 * 18,000 = 144,000 floats

        if (!history_initialized)
        {
            // Khoi tao: Dien deu current_obs vao ca 3 khung de tranh bat lien tuc
            for (int f_idx = 0; f_idx < STACK_FRAMES; ++f_idx)
            {
                std::copy(current_obs.begin(), current_obs.end(), obs_buffer.begin() + f_idx * FRAME_SIZE);
            }
            history_initialized = true;
        }
        else
        {
            // Dich Frame 1 sang Frame 2
            std::copy(obs_buffer.begin() + 1 * FRAME_SIZE, obs_buffer.begin() + 2 * FRAME_SIZE, obs_buffer.begin() + 2 * FRAME_SIZE);
            // Dich Frame 0 sang Frame 1
            std::copy(obs_buffer.begin() + 0 * FRAME_SIZE, obs_buffer.begin() + 1 * FRAME_SIZE, obs_buffer.begin() + 1 * FRAME_SIZE);
            // Nap Frame 0 moi tu current_obs
            std::copy(current_obs.begin(), current_obs.end(), obs_buffer.begin() + 0 * FRAME_SIZE);
        }
    }

    // Anh xa hanh dong alpha [-0.50, +3.50] tu Agent vao mang alpha_closure (Dynamic Smagorinsky Germano-Lilly)
    void apply_actions(const std::vector<float> &actions, float alpha_min = -0.50f, float alpha_max = 3.50f)
    {
#pragma omp parallel for collapse(2)
        for (int ry = 0; ry < ROI_HEIGHT; ++ry)
        {
            for (int rx = 0; rx < ROI_WIDTH; ++rx)
            {
                int gx = ROI_X_START + rx;
                int gy = ROI_Y_START + ry;
                int gid = idx(gx, gy);
                int agent_id = ry * ROI_WIDTH + rx;

                if (solid[gid] == 1)
                {
                    alpha_closure[gid] = 0.0f;
                    continue;
                }

                float a = actions[agent_id];
                // Kep hanh dong trong dai an toan vat ly
                alpha_closure[gid] = std::clamp(a, alpha_min, alpha_max);
            }
        }
    }
};
