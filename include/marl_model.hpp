#pragma once

#include "marl_roi.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>

// ==============================================================================
// MANG NO-RON FCN ACTOR-CRITIC (PARAMETER SHARING CHO 18,000 AGENTS)
// ==============================================================================
// Kien truc Fully Convolutional (1x1 Conv):
// - Dau vao: 6 kenh vi mo [omega*, |u|/U0, Cp, Pixx, Piyy, Pixy]
// - Layer 1: 6 -> 64 (LeakyReLU)
// - Layer 2: 64 -> 64 (LeakyReLU)
// - Actor Head : 64 -> 1 (Sigmoid * alpha_max)
// - Critic Head: 64 -> 1 (Tuyen tinh -> Spatial Value V)
// Tong so tham so: ~4,739 weights (sieu nhe, < 20 KB)
// ==============================================================================

class MarlActorCritic
{
public:
    static constexpr int IN_DIM = 24; // 8 dac trung vi mo x 3 khung thoi gian (Frame Stacking)
    static constexpr int HIDDEN_DIM = 64;

    // Trong so va bias
    std::vector<float> W1; // [HIDDEN_DIM, IN_DIM] = 64 * 6
    std::vector<float> b1; // [HIDDEN_DIM] = 64
    std::vector<float> W2; // [HIDDEN_DIM, HIDDEN_DIM] = 64 * 64
    std::vector<float> b2; // [HIDDEN_DIM] = 64
    std::vector<float> W_actor;  // [1, HIDDEN_DIM] = 64
    float b_actor;
    std::vector<float> W_critic; // [1, HIDDEN_DIM] = 64
    float b_critic;
    float log_std; // Do lech chuan hanh dong (Gaussian Policy)

    // Adam Optimizer buffers
    struct AdamParam
    {
        std::vector<float> m, v;
        void init(size_t n) { m.assign(n, 0.0f); v.assign(n, 0.0f); }
    };
    AdamParam m_W1, m_b1, m_W2, m_b2, m_Wa, m_Wc;
    float m_ba = 0.0f, v_ba = 0.0f;
    float m_bc = 0.0f, v_bc = 0.0f;
    float m_std = 0.0f, v_std = 0.0f;
    int adam_step = 0;

    std::mt19937 rng;
    std::normal_distribution<float> normal_dist;

    MarlActorCritic() : rng(42), normal_dist(0.0f, 1.0f)
    {
        init_weights();
    }

    void init_weights()
    {
        W1.resize(HIDDEN_DIM * IN_DIM);
        b1.assign(HIDDEN_DIM, 0.0f);
        W2.resize(HIDDEN_DIM * HIDDEN_DIM);
        b2.assign(HIDDEN_DIM, 0.0f);
        W_actor.resize(HIDDEN_DIM);
        b_actor = 0.0f;
        W_critic.resize(HIDDEN_DIM);
        b_critic = 0.0f;
        log_std = -0.6f; // sigma ~ 0.548 (kham pha trong dai [-0.70, +0.80])

        // Khoi tao Xavier / Kaiming
        std::uniform_real_distribution<float> dist1(-std::sqrt(6.0f / (IN_DIM + HIDDEN_DIM)), std::sqrt(6.0f / (IN_DIM + HIDDEN_DIM)));
        for (auto &w : W1) w = dist1(rng);

        std::uniform_real_distribution<float> dist2(-std::sqrt(6.0f / (HIDDEN_DIM + HIDDEN_DIM)), std::sqrt(6.0f / (HIDDEN_DIM + HIDDEN_DIM)));
        for (auto &w : W2) w = dist2(rng);

        std::uniform_real_distribution<float> dist3(-0.05f, 0.05f);
        for (auto &w : W_actor) w = dist3(rng);
        for (auto &w : W_critic) w = dist3(rng);

        // Khoi tao bo dem Adam
        m_W1.init(W1.size()); m_b1.init(b1.size());
        m_W2.init(W2.size()); m_b2.init(b2.size());
        m_Wa.init(W_actor.size()); m_Wc.init(W_critic.size());
        adam_step = 0;
    }

    // Lan truyen xuoi (Forward pass) song song tren 18,000 agents
    void forward(const std::vector<float> &obs,
                 std::vector<float> &actions,
                 std::vector<float> &log_probs,
                 std::vector<float> &values,
                 bool deterministic = false,
                 float alpha_min = -0.50f,
                 float alpha_max = 3.50f)
    {
        actions.resize(ROI_AGENTS);
        log_probs.resize(ROI_AGENTS);
        values.resize(ROI_AGENTS);

        float std_val = std::exp(log_std);

#pragma omp parallel for
        for (int i = 0; i < ROI_AGENTS; ++i)
        {
            // 1. Layer 1: [24] -> [64]
            float h1[HIDDEN_DIM];
            for (int h = 0; h < HIDDEN_DIM; ++h)
            {
                float sum = b1[h];
                for (int c = 0; c < IN_DIM; ++c)
                {
                    sum += W1[h * IN_DIM + c] * obs[c * ROI_AGENTS + i];
                }
                h1[h] = (sum > 0.0f) ? sum : 0.1f * sum; // LeakyReLU
            }

            // 2. Layer 2: [64] -> [64]
            float h2[HIDDEN_DIM];
            for (int h = 0; h < HIDDEN_DIM; ++h)
            {
                float sum = b2[h];
                for (int p = 0; p < HIDDEN_DIM; ++p)
                {
                    sum += W2[h * HIDDEN_DIM + p] * h1[p];
                }
                h2[h] = (sum > 0.0f) ? sum : 0.1f * sum; // LeakyReLU
            }

            // 3. Actor Head: [64] -> [1] (Tanh -> dai dieu bien Dynamic Smagorinsky [-0.50, +3.50])
            float sum_act = b_actor;
            for (int h = 0; h < HIDDEN_DIM; ++h)
            {
                sum_act += W_actor[h] * h2[h];
            }
            float tanh_act = std::tanh(std::clamp(sum_act, -5.0f, 5.0f));
            float mean_act = 1.50f + 2.00f * tanh_act; // [-0.50, +3.50]

            // 4. Critic Head: [64] -> [1] (Spatial Value V)
            float sum_val = b_critic;
            for (int h = 0; h < HIDDEN_DIM; ++h)
            {
                sum_val += W_critic[h] * h2[h];
            }
            values[i] = sum_val;

            // 5. Lay mau hanh dong (Sample Action)
            if (deterministic)
            {
                actions[i] = mean_act;
                log_probs[i] = 0.0f;
            }
            else
            {
                // Thread-local Gaussian noise
                float noise = normal_dist(rng);
                float a = std::clamp(mean_act + std_val * noise, alpha_min, alpha_max);
                actions[i] = a;

                // Log probability Gaussian: -0.5 * ((a - mu)/std)^2 - log(std) - 0.5*log(2*pi)
                float diff = (a - mean_act) / (std_val + 1e-8f);
                log_probs[i] = -0.5f * diff * diff - log_std - 0.91893853f;
            }
        }
    }

    // Cap nhat trong so bang bo toi uu Adam dua tren gradient tich luy
    void apply_adam_gradients(const std::vector<float> &g_W1,
                              const std::vector<float> &g_b1,
                              const std::vector<float> &g_W2,
                              const std::vector<float> &g_b2,
                              const std::vector<float> &g_Wa,
                              float g_ba,
                              const std::vector<float> &g_Wc,
                              float g_bc,
                              float lr = 3e-4f)
    {
        adam_step++;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;

        auto step_vec = [&](std::vector<float> &param, AdamParam &adam, const std::vector<float> &grad) {
            for (size_t i = 0; i < param.size(); ++i)
            {
                float g = std::clamp(grad[i], -1.0f, 1.0f);
                adam.m[i] = beta1 * adam.m[i] + (1.0f - beta1) * g;
                adam.v[i] = beta2 * adam.v[i] + (1.0f - beta2) * (g * g);
                float m_hat = adam.m[i] / (1.0f - std::pow(beta1, adam_step));
                float v_hat = adam.v[i] / (1.0f - std::pow(beta2, adam_step));
                param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        };

        auto step_scalar = [&](float &param, float &m, float &v, float grad) {
            float g = std::clamp(grad, -1.0f, 1.0f);
            m = beta1 * m + (1.0f - beta1) * g;
            v = beta2 * v + (1.0f - beta2) * (g * g);
            float m_hat = m / (1.0f - std::pow(beta1, adam_step));
            float v_hat = v / (1.0f - std::pow(beta2, adam_step));
            param -= lr * m_hat / (std::sqrt(v_hat) + eps);
        };

        step_vec(W1, m_W1, g_W1);
        step_vec(b1, m_b1, g_b1);
        step_vec(W2, m_W2, g_W2);
        step_vec(b2, m_b2, g_b2);
        step_vec(W_actor, m_Wa, g_Wa);
        step_scalar(b_actor, m_ba, v_ba, g_ba);
        step_vec(W_critic, m_Wc, g_Wc);
        step_scalar(b_critic, m_bc, v_bc, g_bc);
    }

    // Luu trong so ra file nhi phan
    bool save_weights(const std::string &path = "marl_weights.bin") const
    {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        const char magic[8] = {'M', 'A', 'R', 'L', 'W', 'T', '0', '2'};
        out.write(magic, 8);
        out.write(reinterpret_cast<const char*>(W1.data()), W1.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(b1.data()), b1.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(W2.data()), W2.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(b2.data()), b2.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(W_actor.data()), W_actor.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(&b_actor), sizeof(float));
        out.write(reinterpret_cast<const char*>(W_critic.data()), W_critic.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(&b_critic), sizeof(float));
        out.write(reinterpret_cast<const char*>(&log_std), sizeof(float));
        out.close();
        return true;
    }

    // Doc trong so tu file nhi phan
    bool load_weights(const std::string &path = "marl_weights.bin")
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[8];
        in.read(magic, 8);
        if (std::string(magic, 8) != "MARLWT02") return false;
        in.read(reinterpret_cast<char*>(W1.data()), W1.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(W2.data()), W2.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(W_actor.data()), W_actor.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(&b_actor), sizeof(float));
        in.read(reinterpret_cast<char*>(W_critic.data()), W_critic.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(&b_critic), sizeof(float));
        in.read(reinterpret_cast<char*>(&log_std), sizeof(float));
        in.close();
        return true;
    }
};
