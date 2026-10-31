#pragma once

#include "parameters.hpp"
#include "fields.hpp"
#include "domain.hpp"
#include "solver.hpp"
#include "boundary.hpp"
#include "checkpoint.hpp"
#include "marl_roi.hpp"
#include "marl_reward.hpp"
#include "marl_model.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <filesystem>

class MarlTrainer
{
public:
    MarlROI roi;
    MarlRewardEngine reward_engine;
    MarlActorCritic model;

    const int STEP_FACTOR = 10;        // 10 buoc LBM / 1 hanh dong cua Agent
    const int MACRO_STEPS = 300;       // 300 macro steps = 3,000 buoc LBM / 1 Episode (~1.6 chu ky xoay)
    const float GAMMA = 0.99f;         // Discount factor
    const float GAE_LAMBDA = 0.95f;    // GAE lambda
    const float CLIP_EPS = 0.2f;       // PPO clip epsilon
    const float LEARNING_RATE = 5e-4f; // Adam learning rate
    const int PPO_EPOCHS = 4;          // 4 epochs cap nhat PPO cho moi Episode (chuan Schulman et al. 2017)

    const int WARMUP_STEPS = 1500;     // Khoi dong am tai buoc 1500: lop bien da phat trien hoan hao, Cd o dai chuan 1.15-1.24
    std::string warmup_path = "warmup_re10000.bin";

    MarlTrainer()
    {
        if (std::filesystem::exists("best_marl_weights.bin"))
        {
            if (model.load_weights("best_marl_weights.bin"))
            {
                std::cout << " [MARL TRAINER] Da nap trong so toi uu tu best_marl_weights.bin de tiep tuc toi uu hoa!\n";
            }
        }
    }

    // Tao file khoi dong am (Warm-up Checkpoint) tai buoc 5000 neu chua co hoac sai cau hinh
    void ensure_warmup_checkpoint()
    {
        bool need_generate = true;
        if (std::filesystem::exists(warmup_path))
        {
            std::ifstream in(warmup_path, std::ios::binary);
            if (in)
            {
                char magic[8];
                in.read(magic, 8);
                int nx = 0, ny = 0, step = 0;
                in.read(reinterpret_cast<char *>(&nx), sizeof(nx));
                in.read(reinterpret_cast<char *>(&ny), sizeof(ny));
                in.read(reinterpret_cast<char *>(&step), sizeof(step));
                if (std::string(magic, 8) == "LBMCHK01" && nx == NX && ny == NY && step == WARMUP_STEPS)
                {
                    std::cout << " [WARM-UP] Da ton tai file khoi dong am hop le (" << nx << "x" << ny << ", buoc " << step << "): " << warmup_path << "\n";
                    need_generate = false;
                }
                else
                {
                    std::cout << " [WARM-UP] Checkpoint cu (buoc " << step << ", " << nx << "x" << ny << ") khong khop voi cau hinh moi (buoc " << WARMUP_STEPS << ", " << NX << "x" << NY << "). Dang tao lai...\n";
                }
            }
        }

        if (!need_generate)
            return;

        std::cout << "\n====================================================================\n";
        std::cout << " [WARM-UP] DANG TAO KHOI DONG AM TAI Re=10000 (" << WARMUP_STEPS << " buoc LBM bang LES)...\n";
        std::cout << "====================================================================\n";

        g_cfg.Re = 10000.0f;
        g_cfg.model = "smago";
        g_cfg.update_physics();

        init_domain();
        float Cd = 0.0f, Cl = 0.0f;

        for (int step = 0; step < WARMUP_STEPS; ++step)
        {
            collide();
            stream_and_bc(Cd, Cl);
            update_macroscopic();

            if (step % 500 == 0)
            {
                std::cout << "  - Buoc khoi dong " << step << "/" << WARMUP_STEPS << " | Cd=" << Cd << " | Cl=" << Cl << "\n";
            }
        }

        save_checkpoint(WARMUP_STEPS, WARMUP_STEPS * g_cfg.dt_phy, Cd, Cl, warmup_path);
        std::cout << " [HOAN TAT WARM-UP] Da luu checkpoint khoi dong am vao: " << warmup_path << "\n\n";
    }

    // Chay 1 Episode huan luyen
    float run_episode(int ep_idx, bool train = true)
    {
        // 1. Reset ve trang thai khoi dong am
        int start_step = WARMUP_STEPS;
        float t_phy = 0.0f, Cd = 0.0f, Cl = 0.0f;
        load_checkpoint(start_step, t_phy, Cd, Cl, warmup_path);

        g_cfg.model = "agent"; // Chuyen sang che do Agent
        g_cfg.update_physics();

        roi.reset_history(); // Khoi tao lai Frame Stacking buffer cho Episode moi

        std::vector<std::vector<float>> ep_states;
        std::vector<std::vector<float>> ep_actions;
        std::vector<std::vector<float>> ep_log_probs;
        std::vector<float> ep_rewards;
        std::vector<float> ep_mean_values;

        float total_episode_reward = 0.0f;
        float sum_cd = 0.0f;
        int cd_count = 0;
        bool unstable = false;

        for (int m = 0; m < MACRO_STEPS; ++m)
        {
            // A. Trich xuat quan sat
            compute_vorticity_and_cp();
            roi.extract_observation();

            // B. Mạng Actor-Critic suy luận hành động
            std::vector<float> actions;
            std::vector<float> log_probs;
            std::vector<float> values;
            model.forward(roi.obs_buffer, actions, log_probs, values, !train);

            // Tinh gia tri ky vong trung binh toan vung ROI
            float mean_v = 0.0f;
            for (float v : values)
                mean_v += v;
            mean_v /= static_cast<float>(ROI_AGENTS);

            // C. Ap dung hanh dong alpha(x, y) vao vung ROI trong dai Dynamic Smagorinsky [-0.50, +3.50]
            roi.apply_actions(actions, -0.50f, 3.50f);

            // D. Chay STEP_FACTOR buoc LBM
            for (int sub = 0; sub < STEP_FACTOR; ++sub)
            {
                collide();
                stream_and_bc(Cd, Cl);
                update_macroscopic();

                sum_cd += Cd;
                cd_count++;
            }

            // E. Kiem tra on dinh
            std::string diag;
            if (!check_numerical_stability(start_step + m * STEP_FACTOR, Cd, Cl, diag))
            {
                unstable = true;
            }

            // F. Tinh phan thuong (bao gom ca Cd, Cl vuot chuan 3D va Ap suat mat tru)
            compute_vorticity_and_cp();
            float reward = reward_engine.compute_step_reward(Cd, Cl, unstable);
            total_episode_reward += reward;

            if (train)
            {
                ep_states.push_back(roi.obs_buffer);
                ep_actions.push_back(actions);
                ep_log_probs.push_back(log_probs);
                ep_rewards.push_back(reward);
                ep_mean_values.push_back(mean_v);
            }

            if (unstable)
                break;
        }

        float avg_cd = (cd_count > 0) ? (sum_cd / cd_count) : 0.0f;

        // Neu dang huan luyen va khong no so, thuc hien cap nhat PPO Policy
        if (train && !unstable && ep_rewards.size() > 10)
        {
            update_ppo(ep_states, ep_actions, ep_log_probs, ep_rewards, ep_mean_values);
        }

        std::cout << std::fixed;
        std::cout << "Episode " << std::setw(3) << ep_idx
                  << " | Thuong: " << std::setw(8) << std::setprecision(2) << total_episode_reward
                  << " | Cd_tb: " << std::setw(6) << std::setprecision(3) << avg_cd
                  << " | Cl_cuoi: " << (Cl >= 0 ? "+" : "") << std::setw(6) << std::setprecision(3) << Cl
                  << " | Trang thai: " << (unstable ? "[NO SO]" : "[ON DINH]") << "\n"
                  << std::flush;

        return total_episode_reward;
    }

    // Cap nhat trong so mang Actor-Critic bang PPO
    void update_ppo(const std::vector<std::vector<float>> &states,
                    const std::vector<std::vector<float>> &actions,
                    const std::vector<std::vector<float>> &old_log_probs,
                    const std::vector<float> &rewards,
                    const std::vector<float> &values)
    {
        int T = static_cast<int>(rewards.size());
        if (T < 2)
            return;

        // 1. Tinh Returns va GAE Advantages
        std::vector<float> advantages(T, 0.0f);
        std::vector<float> returns(T, 0.0f);

        float gae = 0.0f;
        for (int t = T - 1; t >= 0; --t)
        {
            float next_val = (t == T - 1) ? 0.0f : values[t + 1];
            float delta = rewards[t] + GAMMA * next_val - values[t];
            gae = delta + GAMMA * GAE_LAMBDA * gae;
            advantages[t] = gae;
            returns[t] = advantages[t] + values[t];
        }

        // Chuan hoa Advantages
        float mean_adv = 0.0f, var_adv = 0.0f;
        for (float a : advantages)
            mean_adv += a;
        mean_adv /= static_cast<float>(T);
        for (float a : advantages)
            var_adv += (a - mean_adv) * (a - mean_adv);
        float std_adv = std::sqrt(var_adv / static_cast<float>(T) + 1e-8f);

        for (int t = 0; t < T; ++t)
        {
            advantages[t] = (advantages[t] - mean_adv) / std_adv;
        }

        // 2. Chay K PPO Epochs de toi uu hoa chinh sach (PPO Multi-Epoch Optimization, Schulman 2017)
        const int STRIDE = 6; // Lay mau 3,000 agents moi buoc thoi gian (300 x 3,000 = 900,000 diem mau)

        for (int epoch = 0; epoch < PPO_EPOCHS; ++epoch)
        {
            std::vector<float> g_W1(model.W1.size(), 0.0f);
            std::vector<float> g_b1(model.b1.size(), 0.0f);
            std::vector<float> g_W2(model.W2.size(), 0.0f);
            std::vector<float> g_b2(model.b2.size(), 0.0f);
            std::vector<float> g_Wa(model.W_actor.size(), 0.0f);
            float total_g_ba = 0.0f;
            std::vector<float> g_Wc(model.W_critic.size(), 0.0f);
            float total_g_bc = 0.0f;

            const float std_val = std::exp(model.log_std);
            const float var_val = std_val * std_val + 1e-8f;
            int sample_count = 0;

#pragma omp parallel
            {
                std::vector<float> loc_W1(model.W1.size(), 0.0f);
                std::vector<float> loc_b1(model.b1.size(), 0.0f);
                std::vector<float> loc_W2(model.W2.size(), 0.0f);
                std::vector<float> loc_b2(model.b2.size(), 0.0f);
                std::vector<float> loc_Wa(model.W_actor.size(), 0.0f);
                float loc_ba = 0.0f;
                std::vector<float> loc_Wc(model.W_critic.size(), 0.0f);
                float loc_bc = 0.0f;
                int loc_count = 0;

#pragma omp for schedule(static)
                for (int t = 0; t < T; ++t)
                {
                    float adv = advantages[t];
                    float ret = returns[t];

                    for (int i = 0; i < ROI_AGENTS; i += STRIDE)
                    {
                        // A. Forward pass agent i tai buoc t
                        float x[MarlActorCritic::IN_DIM];
                        for (int c = 0; c < MarlActorCritic::IN_DIM; ++c)
                        {
                            x[c] = states[t][c * ROI_AGENTS + i];
                        }

                        // Layer 1
                        float h1[MarlActorCritic::HIDDEN_DIM];
                        for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                        {
                            float sum = model.b1[h];
                            for (int c = 0; c < MarlActorCritic::IN_DIM; ++c)
                            {
                                sum += model.W1[h * MarlActorCritic::IN_DIM + c] * x[c];
                            }
                            h1[h] = (sum > 0.0f) ? sum : 0.1f * sum;
                        }

                        // Layer 2
                        float h2[MarlActorCritic::HIDDEN_DIM];
                        for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                        {
                            float sum = model.b2[h];
                            for (int p = 0; p < MarlActorCritic::HIDDEN_DIM; ++p)
                            {
                                sum += model.W2[h * MarlActorCritic::HIDDEN_DIM + p] * h1[p];
                            }
                            h2[h] = (sum > 0.0f) ? sum : 0.1f * sum;
                        }

                        // Actor Head
                        float sum_act = model.b_actor;
                        for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                        {
                            sum_act += model.W_actor[h] * h2[h];
                        }
                        float tanh_act = std::tanh(std::clamp(sum_act, -5.0f, 5.0f));
                        float mean_act = 1.50f + 2.00f * tanh_act; // [-0.50, +3.50]

                        // Critic Head
                        float sum_val = model.b_critic;
                        for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                        {
                            sum_val += model.W_critic[h] * h2[h];
                        }

                        // B. Critic Gradient: MSE Loss = (V - ret)^2
                        float v_err = sum_val - ret;
                        float gz_critic = std::clamp(v_err, -3.0f, 3.0f);

                        // C. Actor PPO Gradient
                        float a = actions[t][i];
                        float old_lp = old_log_probs[t][i];
                        float diff = (a - mean_act) / (std_val + 1e-8f);
                        float new_lp = -0.5f * diff * diff - model.log_std - 0.91893853f;
                        float ratio = std::exp(std::clamp(new_lp - old_lp, -10.0f, 10.0f));

                        // PPO Clip Gradient
                        float pg_weight = adv;
                        if ((ratio > 1.0f + CLIP_EPS && adv > 0.0f) ||
                            (ratio < 1.0f - CLIP_EPS && adv < 0.0f))
                        {
                            pg_weight = 0.0f; // Triet tieu gradient ngoai vung kẹp
                        }

                        // dJ / dz_act = pg_weight * ratio * ((a - mean_act)/sigma^2) * (2.00 * (1 - tanh^2))
                        float dmu_dza = 2.00f * (1.0f - tanh_act * tanh_act);
                        float dlp_dmu = (a - mean_act) / var_val;
                        float gz_actor = -pg_weight * ratio * dlp_dmu * dmu_dza; // Dau tru vi Gradient Descent (toi thieu hoa -J)
                        gz_actor = std::clamp(gz_actor, -2.0f, 2.0f);

                        // D. Tich luy gradient cho Heads
                        loc_ba += gz_actor;
                        loc_bc += gz_critic;
                        for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                        {
                            loc_Wa[h] += gz_actor * h2[h];
                            loc_Wc[h] += gz_critic * h2[h];
                        }

                        // E. Backprop sang Layer 2
                        float gh2[MarlActorCritic::HIDDEN_DIM];
                        float gz2[MarlActorCritic::HIDDEN_DIM];
                        for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                        {
                            gh2[h] = gz_actor * model.W_actor[h] + gz_critic * model.W_critic[h] * 0.1f;
                            gz2[h] = (h2[h] > 0.0f) ? gh2[h] : 0.1f * gh2[h];
                            loc_b2[h] += gz2[h];
                        }

                        for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                        {
                            for (int p = 0; p < MarlActorCritic::HIDDEN_DIM; ++p)
                            {
                                loc_W2[h * MarlActorCritic::HIDDEN_DIM + p] += gz2[h] * h1[p];
                            }
                        }

                        // F. Backprop sang Layer 1
                        for (int p = 0; p < MarlActorCritic::HIDDEN_DIM; ++p)
                        {
                            float gh1_p = 0.0f;
                            for (int h = 0; h < MarlActorCritic::HIDDEN_DIM; ++h)
                            {
                                gh1_p += gz2[h] * model.W2[h * MarlActorCritic::HIDDEN_DIM + p];
                            }
                            float gz1_p = (h1[p] > 0.0f) ? gh1_p : 0.1f * gh1_p;
                            loc_b1[p] += gz1_p;
                            for (int c = 0; c < MarlActorCritic::IN_DIM; ++c)
                            {
                                loc_W1[p * MarlActorCritic::IN_DIM + c] += gz1_p * x[c];
                            }
                        }

                        loc_count++;
                    }
                }

#pragma omp critical
                {
                    sample_count += loc_count;
                    total_g_ba += loc_ba;
                    total_g_bc += loc_bc;
                    for (size_t k = 0; k < g_W1.size(); ++k) g_W1[k] += loc_W1[k];
                    for (size_t k = 0; k < g_b1.size(); ++k) g_b1[k] += loc_b1[k];
                    for (size_t k = 0; k < g_W2.size(); ++k) g_W2[k] += loc_W2[k];
                    for (size_t k = 0; k < g_b2.size(); ++k) g_b2[k] += loc_b2[k];
                    for (size_t k = 0; k < g_Wa.size(); ++k) g_Wa[k] += loc_Wa[k];
                    for (size_t k = 0; k < g_Wc.size(); ++k) g_Wc[k] += loc_Wc[k];
                }
            }

            // Chuan hoa gradient theo tong so mau
            if (sample_count > 0)
            {
                float inv_n = 1.0f / static_cast<float>(sample_count);
                for (auto &g : g_W1) g *= inv_n;
                for (auto &g : g_b1) g *= inv_n;
                for (auto &g : g_W2) g *= inv_n;
                for (auto &g : g_b2) g *= inv_n;
                for (auto &g : g_Wa) g *= inv_n;
                total_g_ba *= inv_n;
                for (auto &g : g_Wc) g *= inv_n;
                total_g_bc *= inv_n;
            }

            // Cap nhat Adam cho tat ca cac tang mang
            model.apply_adam_gradients(g_W1, g_b1, g_W2, g_b2, g_Wa, total_g_ba, g_Wc, total_g_bc, LEARNING_RATE);
        }

        // Cap nhat do lech chuan log_std giam dan tu tu de giam kham pha
        model.log_std = std::clamp(model.log_std - 0.001f, -1.2f, -0.4f);
    }

    // Vong lap huan luyen tong the N Episodes
    void train(int total_episodes = 50)
    {
        std::cout << "\n====================================================================\n";
        std::cout << "        BAT DAU TIEN TRINH HUAN LUYEN MARL CHO TRU TRON Re=10000    \n";
        std::cout << "====================================================================\n";
        std::cout << "  - So Episodes huan luyen  : " << total_episodes << "\n";
        std::cout << "  - So Agent cuc bo         : " << ROI_AGENTS << " agents (150x120)\n";
        std::cout << "  - Thuat toan RL           : Proximal Policy Optimization (PPO)\n";
        std::cout << "  - Muc tieu vat ly         : Cd = 1.143, Lr/D = 0.82, On dinh luoi tho\n";
        std::cout << "====================================================================\n\n";

        // Luon khoi tao hinh hoc mien va cac lien ket Bouzidi truoc khi huan luyen
        init_domain();
        std::cout << " [KHOI TAO MARL] So luong lien ket bien cong Bouzidi: " << bouzidi_links.size() << "\n";

        ensure_warmup_checkpoint();

        float best_reward = -999999.0f;
        if (std::filesystem::exists("best_marl_weights.bin"))
        {
            std::cout << " [BASELINE EVAL] Dang danh gia mo hinh hien tai tren moc khoi dong am buoc " << WARMUP_STEPS << "...\n";
            best_reward = run_episode(0, false);
            std::cout << "  --> Thuong baseline hien tai: " << best_reward << "\n\n";
        }

        for (int ep = 1; ep <= total_episodes; ++ep)
        {
            // Kiem tra tin hieu dung tu stop.bat hoac Ctrl+C
            if (check_stop_signal())
            {
                std::cout << "\n====================================================================\n";
                std::cout << " [TIN HIEU DUNG] Nhan duoc stop.flag! Dung tien trinh huan luyen an toan.\n";
                std::cout << "  - Mo hinh tot nhat da duoc bao toan tai: best_marl_weights.bin\n";
                std::cout << "====================================================================\n\n";
                if (std::filesystem::exists("stop.flag"))
                {
                    std::filesystem::remove("stop.flag");
                }
                break;
            }

            float ep_reward = run_episode(ep, true);

            if (ep_reward > best_reward)
            {
                best_reward = ep_reward;
                model.save_weights("best_marl_weights.bin");
                std::cout << "  --> [LUU MO HINH TOT NHAT] Da luu best_marl_weights.bin (Thuong ky luc: "
                          << best_reward << ")\n";
            }
        }

        std::cout << "\n====================================================================\n";
        std::cout << " [HOAN TAT HUAN LUYEN] Mo hinh MARL toi uu da luu vao: best_marl_weights.bin\n";
        std::cout << " Ban co the chay kiem dinh ngay bang:\n";
        std::cout << "   make run MODEL=agent   hoac   .\\manage.ps1 run -Model agent\n";
        std::cout << "====================================================================\n\n"
                  << std::flush;
    }
};
