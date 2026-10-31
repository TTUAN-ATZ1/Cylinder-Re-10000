#pragma once

#include "parameters.hpp"
#include "fields.hpp"
#include <cmath>
#include <vector>
#include <iostream>

struct MarlRewardEngine
{
    // Cac moc chuan tu cylinder_re10000_benchmark.csv (Dong 2005/2006, Norberg 2003)
    const float CD_TARGET = 1.143f;        // DNS-B3 (Dong et al. 2006, JFM)
    const float CPB_TARGET = -1.129f;       // Ap suat day tru (Dong 2006)
    const float CL_RMS_TARGET = 0.448f;    // He so nang hieu dung dao dong DNS-B3 (0.394 theo Norberg 2003)
    const float CL_PEAK_TARGET = 0.634f;   // Bien do dinh chuan thuc nghiem 3D = sqrt(2) * CL_RMS_TARGET

    // 8 goc do kiem dinh ap suat quanh mat tru
    struct CpProbe
    {
        float angle_rad;
        float target_Cp;
    };
    std::vector<CpProbe> cp_probes;

    MarlRewardEngine()
    {
        const float PI = 3.14159265358979323846f;
        // Diem dung dau mui (0 deg): Cp = 1.000
        cp_probes.push_back({0.0f, 1.000f});
        // Goc 30 deg: Cp = 0.420
        cp_probes.push_back({30.0f * PI / 180.0f, 0.420f});
        // Goc 60 deg: Cp = -0.920
        cp_probes.push_back({60.0f * PI / 180.0f, -0.920f});
        // Dinh hut manh nhat tai 75 deg: Cp = -1.420
        cp_probes.push_back({75.0f * PI / 180.0f, -1.420f});
        // Diem tach dong tai 82 deg: Cp = -1.210
        cp_probes.push_back({82.0f * PI / 180.0f, -1.210f});
        // Vung ben 90 deg: Cp = -1.130
        cp_probes.push_back({90.0f * PI / 180.0f, -1.130f});
        // Vung sau 120 deg: Cp = -1.129
        cp_probes.push_back({120.0f * PI / 180.0f, -1.129f});
        // Day tru phia sau (180 deg): Cp = -1.130
        cp_probes.push_back({180.0f * PI / 180.0f, -1.130f});
    }

    // Tinh phan thuong tuc thoi tai buoc quyet dinh
    float compute_step_reward(float Cd_current, float Cl_current, bool is_unstable)
    {
        if (is_unstable)
        {
            return -50.0f; // Phat cuc nang neu no so
        }

        // 1. Thuong luc can: Trong so manh -15.0 de ep Cd ve chuan Dong (2005/2006) 1.143
        float cd_diff = (Cd_current - CD_TARGET);
        float r_drag = -15.0f * (cd_diff * cd_diff);

        // 2. Thuong kiem soat luc nang Cl theo chuan Dong (2006) & Norberg (2003):
        // Trong 3D thuc te, do co 3D vortex stretching lam tieu tan nang luong xoay nen bien do dinh |Cl| <= 0.634.
        // O mo hinh 2D khong co 3D dissipation, xoáy cuon thanh cac con lan khong lo lam |Cl| phong dai toi 1.5.
        // Phat nang khi |Cl| vuot qua gioi han vat ly 3D CL_PEAK_TARGET (0.634), ep Agent phai dieu che tang
        // do nhot tieu tan SGS de dap bien do dao dong, giup Cd on dinh va Cl ve dung chuan thuc nghiem.
        float cl_excess = std::max(0.0f, std::abs(Cl_current) - CL_PEAK_TARGET);
        float r_lift = -15.0f * (cl_excess * cl_excess);

        // 3. Thuong phan bo ap suat quanh mat tru
        float cp_mse = 0.0f;
        const float probe_radius = radius + 0.8f; // Lay tai lop chat long ngay sat mat tru
        for (const auto &probe : cp_probes)
        {
            float px = xc + probe_radius * std::cos(probe.angle_rad);
            float py = yc + probe_radius * std::sin(probe.angle_rad);

            int gx = std::clamp(static_cast<int>(std::round(px)), 0, NX - 1);
            int gy = std::clamp(static_cast<int>(std::round(py)), 0, NY - 1);

            float cp_sim = Cp[idx(gx, gy)];
            float diff = cp_sim - probe.target_Cp;
            cp_mse += diff * diff;
        }
        cp_mse /= static_cast<float>(cp_probes.size());
        float r_cp = -2.0f * cp_mse;

        // 4. Thuong diem dong bong hoan luu Lr: Tai x/D = 1.32 (tuc cach tam tru 1.32*30 = 39.6 cells)
        // Van toc truc tim uc/U0 tai diem nay phai = 0.0
        int closure_x = static_cast<int>(xc + 1.32f * B);
        int closure_y = static_cast<int>(yc);
        float uc_norm = ux[idx(closure_x, closure_y)] / g_cfg.U0;
        float r_wake = -3.0f * (uc_norm * uc_norm);

        // Tong phan thuong vat ly
        float total_reward = r_drag + r_lift + r_cp + r_wake;
        return total_reward;
    }
};
