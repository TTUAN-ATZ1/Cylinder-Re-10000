#pragma once
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>

// Kich thuoc luoi co dinh (Dam bao mien vat ly lon: 10.5D truoc tru, 50D sau tru)
constexpr int NX = 1401;
constexpr int NY = 501;

constexpr float radius = 15.0f;           // R = 15 o luoi
constexpr float B = 2.0f * radius;        // Duong kinh D = 30 o luoi
constexpr float xc = 315.0f;              // Khoang cach tu Inlet den tam tru = 10.5D
constexpr float yc = 250.05f;             // Tam theo truc Y (lech nhe 0.05 tranh suy bien nut)
constexpr float dx = 0.005f;              // 1 cell = 0.005 m

// Hang so toc do am thanh LBM
constexpr float cs2 = 1.0f / 3.0f;
constexpr float P_inf = cs2 * 1.0f;       // Ap suat chuan tai vo cuc

// Mo hinh D2Q9
constexpr int ex[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr int ey[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constexpr float w[9] = {
    4.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};
constexpr int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// Ham noi suy chi so 1D
inline int idx(int i, int j) { return i * NY + j; }
inline int fidx(int i, int j, int k) { return (i * NY + j) * 9 + k; }

// Cau hinh mo phong dong (Tien loi tuy bien qua CLI)
struct SimulationConfig
{
    float Re = 10000.0f;               // So Reynolds (Mac dinh 10000, co the doi qua CLI)
    float U0 = 0.08f;                  // Van toc dong toi (Ma ~ 0.1385 < 0.15)
    float dt_phy = 0.0004f;            // 1 timestep = 0.0004 s (Cu = dx/dt = 12.5 m/s)
    int max_steps = 50000;             // Tong so buoc chay
    int save_every = 500;              // Xuat file VTK moi 500 buoc
    int log_every = 100;               // In ket qua ra Terminal moi 100 buoc
    bool resume = false;               // Tiep tuc chay tu checkpoint.bin
    std::string model = "lbgk";        // Mo hinh dong: "lbgk", "smago", "agent"
    float smago_Cs = 0.14f;            // Hang so Smagorinsky neu dung mo hinh LES

    // Cac bien dong luc hoc tu dong tinh toan:
    float nu = 0.0f;                   // Do nhot dong hoc
    float tau = 0.0f;                  // Thoi gian thu gian co so
    float omega = 0.0f;                // Tan so va cham co so
    float q_dyn = 0.0f;                // Ap suat dong q = 0.5 * rho * U0^2

    void update_physics()
    {
        nu = (U0 * B) / Re;
        tau = 3.0f * nu + 0.5f;
        omega = 1.0f / tau;
        q_dyn = 0.5f * 1.0f * U0 * U0;
    }

    void print_summary() const
    {
        std::cout << "\n====================================================================\n";
        std::cout << "           CFD LBM D2Q9 SOLVER - CYLINDER FLOW BENCHMARK            \n";
        std::cout << "====================================================================\n";
        std::cout << "  - Kich thuoc luoi      : " << NX << " x " << NY << " (" << NX * NY << " nut luoi)\n";
        std::cout << "  - Duong kinh tru D     : " << B << " cells (R = " << radius << ")\n";
        std::cout << "  - Chieu dai duoi dong  : " << std::fixed << std::setprecision(1) << (NX - xc) / B << " D (50D chuan Dong 2004/2005)\n";
        std::cout << "  - So Reynolds (Re)     : " << std::setprecision(0) << Re << "\n";
        std::cout << "  - Van toc toi (U0)     : " << std::setprecision(4) << U0 << " (Mach = " << U0 * std::sqrt(3.0f) << ")\n";
        std::cout << "  - Do nhot nu           : " << std::scientific << std::setprecision(5) << nu << "\n";
        std::cout << "  - Thoi gian hoi phuc   : tau = " << std::fixed << std::setprecision(6) << tau << " (omega = " << omega << ")\n";
        std::cout << "  - Mo hinh dong         : [" << model << "]\n";
        std::cout << "  - Tong so buoc chay    : " << max_steps << " steps (" << max_steps * dt_phy << " giay vat ly)\n";
        std::cout << "  - Che do khoi dong     : " << (resume ? "[RESUME TU CHECKPOINT]" : "[KHOI TAO MOI]") << "\n";
        std::cout << "====================================================================\n\n" << std::flush;
    }
};

extern SimulationConfig g_cfg;
