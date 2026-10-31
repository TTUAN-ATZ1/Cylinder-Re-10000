#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <csignal>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parameters.hpp"
#include "fields.hpp"
#include "domain.hpp"
#include "solver.hpp"
#include "boundary.hpp"
#include "export_vtk.hpp"
#include "checkpoint.hpp"
#include "marl_roi.hpp"
#include "marl_reward.hpp"
#include "marl_model.hpp"
#include "marl_trainer.hpp"

// Dinh nghia vung nho toan cuc
SimulationConfig g_cfg;
std::vector<BouzidiLink> bouzidi_links;
std::vector<float> f(NX * NY * 9);
std::vector<float> f_post(NX * NY * 9);
std::vector<float> rho(NX * NY, 1.0f);
std::vector<float> ux(NX * NY, 0.0f);
std::vector<float> uy(NX * NY, 0.0f);
std::vector<float> P(NX * NY, P_inf);
std::vector<float> Cp(NX * NY, 0.0f);
std::vector<float> vort(NX * NY, 0.0f);
std::vector<int> solid(NX * NY, 0);
std::vector<float> alpha_closure(NX * NY, 0.0f); // Mang he so dong cho Agent

void print_help()
{
    std::cout << "====================================================================\n";
    std::cout << "           CFD LBM D2Q9 - CYLINDER FLOW SOLVER (CLI HELP)           \n";
    std::cout << "====================================================================\n";
    std::cout << "Cac tham so dong lenh co the tuy bien:\n";
    std::cout << "  --train              : Chay che do HUAN LUYEN MARL PPO\n";
    std::cout << "  --episodes <int>     : So Episode huan luyen (mac dinh: 30)\n";
    std::cout << "  --re <float>         : So Reynolds Re (mac dinh: 10000)\n";
    std::cout << "  --steps <int>        : So buoc mo phong toi da (mac dinh: 50000)\n";
    std::cout << "  --u0 <float>         : Van toc dong toi U0 (mac dinh: 0.08, Ma ~ 0.138)\n";
    std::cout << "  --save-every <int>   : Tan suat xuat file VTK (mac dinh: 500)\n";
    std::cout << "  --log-every <int>    : Tan suat in thong tin Terminal (mac dinh: 100)\n";
    std::cout << "  --model <string>     : Mo hinh dong: 'lbgk', 'smago', 'agent' (mac dinh: 'smago')\n";
    std::cout << "  --cs <float>         : Hang so LES Smagorinsky (mac dinh: 0.14)\n";
    std::cout << "  --resume             : Tiep tuc chay tu checkpoint.bin (song sot khi tat may)\n";
    std::cout << "  --clean              : Xoa toan bo ket qua, VTK, checkpoint va dung chay\n";
    std::cout << "  --help               : Hien thi huong dan nay\n";
    std::cout << "====================================================================\n";
}

int main(int argc, char *argv[])
{
    // Dang ky xu ly tin hieu ngat Ctrl+C
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    bool is_train_mode = false;
    int train_episodes = 30;

    // Xu ly cac tham so dong lenh (CLI Parsing)
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            print_help();
            return 0;
        }
        else if (arg == "--clean")
        {
            clean_all_run_data();
            return 0;
        }
        else if (arg == "--train")
        {
            is_train_mode = true;
        }
        else if (arg == "--episodes" && i + 1 < argc)
        {
            train_episodes = std::stoi(argv[++i]);
        }
        else if (arg == "--resume")
        {
            g_cfg.resume = true;
        }
        else if (arg == "--re" && i + 1 < argc)
        {
            g_cfg.Re = std::stof(argv[++i]);
        }
        else if (arg == "--steps" && i + 1 < argc)
        {
            g_cfg.max_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--u0" && i + 1 < argc)
        {
            g_cfg.U0 = std::stof(argv[++i]);
        }
        else if (arg == "--save-every" && i + 1 < argc)
        {
            g_cfg.save_every = std::stoi(argv[++i]);
        }
        else if (arg == "--log-every" && i + 1 < argc)
        {
            g_cfg.log_every = std::stoi(argv[++i]);
        }
        else if (arg == "--model" && i + 1 < argc)
        {
            g_cfg.model = argv[++i];
        }
        else if (arg == "--cs" && i + 1 < argc)
        {
            g_cfg.smago_Cs = std::stof(argv[++i]);
        }
    }

#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
#endif

    // Neu goi lenh huan luyen MARL
    if (is_train_mode)
    {
        MarlTrainer trainer;
        trainer.train(train_episodes);
        return 0;
    }

    // Tinh toan lai cac hang so vat ly dua tren cau hinh
    g_cfg.update_physics();
    g_cfg.print_summary();

    int start_step = 0;
    float t_phy = 0.0f;
    float Cd = 0.0f, Cl = 0.0f;

    // Khoi tao hinh hoc mien va cac lien ket Bouzidi
    init_domain();
    std::cout << " [KHOI TAO] So luong lien ket bien cong Bouzidi: " << bouzidi_links.size() << "\n";

    // Khoi tao mo hinh Agent neu chay che do model = 'agent'
    MarlROI roi;
    MarlActorCritic agent_model;
    if (g_cfg.model == "agent")
    {
        roi.reset_history();
        if (std::filesystem::exists("best_marl_weights.bin"))
        {
            if (agent_model.load_weights("best_marl_weights.bin"))
            {
                std::cout << " [MARL AGENT] Da nap trong so mo hinh Generation 2 tu best_marl_weights.bin!\n";
            }
            else
            {
                std::cout << " [CANH BAO] File best_marl_weights.bin thuoc the he cu (6 kenh)! Khoi tao trong so ngau nhien cho Generation 2 (24 kenh Frame Stacking).\n";
            }
        }
        else if (std::filesystem::exists("marl_weights.bin"))
        {
            if (agent_model.load_weights("marl_weights.bin"))
            {
                std::cout << " [MARL AGENT] Da nap trong so mo hinh tu marl_weights.bin!\n";
            }
            else
            {
                std::cout << " [CANH BAO] File marl_weights.bin thuoc the he cu! Khoi tao trong so ngau nhien cho Generation 2.\n";
            }
        }
        else
        {
            std::cout << " [CANH BAO] Khong tim thay file trong so Agent! Khoi tao trong so ngau nhien.\n";
            std::cout << "           (Goi y: Ban co the chay 'make train' de huan luyen truoc)\n";
        }
    }

    // Kiem tra neu chay che do phuc hoi Resume
    if (g_cfg.resume)
    {
        if (!load_checkpoint(start_step, t_phy, Cd, Cl, "checkpoint.bin"))
        {
            std::cerr << " [CANH BAO] Khong the resume tu checkpoint.bin! Chuyen sang khoi tao moi.\n";
            start_step = 0;
            t_phy = 0.0f;
        }
        else
        {
            start_step++;
        }
    }

    // Mo file luu luc forces.txt (app neu resume, tao moi neu chay tu dau)
    std::ofstream force_file;
    if (g_cfg.resume && start_step > 0)
    {
        force_file.open("forces.txt", std::ios::app);
    }
    else
    {
        force_file.open("forces.txt", std::ios::out);
        force_file << "# Step Time_s Cd Cl P_in P_top P_near P_out\n";
    }

    std::cout << " [BAT DAU] Tien trinh mo phong dang chay tu buoc " << start_step
              << " den " << g_cfg.max_steps << "...\n\n" << std::flush;

    // Vong lap mo phong thoi gian LBM
    for (int step = start_step; step <= g_cfg.max_steps; ++step)
    {
        // 1. Kiem tra tin hieu Stop Signal
        if (check_stop_signal())
        {
            std::cout << "\n [TIN HIEU DUNG] Nhan duoc yeu cau tam dung mo phong!\n";
            save_checkpoint(step, step * g_cfg.dt_phy, Cd, Cl, "checkpoint.bin");
            force_file.flush();
            force_file.close();
            return 0;
        }

        // 2. Neu dung che do Agent: cap nhat hanh dong moi 10 buoc
        if (g_cfg.model == "agent" && step % 10 == 0)
        {
            compute_vorticity_and_cp();
            roi.extract_observation();
            std::vector<float> actions, log_probs, values;
            agent_model.forward(roi.obs_buffer, actions, log_probs, values, true); // Suy luan chuan deterministic
            roi.apply_actions(actions, -0.50f, 3.50f);
        }

        // 3. Va cham LBM
        collide();

        // 4. Lan truyen + Bien Bouzidi + Bien ngoai vi + Do luc Ladd MEA
        stream_and_bc(Cd, Cl);

        // 5. Cap nhat bien vi mo
        update_macroscopic();

        // 6. Kiem tra on dinh so hoc (Phat hien NaN / Inf / Blow-up)
        if (step % 50 == 0 || step == start_step)
        {
            std::string diag_msg;
            if (!check_numerical_stability(step, Cd, Cl, diag_msg))
            {
                std::cout << "\n";
                std::cout << "*******************************************************************************\n";
                std::cout << "*               [CRITICAL ALERT: BLOW-UP / NaN DETECTED]                      *\n";
                std::cout << "*******************************************************************************\n";
                std::cout << "  - Buoc xay ra su co    : " << step << " (t = " << step * g_cfg.dt_phy << "s)\n";
                std::cout << "  - Trang thai hien tai  : Cd = " << Cd << ", Cl = " << Cl << "\n";
                std::cout << "  - Chi tiet chan doan   : " << diag_msg << "\n";
                std::cout << "  - Nguyen nhan vat ly   : O Re = " << g_cfg.Re << ", do nhot nu = " << g_cfg.nu
                          << " qua nho khien tau = " << g_cfg.tau << " tien sat 0.5.\n";
                std::cout << "                           LBGK co dien khong the triet tieu cac dao dong\n";
                std::cout << "                           song am tren luoi tho D = " << B << " cells.\n";
                std::cout << "  - GIAI PHAP KHAC PHUC  : Chay voi mo hinh LES Smagorinsky:\n";
                std::cout << "                           .\\lbm_solver.exe --re " << g_cfg.Re << " --model smago\n";
                std::cout << "                           hoac huan luyen MARL: make train\n";
                std::cout << "*******************************************************************************\n\n" << std::flush;

                std::ofstream crash_log("crash_dump.txt");
                crash_log << "CRITICAL NUMERICAL BLOW-UP DETECTED\n";
                crash_log << "Step: " << step << "\n";
                crash_log << "Time: " << step * g_cfg.dt_phy << " s\n";
                crash_log << "Re: " << g_cfg.Re << "\n";
                crash_log << "Model: " << g_cfg.model << "\n";
                crash_log << "Cd: " << Cd << ", Cl: " << Cl << "\n";
                crash_log << "Diagnosis: " << diag_msg << "\n";
                crash_log.close();

                force_file.flush();
                force_file.close();
                return 1;
            }
        }

        // 7. Ghi nhat ky luc moi 10 buoc
        if (step % 10 == 0)
        {
            float P_top = P[idx(static_cast<int>(xc), NY - 1)];
            float P_near = P[idx(static_cast<int>(xc), static_cast<int>(yc + 5.0f * radius))];
            float P_in = P[idx(0, NY / 2)];
            float P_out = P[idx(NX - 1, NY / 2)];

            force_file << step << " "
                       << step * g_cfg.dt_phy << " "
                       << Cd << " " << Cl << " "
                       << P_in << " " << P_top << " " << P_near << " " << P_out << "\n";
        }

        // 8. In thong so Terminal moi log_every buoc
        if (step % g_cfg.log_every == 0)
        {
            force_file.flush();
            float P_top = P[idx(static_cast<int>(xc), NY - 1)];
            float P_in = P[idx(0, NY / 2)];

            std::cout << std::fixed;
            std::cout << "Buoc " << std::setw(6) << step
                      << " | t=" << std::setw(6) << std::setprecision(3) << step * g_cfg.dt_phy << "s"
                      << " | Cd=" << std::setw(5) << std::setprecision(3) << Cd
                      << " | Cl=" << (Cl >= 0 ? "+" : "") << std::setw(5) << std::setprecision(3) << Cl
                      << " | P_top=" << std::setprecision(5) << P_top
                      << " | P_in=" << P_in
                      << " | [ON DINH]\n" << std::flush;
        }

        // 9. Xuat VTK moi save_every buoc
        if (step % g_cfg.save_every == 0 && step > 0)
        {
            compute_vorticity_and_cp();
            write_vtk(step);
            std::cout << "  --> [VTK EXPORT] Da xuat file cylinder_" << g_cfg.model << "_" << std::setfill('0') << std::setw(6) << step << ".vtk\n" << std::flush;
        }
    }

    // Dam bao buoc cuoi cung luon co file VTK
    if (g_cfg.max_steps % g_cfg.save_every != 0)
    {
        compute_vorticity_and_cp();
        write_vtk(g_cfg.max_steps);
        std::cout << "  --> [VTK EXPORT] Da xuat file buoc cuoi: cylinder_" << g_cfg.model << "_" << std::setfill('0') << std::setw(6) << g_cfg.max_steps << ".vtk\n" << std::flush;
    }

    force_file.close();

    std::cout << "\n====================================================================\n";
    std::cout << " [HOAN TAT MO PHONG] Da chay thanh cong " << g_cfg.max_steps << " buoc!\n";
    std::cout << " File luc khi dong da luu vao: forces.txt\n";
    std::cout << "====================================================================\n";

    if (std::filesystem::exists("checkpoint.bin"))
    {
        std::filesystem::remove("checkpoint.bin");
    }

    return 0;
}
