#pragma once

#include "parameters.hpp"
#include "fields.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <csignal>
#include <filesystem>

// Co bao dung tu Ctrl+C hoac stop.flag
inline volatile std::sig_atomic_t g_stop_requested = 0;

inline void signal_handler(int sig)
{
    if (sig == SIGINT || sig == SIGTERM)
    {
        g_stop_requested = 1;
    }
}

// Kiem tra tin hieu tam dung tu nguoi dung (qua lenh make stop hoac Ctrl+C)
inline bool check_stop_signal()
{
    if (g_stop_requested)
        return true;

    // Kiem tra su xuat hien cua file co stop.flag
    std::ifstream flag_file("stop.flag");
    if (flag_file.good())
    {
        flag_file.close();
        return true;
    }
    return false;
}

// Luu trang thai mo phong ra file nhi phan checkpoint.bin
inline bool save_checkpoint(int step, float t_phy, float Cd, float Cl, const std::string &path = "checkpoint.bin")
{
    std::cout << "\n====================================================================\n";
    std::cout << " [DANG LUU CHECKPOINT] Luu toan bo trang thai luoi ra " << path << "...\n";
    std::cout << "====================================================================\n";

    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        std::cerr << " [LOI] Khong the mo file " << path << " de ghi!\n";
        return false;
    }

    const char magic[8] = {'L', 'B', 'M', 'C', 'H', 'K', '0', '1'};
    out.write(magic, 8);

    int nx = NX;
    int ny = NY;
    out.write(reinterpret_cast<const char *>(&nx), sizeof(nx));
    out.write(reinterpret_cast<const char *>(&ny), sizeof(ny));
    out.write(reinterpret_cast<const char *>(&step), sizeof(step));
    out.write(reinterpret_cast<const char *>(&t_phy), sizeof(t_phy));
    out.write(reinterpret_cast<const char *>(&g_cfg.Re), sizeof(g_cfg.Re));
    out.write(reinterpret_cast<const char *>(&g_cfg.U0), sizeof(g_cfg.U0));
    out.write(reinterpret_cast<const char *>(&g_cfg.dt_phy), sizeof(g_cfg.dt_phy));
    out.write(reinterpret_cast<const char *>(&Cd), sizeof(Cd));
    out.write(reinterpret_cast<const char *>(&Cl), sizeof(Cl));

    // Luu cac mang phan bo va van toc
    out.write(reinterpret_cast<const char *>(f.data()), f.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(rho.data()), rho.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(ux.data()), ux.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(uy.data()), uy.size() * sizeof(float));

    out.close();

    // Xoa file co stop.flag neu co
    if (std::filesystem::exists("stop.flag"))
    {
        std::filesystem::remove("stop.flag");
    }

    std::cout << " [THANH CONG] Da luu checkpoint tai buoc " << step << " (t = " << t_phy << "s).\n";
    std::cout << "  Ban co the tat may tinh an toan va tiep tuc bat cu luc nao bang:\n";
    std::cout << "    make continue   hoac   .\\manage.ps1 continue\n";
    std::cout << "====================================================================\n" << std::flush;
    return true;
}

// Nap lai trang thai tu file checkpoint.bin
inline bool load_checkpoint(int &step, float &t_phy, float &Cd, float &Cl, const std::string &path = "checkpoint.bin")
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << " [LOI] Khong tim thay file checkpoint '" << path << "' de phuc hoi!\n";
        return false;
    }

    std::cout << "\n====================================================================\n";
    std::cout << " [PHUC HOI CHECKPOINT] Dang doc file " << path << "...\n";
    std::cout << "====================================================================\n";

    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::cerr << " [LOI] Khong the mo file " << path << " de doc!\n";
        return false;
    }

    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != "LBMCHK01")
    {
        std::cerr << " [LOI] Dinh dang file checkpoint khong hop le hoac bi hong!\n";
        return false;
    }

    int nx, ny;
    in.read(reinterpret_cast<char *>(&nx), sizeof(nx));
    in.read(reinterpret_cast<char *>(&ny), sizeof(ny));
    if (nx != NX || ny != NY)
    {
        std::cerr << " [LOI] Kich thuoc luoi khong khop! File: " << nx << "x" << ny
                  << ", Chuong trinh: " << NX << "x" << NY << "\n";
        return false;
    }

    in.read(reinterpret_cast<char *>(&step), sizeof(step));
    in.read(reinterpret_cast<char *>(&t_phy), sizeof(t_phy));
    in.read(reinterpret_cast<char *>(&g_cfg.Re), sizeof(g_cfg.Re));
    in.read(reinterpret_cast<char *>(&g_cfg.U0), sizeof(g_cfg.U0));
    in.read(reinterpret_cast<char *>(&g_cfg.dt_phy), sizeof(g_cfg.dt_phy));
    in.read(reinterpret_cast<char *>(&Cd), sizeof(Cd));
    in.read(reinterpret_cast<char *>(&Cl), sizeof(Cl));

    // Cap nhat lai thong so vat ly
    g_cfg.update_physics();

    // Doc du lieu mang
    in.read(reinterpret_cast<char *>(f.data()), f.size() * sizeof(float));
    in.read(reinterpret_cast<char *>(rho.data()), rho.size() * sizeof(float));
    in.read(reinterpret_cast<char *>(ux.data()), ux.size() * sizeof(float));
    in.read(reinterpret_cast<char *>(uy.data()), uy.size() * sizeof(float));

    in.close();

    // Dong bo hoa f_post va P
    for (size_t i = 0; i < f.size(); ++i)
    {
        f_post[i] = f[i];
    }
    for (size_t i = 0; i < rho.size(); ++i)
    {
        P[i] = rho[i] * cs2;
    }

    std::cout << " [THANH CONG] Da phuc hoi tai buoc " << step << " (t = " << t_phy << "s), Re = " << g_cfg.Re << "!\n";
    std::cout << "====================================================================\n\n" << std::flush;
    return true;
}

// Xoa toan bo du lieu chay khi nguoi dung goi make cancel
inline void clean_all_run_data()
{
    std::cout << "\n====================================================================\n";
    std::cout << " [MAKE CANCEL] Dang xoa toan bo ket qua va file checkpoint cua lan chay nay...\n";

    if (std::filesystem::exists("checkpoint.bin"))
    {
        std::filesystem::remove("checkpoint.bin");
        std::cout << "  - Da xoa checkpoint.bin\n";
    }
    if (std::filesystem::exists("stop.flag"))
    {
        std::filesystem::remove("stop.flag");
        std::cout << "  - Da xoa stop.flag\n";
    }
    if (std::filesystem::exists("crash_dump.txt"))
    {
        std::filesystem::remove("crash_dump.txt");
        std::cout << "  - Da xoa crash_dump.txt\n";
    }

    // Xoa cac file VTK
    int vtk_count = 0;
    try {
        for (const auto &entry : std::filesystem::directory_iterator("."))
        {
            if (entry.is_regular_file())
            {
                std::string fname = entry.path().filename().string();
                if (entry.path().extension() == ".vtk" && fname.rfind("cylinder_", 0) == 0)
                {
                    std::filesystem::remove(entry.path());
                    vtk_count++;
                }
            }
        }
    } catch (...) {}
    std::cout << "  - Da xoa " << vtk_count << " file VTK ket qua.\n";

    // Xoa hoac reset forces.txt
    if (std::filesystem::exists("forces.txt"))
    {
        std::filesystem::remove("forces.txt");
        std::cout << "  - Da lam sach forces.txt\n";
    }

    std::cout << " [HOAN TAT] Mien du lieu da duoc dua ve trang thai nguyen ban!\n";
    std::cout << "====================================================================\n\n" << std::flush;
}
