#pragma once

#include "parameters.hpp"
#include "fields.hpp"
#include <cstdio>
#include <vector>
#include <string>

inline void write_vtk(int step, const std::string &prefix = "")
{
    char filename[256];
    std::string tag = prefix.empty() ? g_cfg.model : prefix;
    std::snprintf(filename, sizeof(filename), "cylinder_%s_%06d.vtk", tag.c_str(), step);
    FILE *fp = std::fopen(filename, "w");
    if (!fp)
        return;

    // Bo dem RAM 8MB de tang toc I/O dia cung
    std::vector<char> buf(8 * 1024 * 1024);
    std::setvbuf(fp, buf.data(), _IOFBF, buf.size());

    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "LBM Cylinder Flow Re=%.0f Model=%s\n", g_cfg.Re, tag.c_str());
    std::fprintf(fp, "ASCII\n");
    std::fprintf(fp, "DATASET RECTILINEAR_GRID\n");
    std::fprintf(fp, "DIMENSIONS %d %d 1\n", NX, NY);

    std::fprintf(fp, "X_COORDINATES %d float\n", NX);
    for (int i = 0; i < NX; ++i)
        std::fprintf(fp, "%.5f ", static_cast<float>(i) * dx);
    std::fprintf(fp, "\n");

    std::fprintf(fp, "Y_COORDINATES %d float\n", NY);
    for (int j = 0; j < NY; ++j)
        std::fprintf(fp, "%.5f ", static_cast<float>(j) * dx);
    std::fprintf(fp, "\n");

    std::fprintf(fp, "Z_COORDINATES 1 float\n0.0\n");

    std::fprintf(fp, "POINT_DATA %d\n", NX * NY);

    // 1. Truong do xoay khong thu nguyen omega*
    std::fprintf(fp, "SCALARS omega float 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < NY; ++j)
    {
        for (int i = 0; i < NX; ++i)
        {
            std::fprintf(fp, "%.4f\n", vort[idx(i, j)]);
        }
    }

    // 2. He so ap suat Cp
    std::fprintf(fp, "SCALARS Cp float 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < NY; ++j)
    {
        for (int i = 0; i < NX; ++i)
        {
            std::fprintf(fp, "%.4f\n", Cp[idx(i, j)]);
        }
    }

    // 3. Ap suat tuyet doi P
    std::fprintf(fp, "SCALARS pressure float 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < NY; ++j)
    {
        for (int i = 0; i < NX; ++i)
        {
            std::fprintf(fp, "%.5f\n", P[idx(i, j)]);
        }
    }

    // 4. Mien vat the solid (1 la tru tron, 0 la chat luu)
    std::fprintf(fp, "SCALARS solid int 1\nLOOKUP_TABLE default\n");
    for (int j = 0; j < NY; ++j)
    {
        for (int i = 0; i < NX; ++i)
        {
            std::fprintf(fp, "%d\n", solid[idx(i, j)]);
        }
    }

    // 5. Truong he so dong Agent alpha(x, y) (neu chay che do Agent)
    if (!alpha_closure.empty())
    {
        std::fprintf(fp, "SCALARS alpha_agent float 1\nLOOKUP_TABLE default\n");
        for (int j = 0; j < NY; ++j)
        {
            for (int i = 0; i < NX; ++i)
            {
                std::fprintf(fp, "%.4f\n", alpha_closure[idx(i, j)]);
            }
        }
    }

    // 6. Vector van toc thuc te (m/s)
    const float C_u = dx / g_cfg.dt_phy; // Chuyen doi sang m/s
    std::fprintf(fp, "VECTORS velocity float\n");
    for (int j = 0; j < NY; ++j)
    {
        for (int i = 0; i < NX; ++i)
        {
            int id = idx(i, j);
            std::fprintf(fp, "%.4f %.4f 0.0\n", ux[id] * C_u, uy[id] * C_u);
        }
    }

    std::fclose(fp);
}
