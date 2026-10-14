#!/usr/bin/env python3
import math
import numpy as np
from pathlib import Path

from ase.io import read
import matplotlib.pyplot as plt

# -----------------------------
# 参数设置
# -----------------------------
base_dir = Path(".")
pattern  = "rndm*.in"

dr   = 0.5
rmax = 10.0

species_X  = "X"
species_La = "La"


effective_thickness = 5.6

# -----------------------------
# 初始化 RDF 统计容器
# -----------------------------
bin_edges   = np.arange(0.0, rmax + dr, dr)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
nbins       = len(bin_centers)

hist_xx  = np.zeros(nbins, dtype=float)
hist_xLa = np.zeros(nbins, dtype=float)

n_struct = 0
N_X  = None
N_La = None
eff_volume = None

# -----------------------------
# 遍历所有结构文件并累积统计
# -----------------------------
files = sorted(base_dir.glob(pattern))
if not files:
    raise FileNotFoundError(
        f"在 {base_dir.resolve()} 下没有找到 {pattern} 文件"
    )

for fpath in files:
    atoms = read(fpath, format="aims")
    atoms.set_pbc([True, True, True])

    symbols = atoms.get_chemical_symbols()
    idx_X   = [i for i, s in enumerate(symbols) if s == species_X]
    idx_La  = [i for i, s in enumerate(symbols) if s == species_La]

    if not idx_X:

        continue

    # ---------- 计算 xy 表面面积，并构造“有效体积” ----------
    if eff_volume is None:
        cell = atoms.get_cell()
        a_vec = cell[0]
        b_vec = cell[1]

        area_xy = np.linalg.norm(np.cross(a_vec, b_vec))

        eff_volume = area_xy * effective_thickness

    if N_X is None:
        N_X  = len(idx_X)
        N_La = len(idx_La)
    else:

        if len(idx_X) != N_X or len(idx_La) != N_La:
            raise RuntimeError(
                f"{fpath} 中 X/La 数目与第一帧不一致，当前脚本假定所有结构原子数相同。"
            )

    n_struct += 1

    # ---------- 统计 X–X ----------
    for a in range(len(idx_X)):
        ia = idx_X[a]
        for b in range(a + 1, len(idx_X)):
            ib = idx_X[b]
            d = atoms.get_distance(ia, ib, mic=True)
            if d < rmax:
                k = int(d / dr)
                hist_xx[k] += 1.0

    # ---------- 统计 X–La ----------
    for ia in idx_X:
        for ib in idx_La:
            d = atoms.get_distance(ia, ib, mic=True)
            if d < rmax:
                k = int(d / dr)
                hist_xLa[k] += 1.0

print(f"共统计结构数: {n_struct}")
if n_struct == 0:
    raise RuntimeError(
        "没有任何结构含有 X 原子，检查文件和 species_X 设置。"
    )

# -----------------------------
# 归一化：得到 g(r)
# -----------------------------
rho_X  = N_X  / eff_volume
rho_La = N_La / eff_volume

g_xx  = np.zeros(nbins, dtype=float)
g_xLa = np.zeros(nbins, dtype=float)

for k, r in enumerate(bin_centers):
    shell_vol = 4.0 * math.pi * (r ** 2) * dr
    if shell_vol == 0.0:
        continue
    # g_AB(r) = hist(r) / [ n_struct * N_A * shell_vol * ρ_B ]
    g_xx[k]  = hist_xx[k]  / (n_struct * N_X * shell_vol * rho_X)
    g_xLa[k] = hist_xLa[k] / (n_struct * N_X * shell_vol * rho_La)

# -----------------------------
# 写数据文件
# -----------------------------
out_dir = base_dir
out_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(out_dir / "rdf_X_X.dat",
           np.column_stack([bin_centers, g_xx]),
           fmt="%12.6f  %16.8f")

np.savetxt(out_dir / "rdf_X_La.dat",
           np.column_stack([bin_centers, g_xLa]),
           fmt="%12.6f  %16.8f")

print("已输出: rdf_X_X.dat, rdf_X_La.dat")

# -----------------------------
# 画 RDF 图
# -----------------------------
plt.style.use("default")
plt.rcParams.update({
    #"font.family": "DejaVu Sans",
    "font.family": "serif",
    "font.serif": ["Times", "STIX", "DejaVu Serif"],
    "axes.linewidth": 2.0,
    "font.size": 20,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.major.width": 3.0,
    "ytick.major.width": 3.0,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times",
    "axes.edgecolor": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.set_facecolor("white")

# 颜色：沿用你前面整篇文章的色系
color_xx  = "#E5893A"
color_xLa = "#4B93C4"

ax.plot(bin_centers, g_xx,
        color=color_xx, lw=2.2, label="Vo–Vo")

ax.plot(bin_centers, g_xLa,
        color=color_xLa, lw=2.2,
        label="Vo–Ce$^{3+}$")

ax.set_xlabel(r"$r$ (Å)", fontsize=22, fontweight="bold")
ax.set_ylabel(r"$\mathbf{g(r)}$", fontsize=22, fontweight="bold")

ax.set_xlim(0.0, rmax)
ax.set_ylim(bottom=0.0)

# 刻度数字也加粗
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontweight("bold")

# 边框再确认加粗
for spine in ax.spines.values():
    spine.set_linewidth(2.0)

# 图例：无边框、字号 13、加粗
leg = ax.legend(frameon=False, fontsize=18, handlelength=1,loc="upper right")
for text in leg.get_texts():
    text.set_fontweight("bold")



#ax.set_title("Radial distribution functions", fontsize=13, pad=8)

plt.tight_layout()
fig.savefig(out_dir / "rdf_X_pairs.png",
            dpi=600, bbox_inches="tight")
plt.show()
"""
# -----------------------------
# 画 RDF 图
# -----------------------------
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # 用于高斯平滑

plt.style.use("default")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.set_facecolor("white")

# --- 色系：砖红 & 蓝 ---
color_xx  = "#E5893A"   # Vo–Vo
color_xLa = "#4B93C4"   # Vo–Ce3+

# --- 高斯平滑 ---
sigma = 1   

g_xx_smooth  = gaussian_filter1d(g_xx,  sigma=sigma)
g_xLa_smooth = gaussian_filter1d(g_xLa, sigma=sigma)

# --- 绘图：全部实线 ---
ax.plot(bin_centers, g_xx_smooth,
        color=color_xx, lw=1.8, label="Vo–Vo")

ax.plot(bin_centers, g_xLa_smooth,
        color=color_xLa, lw=1.8, label="Vo–Ce³⁺")

# --- 坐标系设置 ---
ax.set_xlabel(r"$r$ (Å)", fontsize=13)
ax.set_ylabel(r"$g(r)$", fontsize=13)
ax.set_xlim(0.0, rmax)
ax.set_ylim(bottom=0.0)
ax.legend(frameon=False, fontsize=11)

ax.set_title("Radial distribution functions", fontsize=13, pad=8)

plt.tight_layout()
fig.savefig(out_dir / "rdf_X_pairs.png",
            dpi=600, bbox_inches="tight")
plt.show()
"""
