import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot_cp_vs_ulysses_kv():
    B = 1.0
    a = 2.1930476567007057e-09
    #b = 0.0011911824608909084
    b = 0
    c = 1.1176326223782225
    d = 128.0
    n_kv = 2.0
    n_h = 16.0
    coe = 1.15

    # p 取值
    p_list = [8, 16, 32, 64, 128]


    B_a2a_map = {
        1: 1e10,
        2: 119.54,
        4: 104.07,
        8: 96.5,
        16: 10.33,
        32: 5.94,
        64: 4.87,
        128: 3.50,
    }

    B_p2p_map = {
        1: 1e10,
        2:7.65998,
        4:8.02132,
        8:8.76278,
        16: 8.12177,
        32: 8.02132,
        64: 7.65998,
        128: 7.35998,
    }

    # x 轴：sequence length，使用 1k 步长（单位：tokens）
    s_vals = np.arange(1024, 65536 + 1024, 1024, dtype=float)

    def B_a2a(p: int) -> float:
        return float(B_a2a_map[p])

    def B_p2p(p: int) -> float:
        return float(B_p2p_map[p])

    def y_cp(s: np.ndarray, p: int) -> np.ndarray:
        term_comp = B * (a * (s ** 2) / p + b * s + c * p)
        term_comm = (coe - 1.0) * (4.0 * B * s * n_kv * d / B_p2p(p) * ((p - 1.0) / p))
        return term_comp + term_comm

    def y_ulysses_kv(s: np.ndarray, p: int) -> np.ndarray:
        term_comp = B * (a * s ** 2 + b * s + c) / p
        term_comm1 = 4.0 * (B * s * n_h * d) / B_a2a(p) * ((p - 1.0) / (p ** 2))
        term_comm2 = 4.0 * (B * s * p * d) / B_a2a(p) * ((p - 1.0) / (p ** 2))
        return term_comp + term_comm1 + term_comm2

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10.colors
    for idx, p in enumerate(p_list):
        cp_ms = y_cp(s_vals, p)
        ul_ms = y_ulysses_kv(s_vals, p)
        color = colors[idx % len(colors)]
        ax.plot(s_vals, cp_ms, label=f"cp={p}", color=color, linestyle="-")
        ax.plot(s_vals, ul_ms, label=f"sp={p}", color=color, linestyle="--")

    # x 轴刻度显示 1k, 2k, ...
    def k_formatter(x, _pos):
        return f"{int(x)//1000}k" if x >= 1000 else str(int(x))

    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.set_xlabel("sequence length")
    ax.set_ylabel("ms")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(ncol=2)

    # 仅显示第一象限
    ax.set_xlim(left=0.0, right=float(s_vals.max()))
    # y 最小值置 0，最大值自动
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    # 保存到脚本同目录
    out_dir = os.path.dirname(__file__)
    out_png = os.path.join(out_dir, "cp_vs_ulysses_kv.png")
    out_pdf = os.path.join(out_dir, "cp_vs_ulysses_kv.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"figure saved: {out_png}\nfigure saved: {out_pdf}")
    plt.show()


if __name__ == "__main__":
    plot_cp_vs_ulysses_kv()

