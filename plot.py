import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# 1) 샘플 results DataFrame
# =========================
np.random.seed(0)

methods = ["OLS", "Huber", "LAD"]
dfs = [1, 2, 3, 20, np.inf]   # df=∞ 포함
n_rep = 40                    # Monte Carlo 반복 수

rows = []
for method in methods:
    for df in dfs:
        # 인위적 성능 패턴: OLS > Huber > LAD, df 작을수록(heavy-tail) penalty 큼
        base = {"OLS": 0.120, "Huber": 0.090, "LAD": 0.080}[method]
        tail_penalty = 0.050 / (df if np.isfinite(df) else 20.0)  # ∞면 약한 penalty
        mean_mse = base + tail_penalty
        # rep별 변동
        mse_vals = np.random.normal(loc=mean_mse, scale=0.010, size=n_rep)
        for rep, mse in enumerate(mse_vals, start=1):
            rows.append({"method": method, "df": df, "rep": rep, "mse": float(mse)})

results = pd.DataFrame(rows)
# print(results.head())

# ======================================
# 2) 집계: df × method별 평균/CI 계산
# ======================================
summary = (
    results
    .groupby(["method", "df"], as_index=False)
    .agg(mse_mean=("mse", "mean"),
         mse_std =("mse", "std"),
         R       =("mse", "count"))
)
summary["mse_se"] = summary["mse_std"] / np.sqrt(summary["R"].clip(lower=1))
z = 1.96
summary["ci_low"]  = summary["mse_mean"] - z * summary["mse_se"]
summary["ci_high"] = summary["mse_mean"] + z * summary["mse_se"]

# ===========================
# 3) 플롯 스타일(논문용)
# ===========================
mpl.rcParams.update({
    "figure.figsize": (5.8, 3.2),   # 4–6" x 2.5–4"
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# ===========================
# 4) x축 equal spacing 세팅
# ===========================
df_order  = [1, 2, 3, 20, np.inf]
df_labels = ["1", "2", "3", "20", "∞"]
df_to_pos = {df: i for i, df in enumerate(df_order)}  # 0..4 균등 간격
summary["xpos"] = summary["df"].map(df_to_pos)

# ===========================
# 5) 색상 팔레트(색각이상 배려)
# ===========================
palette = {
    "OLS":   "#0072B2",  # Blue
    "Huber": "#009E73",  # Green
    "LAD":   "#D55E00",  # Vermillion
}
default_colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
methods_sorted = sorted(summary["method"].unique())

# ===========================
# 6) 플롯
# ===========================
fig, ax = plt.subplots()

for i, method in enumerate(methods_sorted):
    sub = summary[summary["method"] == method].sort_values("xpos")
    color = palette.get(method, default_colors[i % len(default_colors)])
    # 평균 라인
    ax.plot(sub["xpos"], sub["mse_mean"],
            marker="o", linewidth=2, markersize=5,
            label=method, color=color)
    # 95% CI 밴드
    ax.fill_between(sub["xpos"], sub["ci_low"], sub["ci_high"],
                    alpha=0.15, color=color, linewidth=0)

# y축만 은은한 보조선
ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)

# 라벨/범례
ax.set_xticks(range(len(df_order)))
ax.set_xticklabels(df_labels)
ax.set_xlabel("Degrees of freedom (t-distribution)")
ax.set_ylabel("Mean Squared Error (MSE)")
ax.set_title("MSE vs degrees of freedom by method")
ax.legend(title="Method", frameon=False, ncol=len(methods_sorted))

ax.margins(x=0.03, y=0.05)
plt.tight_layout()
plt.show()

# 저장이 필요하면 다음 한 줄:
# plt.savefig("mse_vs_df_by_method.pdf", bbox_inches="tight")