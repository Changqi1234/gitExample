import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
# =========
# 1. 读取 LT 数据
# =========
lt_path = "LTYielddata2024.csv"
lt_raw = pd.read_csv(lt_path, header=3)

# 先清理列名两端空格
lt_raw.columns = lt_raw.columns.str.strip()

print("LT columns:")
print(list(lt_raw.columns))

lt = lt_raw.rename(columns={
    "HeatTrt": "Treatment",
    "Rotten fruit #": "rotten_count",
    "Rotten fruitWeight (g)": "rotten_weight",
    "Non-rotten fruit#": "healthy_count",
    "Non-rotten fruitWeight (g)": "healthy_weight",
    "Total #": "total_count",
    "Total Weight (g)": "total_weight"
}).copy()

lt["Cultivar"] = lt["Cultivar"].astype(str).str.strip()
lt["Treatment"] = lt["Treatment"].astype(str).str.strip()
lt["Plot"] = pd.to_numeric(lt["Plot"], errors="coerce")

# outcome: by-weight rotten proportion
lt["y_weight"] = lt["rotten_weight"] / lt["total_weight"]

# 构造 long-term pair_id
# 假设同 cultivar 内 control plot = OTC plot + 4
def make_lt_pair_id(row):
    plot = row["Plot"]
    if row["Cultivar"] == "St":
        return plot if row["Treatment"] == "OTC" else plot - 4
    elif row["Cultivar"] == "MQ":
        return plot if row["Treatment"] == "OTC" else plot - 4
    else:
        raise ValueError("Unexpected cultivar")

lt["pair_index"] = lt.apply(make_lt_pair_id, axis=1)
lt["regime"] = "LT"

# 宽表：每对一行
lt_wide = (
    lt.pivot_table(
        index=["Cultivar", "pair_index", "regime"],
        columns="Treatment",
        values="y_weight",
        aggfunc="first"
    )
    .reset_index()
)

lt_wide["d"] = lt_wide["OTC"] - lt_wide["Control"]
lt_pairs = lt_wide[["Cultivar", "pair_index", "regime", "d"]].copy()

# =========
# 2. 读取 Acute 数据
# =========
acute_path = "Acute HS-Yield_RawData 2024.xlsx"
acute = pd.read_excel(acute_path, sheet_name="Acute Heat stress", header=4)

acute = acute.rename(columns={
    "Rotten fruitWeight (g)": "rotten_weight",
    " Total Weight (g)": "total_weight",
    "Non-rotten fruitWeight (g)": "healthy_weight",
    "Rotten fruit #": "rotten_count",
    "Non-rotten fruit#": "healthy_count",
    "Total #": "total_count",
    "# OTC/ plot": "rep"
}).copy()

acute = acute.dropna(subset=["Cultivar", "Treatment", "rep"])
acute["Cultivar"] = acute["Cultivar"].astype(str).str.strip()
acute["Treatment"] = acute["Treatment"].astype(str).str.strip()
acute["rep"] = acute["rep"].astype(int)

acute["y_weight"] = acute["rotten_weight"] / acute["total_weight"]

# 只保留 A/B/C/D 和对应 control
acute = acute[acute["Treatment"].isin(["A", "B", "C", "D", "AC", "BC", "CC", "DC"])].copy()

# treatment family
acute["regime"] = acute["Treatment"].replace({
    "AC": "A",
    "BC": "B",
    "CC": "C",
    "DC": "D"
})

acute["group"] = np.where(
    acute["Treatment"].isin(["A", "B", "C", "D"]),
    "treat",
    "control"
)

acute_wide = (
    acute.pivot_table(
        index=["Cultivar", "rep", "regime"],
        columns="group",
        values="y_weight",
        aggfunc="first"
    )
    .reset_index()
)

acute_wide["d"] = acute_wide["treat"] - acute_wide["control"]
acute_pairs = acute_wide[["Cultivar", "rep", "regime", "d"]].copy()
acute_pairs = acute_pairs.rename(columns={"rep": "pair_index"})

# =========
# 3. 合并 pair-level 数据
# =========
pair_df = pd.concat([lt_pairs, acute_pairs], ignore_index=True)

# 编码
regime_order = ["LT", "A", "B", "C", "D"]
pair_df["regime"] = pd.Categorical(pair_df["regime"], categories=regime_order, ordered=True)
pair_df["regime_idx"] = pair_df["regime"].cat.codes

# cultivar: St/Stevens = 0, MQ = 1
pair_df["cultivar_code"] = pair_df["Cultivar"].replace({
    "St": 0,
    "Stevens": 0,
    "MQ": 1
}).astype(int)

print(pair_df)

# =========
# 4. 贝叶斯层级模型
# =========
d = pair_df["d"].values
regime_idx = pair_df["regime_idx"].values
cultivar = pair_df["cultivar_code"].values
n_regimes = len(regime_order)

with pm.Model() as model:
    # hyperpriors
    mu0 = pm.Normal("mu0", mu=0, sigma=10)
    tau = pm.HalfNormal("tau", sigma=1)

    # regime-level effects
    mu_regime = pm.Normal("mu_regime", mu=mu0, sigma=tau, shape=n_regimes)

    # cultivar effect
    beta_cultivar = pm.Normal("beta_cultivar", mu=0, sigma=10)

    # residual SD
    sigma = pm.HalfNormal("sigma", sigma=1)

    # mean model
    mu_obs = mu_regime[regime_idx] + beta_cultivar * cultivar

    # likelihood
    d_obs = pm.Normal("d_obs", mu=mu_obs, sigma=sigma, observed=d)

    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=5,
        cores=1,
        target_accept=0.97,
        random_seed=123,
        return_inferencedata=False,
        compute_convergence_checks=False
    )
    

# =========
# 5. 查看结果
# =========

mu_samples = trace.get_values("mu_regime", combine=True)
# shape 通常是 (n_samples, 5)

posterior_mu = mu_samples.mean(axis=0)
for name, val in zip(regime_order, posterior_mu):
    print(f"{name}: {val:.4f}")


beta_samples = trace.get_values("beta_cultivar", combine=True)
sigma_samples = trace.get_values("sigma", combine=True)
tau_samples = trace.get_values("tau", combine=True)
mu0_samples = trace.get_values("mu0", combine=True)

print("beta_cultivar mean =", beta_samples.mean())
print("sigma mean =", sigma_samples.mean())
print("tau mean =", tau_samples.mean())
print("mu0 mean =", mu0_samples.mean())


# =========================================
# A. 取 posterior samples
# =========================================
mu_regime_samps = trace.get_values("mu_regime", combine=True)   # shape: (S, n_regimes)
beta_samps = trace.get_values("beta_cultivar", combine=True)    # shape: (S,)
sigma_samps = trace.get_values("sigma", combine=True)           # shape: (S,)
tau_samps = trace.get_values("tau", combine=True)               # shape: (S,)
mu0_samps = trace.get_values("mu0", combine=True)               # shape: (S,)

print("mu_regime_samps shape:", mu_regime_samps.shape)
print("beta_samps shape:", beta_samps.shape)
print("sigma_samps shape:", sigma_samps.shape)

# =========================================
# B. 1) Raw pair differences sanity check
# =========================================
print("\n" + "="*60)
print("1) RAW PAIR DIFFERENCES CHECK")
print("="*60)
raw_summary = pair_df.groupby("regime")["d"].agg(["mean", "median", "std", "count"])
print(raw_summary)

# 可选画图
plt.figure(figsize=(8, 5))
pair_df.boxplot(column="d", by="regime")
plt.axhline(0, color="red", linestyle="--")
plt.title("Raw pair differences by regime")
plt.suptitle("")
plt.ylabel("d = treat - control")
plt.show()

# =========================================
# C. 2) Posterior mean / CI for each regime
# =========================================
print("\n" + "="*60)
print("2) POSTERIOR SUMMARY FOR EACH REGIME")
print("="*60)

posterior_mu = mu_regime_samps.mean(axis=0)
posterior_ci = np.quantile(mu_regime_samps, [0.025, 0.975], axis=0)

for i, name in enumerate(regime_order):
    print(
        f"{name}: mean={posterior_mu[i]:.4f}, "
        f"95% CrI=[{posterior_ci[0, i]:.4f}, {posterior_ci[1, i]:.4f}]"
    )

print(f"\nmu0 mean = {mu0_samps.mean():.4f}")
print(f"tau mean = {tau_samps.mean():.4f}")
print(f"sigma mean = {sigma_samps.mean():.4f}")
print(f"beta_cultivar mean = {beta_samps.mean():.4f}")

# =========================================
# D. 3) Pairwise posterior comparisons
# =========================================
print("\n" + "="*60)
print("3) PAIRWISE POSTERIOR COMPARISONS")
print("="*60)

regime_map = {name: i for i, name in enumerate(regime_order)}

def compare_regimes(name1, name2, mu_samples, regime_map):
    i = regime_map[name1]
    j = regime_map[name2]
    diff = mu_samples[:, i] - mu_samples[:, j]
    p_gt = np.mean(diff > 0)
    ci = np.quantile(diff, [0.025, 0.975])
    print(f"{name1} - {name2}:")
    print(f"  P({name1} > {name2}) = {p_gt:.3f}")
    print(f"  95% CrI = [{ci[0]:.4f}, {ci[1]:.4f}]")
    return diff

lt_minus_a = compare_regimes("LT", "A", mu_regime_samps, regime_map)
lt_minus_b = compare_regimes("LT", "B", mu_regime_samps, regime_map)
lt_minus_c = compare_regimes("LT", "C", mu_regime_samps, regime_map)
lt_minus_d = compare_regimes("LT", "D", mu_regime_samps, regime_map)

# =========================================
# E. 4) Posterior predictive check
# =========================================
print("\n" + "="*60)
print("4) POSTERIOR PREDICTIVE CHECK")
print("="*60)

S = len(beta_samps)
n = len(d)

# 抽一部分 posterior draws 做 replicate
n_ppc = min(300, S)
idx = np.random.choice(S, size=n_ppc, replace=False)

d_rep = np.zeros((n_ppc, n))

for k, s in enumerate(idx):
    mu_obs_s = mu_regime_samps[s, regime_idx] + beta_samps[s] * cultivar
    d_rep[k, :] = np.random.normal(loc=mu_obs_s, scale=sigma_samps[s], size=n)

# 比较整体 mean 和 sd
obs_mean = d.mean()
rep_mean = d_rep.mean(axis=1)

obs_sd = d.std(ddof=1)
rep_sd = d_rep.std(axis=1, ddof=1)

print(f"Observed mean(d) = {obs_mean:.4f}")
print("Posterior predictive mean(d) 95% interval =",
      np.quantile(rep_mean, [0.025, 0.975]))

print(f"Observed sd(d) = {obs_sd:.4f}")
print("Posterior predictive sd(d) 95% interval =",
      np.quantile(rep_sd, [0.025, 0.975]))

# 画图：mean(d)
plt.figure(figsize=(7, 4))
plt.hist(rep_mean, bins=30, alpha=0.7, edgecolor="black")
plt.axvline(obs_mean, color="red", linewidth=2, label="Observed mean(d)")
plt.title("Posterior predictive check: mean(d)")
plt.legend()
plt.show()

# 画图：sd(d)
plt.figure(figsize=(7, 4))
plt.hist(rep_sd, bins=30, alpha=0.7, edgecolor="black")
plt.axvline(obs_sd, color="red", linewidth=2, label="Observed sd(d)")
plt.title("Posterior predictive check: sd(d)")
plt.legend()
plt.show()

# 画图：整体分布粗比较
plt.figure(figsize=(7, 4))
plt.hist(d, bins=15, alpha=0.6, density=True, edgecolor="black", label="Observed d")
for k in range(min(20, n_ppc)):
    plt.hist(d_rep[k, :], bins=15, alpha=0.05, density=True, color="gray")
plt.title("Observed d vs posterior predictive replicates")
plt.legend()
plt.show()

# =========================================
# F. 5) Residual check
# =========================================
print("\n" + "="*60)
print("5) RESIDUAL CHECK")
print("="*60)

mu_regime_mean = mu_regime_samps.mean(axis=0)
beta_mean = beta_samps.mean()

fitted = mu_regime_mean[regime_idx] + beta_mean * cultivar
resid = d - fitted

pair_df["fitted"] = fitted
pair_df["resid"] = resid

resid_summary = pair_df.groupby("regime")["resid"].agg(["mean", "median", "std", "count"])
print(resid_summary)

# residual vs fitted
plt.figure(figsize=(7, 4))
plt.scatter(fitted, resid)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.title("Residual vs fitted")
plt.show()

# residual histogram
plt.figure(figsize=(7, 4))
plt.hist(resid, bins=15, edgecolor="black", alpha=0.7)
plt.axvline(0, color="red", linestyle="--")
plt.title("Residual histogram")
plt.xlabel("Residual")
plt.show()

# residual by regime
plt.figure(figsize=(8, 5))
pair_df.boxplot(column="resid", by="regime")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals by regime")
plt.suptitle("")
plt.ylabel("Residual")
plt.show()

# =========================================
# G. 6) Shrinkage check
# =========================================
print("\n" + "="*60)
print("6) SHRINKAGE CHECK")
print("="*60)

raw_mean = pair_df.groupby("regime")["d"].mean().reindex(regime_order)
post_mean = pd.Series(posterior_mu, index=regime_order)

shrink_df = pd.DataFrame({
    "raw_mean_d": raw_mean,
    "posterior_mu": post_mean
})
print(shrink_df)

plt.figure(figsize=(7, 5))
x = np.arange(len(regime_order))
plt.scatter(x, raw_mean.values, label="raw mean(d)", s=80)
plt.scatter(x, post_mean.values, label="posterior mu_g", s=80)
for i in range(len(regime_order)):
    plt.plot([x[i], x[i]], [raw_mean.values[i], post_mean.values[i]], color="gray", alpha=0.6)
plt.xticks(x, regime_order)
plt.axhline(0, color="red", linestyle="--")
plt.ylabel("Effect")
plt.title("Shrinkage check: raw mean(d) vs posterior mu_g")
plt.legend()
plt.show()

# =========================================
# H. 7) Credible interval reasonableness
# =========================================
print("\n" + "="*60)
print("7) CREDIBLE INTERVAL CHECK")
print("="*60)

ci_width = posterior_ci[1, :] - posterior_ci[0, :]
ci_df = pd.DataFrame({
    "regime": regime_order,
    "posterior_mean": posterior_mu,
    "ci_lower": posterior_ci[0, :],
    "ci_upper": posterior_ci[1, :],
    "ci_width": ci_width
})
print(ci_df)

plt.figure(figsize=(8, 5))
for i, name in enumerate(regime_order):
    plt.plot([i, i], [posterior_ci[0, i], posterior_ci[1, i]], color="black")
    plt.scatter(i, posterior_mu[i], s=80)
plt.xticks(np.arange(len(regime_order)), regime_order)
plt.axhline(0, color="red", linestyle="--")
plt.ylabel("Posterior effect")
plt.title("Posterior means and 95% credible intervals")
plt.show()

# =========================================
# I. 8) Nonparametric sanity check
#    这里做简单的 sign-based check
# =========================================
print("\n" + "="*60)
print("8) NONPARAMETRIC SANITY CHECK")
print("="*60)

sign_check = pair_df.groupby("regime")["d"].apply(lambda x: np.mean(x > 0))
print("Proportion of positive pair differences by regime:")
print(sign_check)

# 也可以看中位数
median_check = pair_df.groupby("regime")["d"].median()
print("\nMedian pair differences by regime:")
print(median_check)

# =========================================
# J. 9) Final checklist summary
# =========================================
print("\n" + "="*60)
print("9) FINAL CHECKLIST")
print("="*60)

rep_mean_ci = np.quantile(rep_mean, [0.025, 0.975])
rep_sd_ci = np.quantile(rep_sd, [0.025, 0.975])

print("Observed mean(d) within PPC interval?:",
      rep_mean_ci[0] <= obs_mean <= rep_mean_ci[1])

print("Observed sd(d) within PPC interval?:",
      rep_sd_ci[0] <= obs_sd <= rep_sd_ci[1])

print("Residual means by regime (should be near 0):")
print(resid_summary["mean"])

print("\nShrinkage table:")
print(shrink_df)

print("\nPosterior CI table:")
print(ci_df)

print("\nNonparametric direction check:")
print(sign_check)
