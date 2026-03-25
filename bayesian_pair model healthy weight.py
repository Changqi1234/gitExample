import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# =========
# 1. 读取 LT 数据
# =========
lt_path = "LTYielddata2024.csv"
lt_raw = pd.read_csv(lt_path, header=3)

# 清理列名空格
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

# ===== outcome 改成 healthy fruit total weight =====
lt["y_healthy"] = pd.to_numeric(lt["healthy_weight"], errors="coerce")

# 构造 long-term pair_id
# 假设同 cultivar 内 control plot = OTC plot + 4
def make_lt_pair_id(row):
    plot = row["Plot"]
    if pd.isna(plot):
        return np.nan
    if row["Cultivar"] in ["St", "MQ", "Stevens"]:
        return plot if row["Treatment"] == "OTC" else plot - 4
    else:
        raise ValueError(f"Unexpected cultivar: {row['Cultivar']}")

lt["pair_index"] = lt.apply(make_lt_pair_id, axis=1)
lt["regime"] = "LT"

# 宽表：每对一行
lt_wide = (
    lt.pivot_table(
        index=["Cultivar", "pair_index", "regime"],
        columns="Treatment",
        values="y_healthy",
        aggfunc="first"
    )
    .reset_index()
)

# 只保留成对完整的
lt_wide = lt_wide.dropna(subset=["OTC", "Control"]).copy()

# pair difference
lt_wide["d"] = lt_wide["OTC"] - lt_wide["Control"]
lt_pairs = lt_wide[["Cultivar", "pair_index", "regime", "d"]].copy()

# =========
# 2. 读取 Acute 数据
# =========
acute_path = "Acute HS-Yield_RawData 2024.xlsx"
acute = pd.read_excel(acute_path, sheet_name="Acute Heat stress", header=4)

acute.columns = acute.columns.str.strip()

print("Acute columns:")
print(list(acute.columns))

acute = acute.rename(columns={
    "Rotten fruitWeight (g)": "rotten_weight",
    "Total Weight (g)": "total_weight",
    "Non-rotten fruitWeight (g)": "healthy_weight",
    "Rotten fruit #": "rotten_count",
    "Non-rotten fruit#": "healthy_count",
    "Total #": "total_count",
    "# OTC/ plot": "rep"
}).copy()

acute = acute.dropna(subset=["Cultivar", "Treatment", "rep"])
acute["Cultivar"] = acute["Cultivar"].astype(str).str.strip()
acute["Treatment"] = acute["Treatment"].astype(str).str.strip()
acute["rep"] = pd.to_numeric(acute["rep"], errors="coerce")
acute = acute.dropna(subset=["rep"]).copy()
acute["rep"] = acute["rep"].astype(int)

# ===== outcome 改成 healthy fruit total weight =====
acute["y_healthy"] = pd.to_numeric(acute["healthy_weight"], errors="coerce")

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
        values="y_healthy",
        aggfunc="first"
    )
    .reset_index()
)

# 只保留成对完整的
acute_wide = acute_wide.dropna(subset=["treat", "control"]).copy()

acute_wide["d"] = acute_wide["treat"] - acute_wide["control"]
acute_pairs = acute_wide[["Cultivar", "rep", "regime", "d"]].copy()
acute_pairs = acute_pairs.rename(columns={"rep": "pair_index"})

# =========
# 3. 合并 pair-level 数据
# =========
pair_df = pd.concat([lt_pairs, acute_pairs], ignore_index=True)

# 统一 cultivar 名称
pair_df["Cultivar"] = pair_df["Cultivar"].replace({
    "Stevens": "St"
})

pair_df = pair_df.dropna(subset=["Cultivar", "d"]).copy()

# 编码
regime_order = ["LT", "A", "B", "C", "D"]
pair_df["regime"] = pd.Categorical(pair_df["regime"], categories=regime_order, ordered=True)
pair_df = pair_df.dropna(subset=["regime"]).copy()
pair_df["regime_idx"] = pair_df["regime"].cat.codes

pair_df["cultivar_code"] = pair_df["Cultivar"].map({
    "St": 0,
    "MQ": 1
})

pair_df = pair_df.dropna(subset=["cultivar_code"]).copy()
pair_df["cultivar_code"] = pair_df["cultivar_code"].astype(int)

print("\nPair-level data:")
print(pair_df)

# =========
# 4. 贝叶斯层级模型（修正版：non-centered + Student-t + regime-specific sigma）
# =========
d = pair_df["d"].values.astype(float)
regime_idx = pair_df["regime_idx"].values
cultivar = pair_df["cultivar_code"].values
n_regimes = len(regime_order)

with pm.Model() as model:
    # -------------------------
    # Hyperpriors
    # -------------------------
    mu0 = pm.Normal("mu0", mu=0, sigma=80)
    tau = pm.HalfNormal("tau", sigma=80)

    # non-centered regime effects
    z_regime = pm.Normal("z_regime", mu=0, sigma=1, shape=n_regimes)
    mu_regime = pm.Deterministic("mu_regime", mu0 + z_regime * tau)

    # cultivar main effect
    beta_cultivar = pm.Normal("beta_cultivar", mu=0, sigma=80)

    # regime-specific residual scales
    sigma_regime = pm.HalfNormal("sigma_regime", sigma=150, shape=n_regimes)

    # Student-t degrees of freedom
    nu_minus_two = pm.Exponential("nu_minus_two", lam=1/10)
    nu = pm.Deterministic("nu", nu_minus_two + 2)

    # mean model
    mu_obs = mu_regime[regime_idx] + beta_cultivar * cultivar

    # likelihood
    d_obs = pm.StudentT(
        "d_obs",
        nu=nu,
        mu=mu_obs,
        sigma=sigma_regime[regime_idx],
        observed=d
    )

    trace = pm.sample(
        draws=3000,
        tune=4000,
        chains=4,
        cores=1,
        init="jitter+adapt_diag",
        target_accept=0.99,
        random_seed=123,
        return_inferencedata=False,
        compute_convergence_checks=False
    )

# =========
# 5. 查看结果
# =========
mu_samples = trace.get_values("mu_regime", combine=True)               # shape: (S, 5)
beta_samples = trace.get_values("beta_cultivar", combine=True)        # shape: (S,)
sigma_regime_samples = trace.get_values("sigma_regime", combine=True) # shape: (S, 5)
tau_samples = trace.get_values("tau", combine=True)
mu0_samples = trace.get_values("mu0", combine=True)
nu_samples = trace.get_values("nu", combine=True)

posterior_mu = mu_samples.mean(axis=0)
for name, val in zip(regime_order, posterior_mu):
    print(f"{name}: {val:.4f}")

print("beta_cultivar mean =", beta_samples.mean())
print("sigma_regime mean =", sigma_regime_samples.mean(axis=0))
print("tau mean =", tau_samples.mean())
print("mu0 mean =", mu0_samples.mean())
print("nu mean =", nu_samples.mean())

print("mu_regime_samps shape:", mu_samples.shape)
print("beta_samps shape:", beta_samples.shape)
print("sigma_regime_samps shape:", sigma_regime_samples.shape)
print("nu_samps shape:", nu_samples.shape)

# =========================================
# B. 1) Raw pair differences sanity check
# =========================================
print("\n" + "="*60)
print("1) RAW PAIR DIFFERENCES CHECK (HEALTHY FRUIT WEIGHT)")
print("="*60)
raw_summary = pair_df.groupby("regime")["d"].agg(["mean", "median", "std", "count"])
print(raw_summary)

plt.figure(figsize=(8, 5))
pair_df.boxplot(column="d", by="regime")
plt.axhline(0, color="red", linestyle="--")
plt.title("Raw pair differences by regime (healthy fruit total weight)")
plt.suptitle("")
plt.ylabel("d = treat - control (healthy fruit total weight)")
plt.show()

# =========================================
# C. 2) Posterior mean / CI for each regime
# =========================================
print("\n" + "="*60)
print("2) POSTERIOR SUMMARY FOR EACH REGIME")
print("="*60)

posterior_ci = np.quantile(mu_samples, [0.025, 0.975], axis=0)

for i, name in enumerate(regime_order):
    print(
        f"{name}: mean={posterior_mu[i]:.4f}, "
        f"95% CrI=[{posterior_ci[0, i]:.4f}, {posterior_ci[1, i]:.4f}]"
    )

print(f"\nmu0 mean = {mu0_samples.mean():.4f}")
print(f"tau mean = {tau_samples.mean():.4f}")
print(f"nu mean = {nu_samples.mean():.4f}")
print(f"beta_cultivar mean = {beta_samples.mean():.4f}")

sigma_regime_mean = sigma_regime_samples.mean(axis=0)
print("sigma_regime means by regime:")
for i, name in enumerate(regime_order):
    print(f"  {name}: {sigma_regime_mean[i]:.4f}")

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

lt_minus_a = compare_regimes("LT", "A", mu_samples, regime_map)
lt_minus_b = compare_regimes("LT", "B", mu_samples, regime_map)
lt_minus_c = compare_regimes("LT", "C", mu_samples, regime_map)
lt_minus_d = compare_regimes("LT", "D", mu_samples, regime_map)

# =========================================
# E. 4) Posterior predictive check
# =========================================
print("\n" + "="*60)
print("4) POSTERIOR PREDICTIVE CHECK")
print("="*60)

S = len(beta_samples)
n = len(d)

n_ppc = min(300, S)
idx = np.random.choice(S, size=n_ppc, replace=False)

d_rep = np.zeros((n_ppc, n))

for k, s in enumerate(idx):
    mu_obs_s = mu_samples[s, regime_idx] + beta_samples[s] * cultivar
    sigma_obs_s = sigma_regime_samples[s, regime_idx]
    nu_s = nu_samples[s]

    d_rep[k, :] = mu_obs_s + sigma_obs_s * np.random.standard_t(df=nu_s, size=n)

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

plt.figure(figsize=(7, 4))
plt.hist(rep_mean, bins=30, alpha=0.7, edgecolor="black")
plt.axvline(obs_mean, color="red", linewidth=2, label="Observed mean(d)")
plt.title("Posterior predictive check: mean(d)")
plt.legend()
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(rep_sd, bins=30, alpha=0.7, edgecolor="black")
plt.axvline(obs_sd, color="red", linewidth=2, label="Observed sd(d)")
plt.title("Posterior predictive check: sd(d)")
plt.legend()
plt.show()

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

mu_regime_mean = mu_samples.mean(axis=0)
beta_mean = beta_samples.mean()

fitted = mu_regime_mean[regime_idx] + beta_mean * cultivar
resid = d - fitted

pair_df["fitted"] = fitted
pair_df["resid"] = resid

resid_summary = pair_df.groupby("regime")["resid"].agg(["mean", "median", "std", "count"])
print(resid_summary)

plt.figure(figsize=(7, 4))
plt.scatter(fitted, resid)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.title("Residual vs fitted")
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(resid, bins=15, edgecolor="black", alpha=0.7)
plt.axvline(0, color="red", linestyle="--")
plt.title("Residual histogram")
plt.xlabel("Residual")
plt.show()

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
# =========================================
print("\n" + "="*60)
print("8) NONPARAMETRIC SANITY CHECK")
print("="*60)

sign_check = pair_df.groupby("regime")["d"].apply(lambda x: np.mean(x > 0))
print("Proportion of positive pair differences by regime:")
print(sign_check)

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
