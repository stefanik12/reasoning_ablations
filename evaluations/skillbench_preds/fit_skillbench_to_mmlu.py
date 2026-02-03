#!/usr/bin/env python3
# Leave-one-model-out (LOMO) cross-model generalization for analytical OLS, multi-target.
# For each held-out model: fit on all other models, evaluate on held-out; repeat for every MMLU task
# INCLUDING the aggregate column "mmlu".
# Report only aggregate (averaged over held-out folds), but separately for each target column.
# Also write a CSV with the aggregate results.
# Filenames match the provided scripts:
#   schoolbench_{repo_last}_metrics.csv and lmeval_{repo_last}.csv

import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau

def resolve_paths(root, model):
    # Resolve CSV paths exactly as written by the evaluation scripts.
    short = model.split("/")[-1]
    sb = os.path.join(root, f"schoolbench_{short}_metrics.csv")
    lm = os.path.join(root, f"lmeval_{short}.csv")
    if os.path.exists(sb) and os.path.exists(lm): return sb, lm
    # Case-insensitive fallback for filesystems or runs with different casing.
    short_l = short.lower()
    allcsv = glob.glob(os.path.join(root, "*.csv"))
    sb2 = next((p for p in allcsv if os.path.basename(p).lower() == f"schoolbench_{short_l}_metrics.csv"), None)
    lm2 = next((p for p in allcsv if os.path.basename(p).lower() == f"lmeval_{short_l}.csv"), None)
    if sb2 and lm2: return sb2, lm2
    raise SystemExit(f"Missing expected CSVs for {model} in {root}\nExpected: {os.path.basename(sb)} and {os.path.basename(lm)}")

def infer_skills(cols):
    # Extract skill names from columns like: skill.<skill>.base_top1_acc / skill.<skill>.gap / etc.
    out = set()
    for c in cols:
        m = re.match(r"^skill\.([^.]+)\..+$", c)
        if m: out.add(m.group(1))
    return sorted(out)

def select_features(cols_all, kind, metric, skill_regex = None):
    # Build the list of per-skill feature columns to use.
    ss = infer_skills(cols_all)
    if skill_regex: ss = [s for s in ss if re.search(skill_regex, s)]
    cols = [f"skill.{s}.gap" for s in ss] if kind == "gap" else [f"skill.{s}.{kind}_{metric}" for s in ss]
    return cols

def fit_ols(X, y):
    # Analytical OLS via pseudoinverse: beta = pinv([1, X]) y.
    Xa = np.c_[np.ones((len(X), 1)), X]
    b = np.linalg.pinv(Xa) @ y
    return b[1:], float(b[0])

def predict(X, w, b0): return X @ w + b0  # Linear prediction with intercept.

def r2(y, yhat):
    ssr = np.sum((y - yhat) ** 2); sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ssr / sst) if sst > 0 else 0.0


def mse(y, yhat): return float(np.mean((y - yhat) ** 2))

def zfit(X):
    mu = X.mean(0); sig = X.std(0); sig[sig == 0] = 1.0
    return mu, sig

def zapply(X, mu, sig): return (X - mu) / sig

def model_df(root, model, join_on_branch):
    sb_csv, lm_csv = resolve_paths(root, model)
    sb = pd.read_csv(sb_csv); lm = pd.read_csv(lm_csv)
    key = ["step", "branch"] if join_on_branch and "branch" in sb.columns and "branch" in lm.columns else ["step"]
    if key[0] in sb:
        df = pd.merge(sb, lm, on = key, how = "inner")
    else:
        df = pd.concat([sb, lm], axis=1)

    df["model"] = model
    print(f"Loaded {model}: {os.path.basename(sb_csv)} + {os.path.basename(lm_csv)} -> rows = {len(df)}")
    return df

def build_XY(df, feat_cols, y_cols, standardize, mu = None, sig = None):
    d = df[feat_cols + y_cols].dropna()
    X = d[feat_cols].to_numpy(np.float64)
    Y = d[y_cols].to_numpy(np.float64)
    if standardize:
        if mu is None or sig is None: mu, sig = zfit(X)
        X = zapply(X, mu, sig)
    return X, Y, mu, sig

def plot_feature_development(df_all, models, feat_cols, plot_cols, out_dir, by_branch):
    # Plot each selected feature vs step for each model (separate plot per feature).
    # If branch exists and by_branch is True, average within (model, step) across branches before plotting.
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok = True)
    cols_needed = ["model", "step"] + (["branch"] if ("branch" in df_all.columns and by_branch) else []) + plot_cols
    d = df_all[cols_needed].dropna(subset = plot_cols).copy()

    if "branch" in d.columns and by_branch:
        g = d.groupby(["model", "step"], as_index = False)[plot_cols].mean()
        d = g

    # Ensure numeric step for sorting if possible.
    if not np.issubdtype(d["step"].dtype, np.number):
        d["step"] = pd.to_numeric(d["step"], errors = "coerce")
        d = d.dropna(subset = ["step"])

    for c in plot_cols:
        plt.figure()
        for m in models:
            dm = d[d["model"] == m].sort_values("step")
            if len(dm) == 0: continue
            plt.plot(dm["step"].to_numpy(), dm[c].to_numpy(), label = m)
        plt.xlabel("step"); plt.ylabel(c); plt.title(c)
        if len(models) <= 12: plt.legend(fontsize = 8)
        fp = os.path.join(out_dir, f"feature_{re.sub(r'[^A-Za-z0-9._+-]+','-',c)}.png")
        plt.tight_layout(); plt.savefig(fp, dpi = 150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action = "append", required = True, help = 'Repeatable HF repo id, e.g. "allenai/OLMo-7B"')
    ap.add_argument("--dir", default = ".", help = "Directory containing the CSVs")
    ap.add_argument("--out_csv", default = "lomo_mmlu_tasks_report.csv", help = "Output CSV path for aggregate report")
    ap.add_argument("--join_on_branch", action = "store_true")
    ap.add_argument("--feature_kind", choices = ["base", "cf", "gap"], default = "base")
    ap.add_argument("--feature_metric", default = "top1_acc")
    ap.add_argument("--skill_regex", default = None)
    ap.add_argument("--standardize", action = "store_true")
    ap.add_argument("--corr", choices = ["pearson", "spearman", "kendall"], default = "pearson")
    args = ap.parse_args()

    def metrics(y, yhat):
        return {"n": int(len(y)), "r2": r2(y, yhat), "corr": corr(y, yhat), "mse": mse(y, yhat)}

    def corr_fn(kind):
        if kind == "pearson":
            def f(y, yhat):
                if len(y) < 2: return 0.0
                a = y - y.mean();
                b = yhat - yhat.mean()
                d = np.sqrt((a @ a) * (b @ b))
                return float((a @ b) / d) if d > 0 else 0.0

            return f
        if kind == "spearman":
            return lambda y, yhat: float(spearmanr(y, yhat, nan_policy="omit").correlation or 0.0) if len(
                y) >= 2 else 0.0
        if kind == "kendall":
            return lambda y, yhat: float(kendalltau(y, yhat, nan_policy="omit").correlation or 0.0) if len(
                y) >= 2 else 0.0
        raise ValueError(kind)

    corr = corr_fn(args.corr)

    dfs = {m: model_df(args.dir, m, args.join_on_branch) for m in args.model}

    # Determine MMLU target columns: include aggregate "mmlu" and all subtasks "mmlu_*".
    all_cols = set().union(*[set(df.columns) for df in dfs.values()])
    y_cols = ["mmlu"] + sorted([c for c in all_cols if c.startswith("mmlu_")])
    y_cols = [c for c in y_cols if all(c in df.columns for df in dfs.values())]
    if not y_cols: raise SystemExit("No MMLU target columns found.")

    # Determine a common feature set: intersection across all models.
    cand = select_features(list(all_cols), args.feature_kind, args.feature_metric, args.skill_regex)
    feat_cols = [c for c in cand if all(c in df.columns for df in dfs.values())]
    if not feat_cols:
        ex = "skill.<skill>.gap" if args.feature_kind == "gap" else f"skill.<skill>.{args.feature_kind}_{args.feature_metric}"
        raise SystemExit(f"No common features across all models; expected columns like {ex}")
    print(f"Common features = {len(feat_cols)}, targets = {len(y_cols)}")

    # Leave-one-model-out folds; accumulate per-target metrics per method per fold.
    models = list(dfs.keys())
    acc = {(t, m): {"r2": [], "corr": [], "mse": [], "n": []}
           for t in y_cols for m in ["baseline_constant", "baseline_mean_feature_1D", "ols_full"]}

    # LOMO: accumulate per-target metrics per method per fold.
    for held_out in models:
        train_df = pd.concat([dfs[m] for m in models if m != held_out], ignore_index = True)
        test_df = dfs[held_out]
        fold_y = [c for c in y_cols if c in train_df.columns and c in test_df.columns]
        if not fold_y: continue

        Xtr, Ytr, mu, sig = build_XY(train_df, feat_cols, fold_y, args.standardize)
        Xte, Yte, _, _ = build_XY(test_df, feat_cols, fold_y, args.standardize, mu = mu, sig = sig)
        if len(Xtr) == 0 or len(Xte) == 0: continue

        Yhat0 = np.repeat(Ytr.mean(0, keepdims = True), Yte.shape[0], axis = 0)

        # Baseline: mean-feature 1D OLS per target.
        x1tr = Xtr.mean(1, keepdims = True); x1te = Xte.mean(1, keepdims = True)
        Yhat1 = np.zeros_like(Yte)
        for j in range(Ytr.shape[1]):
            w1, b1 = fit_ols(x1tr, Ytr[:, j]); Yhat1[:, j] = predict(x1te, w1, b1)

        # Full OLS per target.
        Yhat2 = np.zeros_like(Yte)
        for j in range(Ytr.shape[1]):
            w, b0 = fit_ols(Xtr, Ytr[:, j]); Yhat2[:, j] = predict(Xte, w, b0)

        for j, task in enumerate(fold_y):
            for name, Yhat in [("baseline_constant", Yhat0), ("baseline_mean_feature_1D", Yhat1), ("ols_full", Yhat2)]:
                y = Yte[:, j]; yhat = Yhat[:, j]
                acc[(task, name)]["r2"].append(r2(y, yhat))
                acc[(task, name)]["corr"].append(corr(y, yhat))
                acc[(task, name)]["mse"].append(mse(y, yhat))
                acc[(task, name)]["n"].append(len(y))

    # Aggregate report per target per method; print and write CSV.
    rows = []
    for task in y_cols:
        for method in ["baseline_constant", "baseline_mean_feature_1D", "ols_full"]:
            a = acc[(task, method)]
            rows.append({
                "task": task, "method": method, "folds": len(a["r2"]), "total_n": int(sum(a["n"])),
                "avg_r2": np.nan if not a["r2"] else float(np.mean(a["r2"])),
                f"avg_{args.corr}": np.nan if not a["corr"] else float(np.mean(a["corr"])),
                "avg_mse": np.nan if not a["mse"] else float(np.mean(a["mse"]))
            })
    out = pd.DataFrame(rows).sort_values(["task", "method"]).reset_index(drop = True)
    out.to_csv(args.out_csv, index = False)

    print("\nAggregate report (avg over held-out folds), per target:")
    for task in y_cols:
        sub = out[out["task"] == task]
        if sub["folds"].max() == 0: print(f"{task} : no folds"); continue
        s0 = sub[sub["method"] == "baseline_constant"].iloc[0]
        s1 = sub[sub["method"] == "baseline_mean_feature_1D"].iloc[0]
        s2 = sub[sub["method"] == "ols_full"].iloc[0]
        print(f"{task} | const r2 = {s0.avg_r2:.4f} {args.corr} = {s0[f'avg_{args.corr}']:.4f} mse = {s0.avg_mse:.6f} | "
              f"mean1d r2 = {s1.avg_r2:.4f} {args.corr} = {s1[f'avg_{args.corr}']:.4f} mse = {s1.avg_mse:.6f} | "
              f"ols r2 = {s2.avg_r2:.4f} {args.corr} = {s2[f'avg_{args.corr}']:.4f} mse = {s2.avg_mse:.6f}")
    print(f"\nWrote CSV: {args.out_csv}")

if __name__ == "__main__":
    main()
