#!/usr/bin/env python3
# Plot paired base+cf curves from raw schoolbench_{repo_last}_metrics.csv (not merged).
# Unified handling of aggregate and per-skill metrics.
# Always plots base/cf pairs; plots gap when present.
# Filters only by --skill_regex (skill.* only) and --metric_regex (all metrics).

import argparse, glob, os, re
import numpy as np, pandas as pd

def resolve_schoolbench(root, model):
    short = model.split("/")[-1]
    p = os.path.join(root, f"schoolbench_{short}_metrics.csv")
    if os.path.exists(p): return p
    short_l = short.lower()
    for q in glob.glob(os.path.join(root, "*.csv")):
        if os.path.basename(q).lower() == f"schoolbench_{short_l}_metrics.csv": return q
    raise SystemExit(f"Missing metrics CSV for {model}")

def normalize_cols(df):
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df

def strip_quotes(s):
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')): return s[1:-1]
    return s

def load_metrics(root, model, average_over_branch):
    df = normalize_cols(pd.read_csv(resolve_schoolbench(root, model)))
    df["model"] = model
    df["step"] = pd.to_numeric(df["step"], errors = "coerce")
    df = df.dropna(subset = ["step"])
    if average_over_branch and "branch" in df.columns:
        num = [c for c in df.columns if c not in ["model", "branch"] and pd.api.types.is_numeric_dtype(df[c])]
        df = df.groupby(["model", "step"], as_index = False)[num].mean()
    return df

def slug(s): return re.sub(r"[^A-Za-z0-9._+-]+", "-", s)

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors = "coerce")
    return df

def parse_groups(cols):
    # key -> {"base": col?, "cf": col?, "gap": col?}
    out = {}

    for c in cols:
        m = re.match(r"^(base|cf)_([^.]+)$", c)
        if m:
            out.setdefault(("agg", "aggregate", m.group(2)), {})[m.group(1)] = c
            continue
        m = re.match(r"^skill\.([^.]+)\.(base|cf)_([^.]+)$", c)
        if m:
            out.setdefault(("skill", m.group(1), m.group(3)), {})[m.group(2)] = c
            continue
        m = re.match(r"^skill\.([^.]+)\.gap$", c)
        if m:
            out.setdefault(("skill", m.group(1), "gap"), {})["gap"] = c

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action = "append", required = True)
    ap.add_argument("--dir", default = ".")
    ap.add_argument("--out_dir", default = "plots_pairs")
    ap.add_argument("--skill_regex", default = r".*")
    ap.add_argument("--metric_regex", default = r".*")
    ap.add_argument("--average_over_branch", action = "store_true")
    ap.add_argument("--rescale_90", action = "store_true")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok = True)

    s_re = re.compile(strip_quotes(args.skill_regex))
    m_re = re.compile(strip_quotes(args.metric_regex))

    df_all = pd.concat([load_metrics(args.dir, m, args.average_over_branch) for m in args.model],
                       ignore_index = True)
    models = args.model

    groups = parse_groups(df_all.columns)

    keys = []
    for (scope, name, metric), cols in groups.items():
        if not m_re.search(metric): continue
        if scope == "skill" and not s_re.search(name): continue
        if "base" in cols and "cf" in cols or "gap" in cols:
            keys.append((scope, name, metric))

    if not keys: raise SystemExit("No base/cf groups matched filters")

    use_cols = sorted({c for k in keys for c in groups[k].values()})
    df_all = coerce_numeric(df_all, use_cols)

    wrote = 0
    for scope, name, metric in sorted(keys):
        cols = groups[(scope, name, metric)]
        plt.figure(figsize = (9, 4)); ax = plt.gca(); all_y = []

        for m in models:
            for kind in ["base", "cf", "gap"]:
                c = cols.get(kind)
                if not c: continue
                dm = df_all[df_all["model"] == m][["step", c]].dropna().sort_values("step")
                if len(dm):
                    ax.plot(dm["step"], dm[c], label = f"{m} | {kind}")
                    all_y.append(dm[c].to_numpy())

        if not all_y:
            plt.close(); continue

        title = f"aggregate {metric}" if scope == "agg" else f"skill.{name}.{metric}"
        ax.set_title(title)
        ax.set_xlabel("step"); ax.set_ylabel(metric); ax.grid(True, alpha = 0.3)

        if len(models) * len(cols) <= 20: ax.legend(fontsize = 7)
        else: ax.legend(fontsize = 6, ncol = 2)

        if args.rescale_90:
            ys = np.concatenate(all_y)
            lo, hi = np.nanpercentile(ys, [5, 95])
            if lo < hi: ax.set_ylim(lo, hi)

        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{scope}__{slug(name)}__{slug(metric)}.png"), dpi = 150)
        plt.close()
        wrote += 1

    print(f"Wrote {wrote} PNG(s) into {args.out_dir}")

if __name__ == "__main__":
    main()
