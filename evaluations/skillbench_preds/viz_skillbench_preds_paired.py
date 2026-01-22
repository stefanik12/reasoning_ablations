#!/usr/bin/env python3
# Plot paired base+cf curves per skill from raw schoolbench_{repo_last}_metrics.csv (not merged).
# For each (skill, metric): draw base_<metric> and cf_<metric> together (optionally gap) in one figure.
# One line per (model, variant); saves PNGs. Robust to BOM/whitespace headers and quoted regex args.

import argparse, glob, os, re
import numpy as np, pandas as pd

def resolve_schoolbench(root, model):
    short = model.split("/")[-1]
    sb = os.path.join(root, f"schoolbench_{short}_metrics.csv")
    if os.path.exists(sb): return sb
    short_l = short.lower()
    allcsv = glob.glob(os.path.join(root, "*.csv"))
    sb2 = next((p for p in allcsv if os.path.basename(p).lower() == f"schoolbench_{short_l}_metrics.csv"), None)
    if sb2: return sb2
    raise SystemExit(f"Missing metrics CSV for {model} in {root}")

def normalize_cols(df):
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df

def strip_wrapping_quotes(s):
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')): return s[1:-1]
    return s

def load_metrics(root, model, average_over_branch):
    path = resolve_schoolbench(root, model)
    df = normalize_cols(pd.read_csv(path))
    if "step" not in df.columns: raise SystemExit(f"{os.path.basename(path)} missing column 'step'")
    df["model"] = model
    df["step"] = pd.to_numeric(df["step"], errors = "coerce")
    df = df.dropna(subset = ["step"])
    if average_over_branch and "branch" in df.columns:
        num = [c for c in df.columns if c not in ["model", "branch"] and pd.api.types.is_numeric_dtype(df[c])]
        df = df.groupby(["model", "step"], as_index = False)[num].mean()
    return df, path

def slug(s): return re.sub(r"[^A-Za-z0-9._+-]+", "-", s)

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors = "coerce")
    return df

def parse_skill_cols(cols):
    # Parse columns like skill.<skill>.(base|cf)_<metric> and skill.<skill>.gap
    # Returns mapping: (skill, metric) -> {"base": col, "cf": col, "gap": col?}
    pat = re.compile(r"^skill\.([^.]+)\.(base|cf)_([^.]+)$")
    out = {}
    for c in cols:
        m = pat.match(c)
        if m:
            skill, kind, metric = m.group(1), m.group(2), m.group(3)
            out.setdefault((skill, metric), {})[kind] = c
            continue
        m2 = re.match(r"^skill\.([^.]+)\.gap$", c)
        if m2:
            skill = m2.group(1)
            out.setdefault((skill, "gap"), {})["gap"] = c
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action = "append", required = True)
    ap.add_argument("--dir", default = ".")
    ap.add_argument("--out_dir", default = "plots_pairs")
    ap.add_argument("--skill_regex", default = r".*", help = "Regex over skill name (default: .*)")
    ap.add_argument("--metric_regex", default = r".*", help = "Regex over metric name (default: .*)")
    ap.add_argument("--average_over_branch", action = "store_true")
    ap.add_argument("--include_gap", action = "store_true", help = "Also plot skill.<skill>.gap when present")
    ap.add_argument("--max_plots", type = int, default = 0)
    ap.add_argument("--rescale_90", action = "store_true", help = "Set y-limits to 5th..95th percentiles per plot")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok = True)

    s_re = re.compile(strip_wrapping_quotes(args.skill_regex))
    m_re = re.compile(strip_wrapping_quotes(args.metric_regex))

    dfs = []
    for m in args.model:
        df, path = load_metrics(args.dir, m, args.average_over_branch)
        print(f"Loaded metrics {m}: {os.path.basename(path)} rows = {len(df)}")
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index = True)

    # Build (skill, metric) -> cols mapping from available columns.
    mapping = parse_skill_cols(df_all.columns)
    keys = [(s, met) for (s, met) in mapping.keys() if s_re.search(s) and m_re.search(met)]
    keys = sorted(keys)
    if args.max_plots > 0: keys = keys[:args.max_plots]
    if not keys: raise SystemExit("No (skill, metric) pairs matched skill_regex/metric_regex")

    # Coerce all referenced columns to numeric once.
    use_cols = sorted({c for k in keys for c in mapping[k].values()})
    df_all = coerce_numeric(df_all, use_cols)
    models = args.model
    wrote = 0

    for skill, metric in keys:
        cols = mapping[(skill, metric)]
        base_c = cols.get("base"); cf_c = cols.get("cf"); gap_c = cols.get("gap") if args.include_gap else None
        if metric == "gap" and not args.include_gap: continue
        if (not base_c or not cf_c) and not (args.include_gap and gap_c): continue  # need base+cf unless plotting gap
        plt.figure(figsize = (9, 4))
        ax = plt.gca()
        all_y = []

        # Plot base/cf lines per model.
        for m in models:
            if base_c:
                dm = df_all[df_all["model"] == m][["step", base_c]].dropna().sort_values("step")
                if len(dm): ax.plot(dm["step"], dm[base_c], label = f"{m} | base"); all_y.append(dm[base_c].to_numpy())
            if cf_c:
                dm = df_all[df_all["model"] == m][["step", cf_c]].dropna().sort_values("step")
                if len(dm): ax.plot(dm["step"], dm[cf_c], label = f"{m} | cf"); all_y.append(dm[cf_c].to_numpy())
            if gap_c:
                dm = df_all[df_all["model"] == m][["step", gap_c]].dropna().sort_values("step")
                if len(dm): ax.plot(dm["step"], dm[gap_c], label = f"{m} | gap"); all_y.append(dm[gap_c].to_numpy())

        if not all_y: plt.close(); continue
        ax.set_xlabel("step"); ax.set_ylabel(f"{skill}.{metric}")
        ax.set_title(f"skill.{skill} {(metric if metric == 'gap' else metric)} (base vs cf{' vs gap' if gap_c else ''})")
        ax.grid(True, alpha = 0.3)
        if len(models) * (2 + (1 if gap_c else 0)) <= 20: ax.legend(fontsize = 7)
        else: ax.legend(fontsize = 6, ncol = 2)

        if args.rescale_90:
            ys = np.concatenate(all_y)
            lo, hi = np.nanpercentile(ys, [5, 95])
            ax.set_ylim(lo, hi)

        plt.tight_layout()
        out = os.path.join(args.out_dir, f"skill_{slug(skill)}__{slug(metric)}.png")
        plt.savefig(out, dpi = 150); plt.close()
        wrote += 1

    print(f"Wrote {wrote} PNG(s) into {args.out_dir}")

if __name__ == "__main__":
    main()
