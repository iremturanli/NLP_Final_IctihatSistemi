"""İstatistiksel anlamlılık — akademik ciddiyetin kanıtı.

- Bootstrap %95 güven aralığı (per-query metriklerden)
- İki sistem arası paired bootstrap / paired t-test
- Çoklu seed ortalama ± std

Reviewer "0.933 vs 0.800 anlamlı mı?" diye sorduğunda cevap burada.
"""
import numpy as np


def bootstrap_ci(per_query_values, n_boot=10000, ci=0.95, seed=42):
    """Tek metrik için bootstrap güven aralığı.

    per_query_values: list/array (her sorgu için metrik değeri)
    Döndürür: (mean, lo, hi)
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_query_values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap_test(values_a, values_b, n_boot=10000, seed=42):
    """İki sistemin paired bootstrap testi (aynı sorgular üzerinde).

    values_a, values_b: aynı sırada per-query metrik değerleri (sistem A ve B)
    Döndürür: dict {diff_mean, p_value, ci_lo, ci_hi}
    H0: A ile B arasında fark yok. p = boot örneklerinde farkın işaret değiştirme oranı.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), "paired test için eşit uzunluk gerekir"
    n = len(a)
    diffs = a - b
    observed = diffs.mean()
    boot_diffs = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    # iki yönlü p: ortalama farkın 0'ı geçme oranı
    if observed >= 0:
        p = 2 * np.mean(boot_diffs <= 0)
    else:
        p = 2 * np.mean(boot_diffs >= 0)
    p = min(1.0, float(p))
    lo = float(np.percentile(boot_diffs, 2.5))
    hi = float(np.percentile(boot_diffs, 97.5))
    return {"diff_mean": float(observed), "p_value": p, "ci_lo": lo, "ci_hi": hi}


def paired_t_test(values_a, values_b):
    """Klasik paired t-test (scipy varsa)."""
    try:
        from scipy import stats as ss
        t, p = ss.ttest_rel(values_a, values_b)
        return {"t": float(t), "p_value": float(p)}
    except ImportError:
        return {"t": None, "p_value": None, "note": "scipy yok"}


def multi_seed_summary(seed_metric_dicts):
    """Çoklu seed sonuçlarını ortala.

    seed_metric_dicts: list of dict (her seed için {metric: value})
    Döndürür: {metric: {"mean":..., "std":...}}
    """
    keys = seed_metric_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in seed_metric_dicts]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def significance_stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_comparison(name_a, per_query_a, name_b, per_query_b, metric_key):
    """İki sistemi tek metrikte karşılaştır, anlamlılıkla birlikte yazdırılabilir string."""
    a = [m[metric_key] for m in per_query_a]
    b = [m[metric_key] for m in per_query_b]
    mean_a, lo_a, hi_a = bootstrap_ci(a)
    mean_b, lo_b, hi_b = bootstrap_ci(b)
    test = paired_bootstrap_test(a, b)
    stars = significance_stars(test["p_value"])
    return (
        f"{metric_key}:\n"
        f"  {name_a:<28} {mean_a:.3f}  [{lo_a:.3f}, {hi_a:.3f}]\n"
        f"  {name_b:<28} {mean_b:.3f}  [{lo_b:.3f}, {hi_b:.3f}]\n"
        f"  Δ = {test['diff_mean']:+.3f}  (p={test['p_value']:.4f} {stars})"
    )


if __name__ == "__main__":
    # Demo
    np.random.seed(0)
    a = np.random.binomial(1, 0.93, 100)
    b = np.random.binomial(1, 0.80, 100)
    pa = [{"recall@1": x} for x in a]
    pb = [{"recall@1": x} for x in b]
    print(format_comparison("fine-tuned", pa, "baseline", pb, "recall@1"))
