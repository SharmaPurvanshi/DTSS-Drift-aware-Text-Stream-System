from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import ks_2samp


try:
    from alibi_detect.cd import KSDrift
except Exception:
    KSDrift = None
def _as_2d(X: np.ndarray) -> np.ndarray:
   
    X = np.asarray(X, dtype=float)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X[None, :]
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return X


def _stack_buffer(buf: list[np.ndarray]) -> np.ndarray:
    
    if len(buf) == 0:
        return np.zeros((0, 0), dtype=float)
    rows = [_as_2d(x).reshape(1, -1) for x in buf]
    return np.vstack(rows).astype(float)


def _finite_or(x: float, default: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return float(default)
    return xf if np.isfinite(xf) else float(default)


def _permutation_p_value(observed: float, permuted: np.ndarray) -> float:
 
    observed = _finite_or(observed, 0.0)
    permuted = np.asarray(permuted, dtype=float)
    permuted = permuted[np.isfinite(permuted)]

    if permuted.size == 0:
        return 1.0

    return float((1.0 + np.sum(permuted >= observed)) / (permuted.size + 1.0))


def _num_permutations_from_env(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None or str(value).strip() == "":
        return int(default)
    try:
        return int(max(1, float(value)))
    except Exception:
        return int(default)



class AlibiKSTest:
    
    def __init__(
        self,
        alpha: float,
        l_hist_size: int,
        l_new_size: int,
        hist_min_size: int,
        correction: str = "bonferroni",
        alternative: str = "two-sided",
    ):
        if KSDrift is None:
            raise ImportError("alibi-detect not installed. Run: pip install alibi-detect")

        self.alpha = float(alpha)
        self.l_hist_size = int(max(1, l_hist_size))
        self.l_new_size = int(max(1, l_new_size))
        self.hist_min_size = int(max(1, hist_min_size))

        self.correction = str(correction)
        self.alternative = str(alternative)

        self.l_hist: list[np.ndarray] = []
        self.l_new: list[np.ndarray] = []

        self.last_score: float = float("nan")
        self.last_p_value: float = 1.0
        self.last_statistic: float = 0.0

    @staticmethod
    def _get_first_present(dct: dict, keys: list[str], default=None):
        for k in keys:
            if k in dct and dct[k] is not None:
                return dct[k]
        return default

    def _manual_ks_stats(self, X_hist: np.ndarray, X_new: np.ndarray) -> tuple[float, float]:
       
        X_hist = _as_2d(X_hist)
        X_new = _as_2d(X_new)

        if (
            X_hist.size == 0
            or X_new.size == 0
            or X_hist.ndim != 2
            or X_new.ndim != 2
            or X_hist.shape[1] == 0
            or X_hist.shape[1] != X_new.shape[1]
        ):
            return 0.0, 1.0

        d_vals = []
        p_vals = []
        for j in range(X_hist.shape[1]):
            d_stat, p_val = ks_2samp(
                X_hist[:, j],
                X_new[:, j],
                alternative=self.alternative,
                mode="auto",
            )
            d_vals.append(_finite_or(d_stat, 0.0))
            p_vals.append(_finite_or(p_val, 1.0))

        max_d = float(np.nanmax(d_vals)) if d_vals else 0.0
        min_p = float(np.nanmin(p_vals)) if p_vals else 1.0
        return max_d, min_p

    def _parse_alibi_output(
        self,
        pred: dict,
        X_hist: np.ndarray,
        X_new: np.ndarray,
    ) -> tuple[bool, float, float]:
        data = pred.get("data", pred) if isinstance(pred, dict) else {}

        is_drift_raw = data.get("is_drift", False)
        is_drift_arr = np.asarray(is_drift_raw).reshape(-1)
        is_drift = bool(np.any(is_drift_arr)) if is_drift_arr.size else False

        p_feat = self._get_first_present(
            data,
            keys=[
                "p_val_feature",
                "p_vals_feature",
                "p_val_per_feature",
                "p_value_feature",
                "p_value_per_feature",
            ],
            default=None,
        )

        if p_feat is not None:
            p_feat = np.asarray(p_feat, dtype=float).reshape(-1)
            min_p = float(np.nanmin(p_feat)) if p_feat.size else np.nan
        else:
            p_raw = data.get("p_val", np.nan)
            p_arr = np.asarray(p_raw, dtype=float).reshape(-1)
            min_p = float(np.nanmin(p_arr)) if p_arr.size else np.nan

        dist_feat = self._get_first_present(
            data,
            keys=["distance_feature", "dist_feature", "distance", "dist"],
            default=None,
        )

        if dist_feat is not None:
            dist_feat = np.asarray(dist_feat, dtype=float).reshape(-1)
            max_d = float(np.nanmax(dist_feat)) if dist_feat.size else np.nan
        else:
            max_d = np.nan

        fallback_d, fallback_p = self._manual_ks_stats(X_hist, X_new)

        max_d = _finite_or(max_d, fallback_d)
        min_p = _finite_or(min_p, fallback_p)
        min_p = float(np.clip(min_p, 0.0, 1.0))

        return bool(is_drift), max_d, min_p

    def detect_drift(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)

        # 1. Fill history only. Do not also put these points into l_new.
        if len(self.l_hist) < self.hist_min_size:
            self.l_hist.append(z)
            self.last_score = 0.0
            self.last_p_value = 1.0
            self.last_statistic = 0.0
            return False, [0.0, 1.0]

        # 2. After history is ready, collect a separate new/test buffer.
        self.l_new.append(z)

        if len(self.l_new) < self.l_new_size:
    # Sample-and-hold: keep the most recent test result instead of zeroing.
            return False, [self.last_statistic, self.last_p_value]

        X_hist = _stack_buffer(self.l_hist[-self.l_hist_size:]).astype(np.float32)
        X_new = _stack_buffer(self.l_new).astype(np.float32)

        if (
            X_hist.size == 0
            or X_new.size == 0
            or X_hist.ndim != 2
            or X_new.ndim != 2
            or X_hist.shape[1] == 0
            or X_hist.shape[1] != X_new.shape[1]
        ):
            self.l_new = []
            self.last_score = 0.0
            self.last_p_value = 1.0
            self.last_statistic = 0.0
            return False, [0.0, 1.0]

        cd = KSDrift(
            x_ref=X_hist,
            p_val=self.alpha,
            correction=self.correction,
            alternative=self.alternative,
        )

        pred = cd.predict(X_new)
        is_drift, max_d, min_p = self._parse_alibi_output(pred, X_hist, X_new)

        self.last_p_value = float(min_p)
        self.last_statistic = float(max_d)
        self.last_score = float(1.0 - min_p)

        # 4. Update rolling history only after the test.
        self.l_hist.extend(self.l_new)
        if len(self.l_hist) > self.l_hist_size:
            self.l_hist = self.l_hist[-self.l_hist_size:]

        self.l_new = []

        return bool(is_drift), [float(max_d), float(min_p)]

    def ks_stats_between(self, X_ref, X_new):
        
        max_d, _ = self._manual_ks_stats(_as_2d(X_ref), _as_2d(X_new))
        return {"score": float(max_d)}


class MMDTest:
    
    
    def __init__(
        self,
        l_hist_size: int,
        l_new_size: int,
        hist_min_size: int,
        lambda_mmd: Optional[float] = None,
        gamma: float | None = None,
        alpha: Optional[float] = None,
        n_permutations: int = 199,
        random_state: int = 42,
    ):
        self.l_hist_size = int(max(1, l_hist_size))
        self.l_new_size = int(max(1, l_new_size))
        self.hist_min_size = int(max(1, hist_min_size))

        # Backward compatibility:
        # Older code passed lambda_mmd as a fixed threshold. In the calibrated
        # version, if alpha is not explicitly supplied, lambda_mmd is interpreted
        # as the p-value significance level when it is in (0, 1).
        if alpha is None:
            if lambda_mmd is not None and 0.0 < float(lambda_mmd) < 1.0:
                alpha = float(lambda_mmd)
            else:
                alpha = 0.05

        self.alpha = float(alpha)
        self.gamma = gamma

        self.n_permutations = _num_permutations_from_env(
            "DTSS_MMD_PERMUTATIONS",
            int(n_permutations),
        )
        self.rng = np.random.default_rng(int(random_state))

        self.l_hist: list[np.ndarray] = []
        self.l_new: list[np.ndarray] = []

        self.last_score: float = float("nan")
        self.last_p_value: float = 1.0
        self.last_statistic: float = 0.0
        self.last_critical_value: float = 0.0

    @staticmethod
    def _rbf_kernel_from_gamma(X: np.ndarray, gamma: float) -> np.ndarray:
        dists = cdist(X, X, metric="sqeuclidean")
        return np.exp(-gamma * dists)

    def _select_gamma(self, Z: np.ndarray) -> float:
        if self.gamma is not None:
            return float(self.gamma)

        if Z.shape[0] < 2:
            return 1.0

        d = pdist(Z, metric="euclidean")
        d = d[np.isfinite(d)]
        if d.size == 0:
            return 1.0

        med = float(np.median(d))
        if med <= 0.0:
            return 1.0

        return float(1.0 / (2.0 * med * med))

    @staticmethod
    def _mmd2_from_kernel(K: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
        idx_x = np.asarray(idx_x, dtype=int)
        idx_y = np.asarray(idx_y, dtype=int)

        m = idx_x.size
        n = idx_y.size

        if m < 2 or n < 2:
            return 0.0

        Kxx = K[np.ix_(idx_x, idx_x)]
        Kyy = K[np.ix_(idx_y, idx_y)]
        Kxy = K[np.ix_(idx_x, idx_y)]

        sum_xx = float(Kxx.sum() - np.trace(Kxx))
        sum_yy = float(Kyy.sum() - np.trace(Kyy))

        mmd2 = (
            sum_xx / float(m * (m - 1))
            + sum_yy / float(n * (n - 1))
            - 2.0 * float(Kxy.mean())
        )

        return float(mmd2)

    def _mmd2(self, X, Y):
        X = _as_2d(np.asarray(X, dtype=float))
        Y = _as_2d(np.asarray(Y, dtype=float))
        if X.shape[0] < 2 or Y.shape[0] < 2:
            return 0.0

        Z = np.vstack([X, Y])
        gamma = self._select_gamma(Z)
        K = self._rbf_kernel_from_gamma(Z, gamma)

        m = X.shape[0]
        idx_x = np.arange(m)
        idx_y = np.arange(m, m + Y.shape[0])

        return self._mmd2_from_kernel(K, idx_x, idx_y)

    def _permutation_test(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, float, float]:
        X = _as_2d(X)
        Y = _as_2d(Y)

        m = X.shape[0]
        n = Y.shape[0]

        if m < 2 or n < 2:
            return 0.0, 1.0, 0.0

        Z = np.vstack([X, Y])
        gamma = self._select_gamma(Z)
        K = self._rbf_kernel_from_gamma(Z, gamma)

        idx_x = np.arange(m)
        idx_y = np.arange(m, m + n)

        observed = self._mmd2_from_kernel(K, idx_x, idx_y)

        perm_stats = np.zeros(self.n_permutations, dtype=float)
        all_idx = np.arange(m + n)

        for b in range(self.n_permutations):
            perm = self.rng.permutation(all_idx)
            px = perm[:m]
            py = perm[m:]
            perm_stats[b] = self._mmd2_from_kernel(K, px, py)

        p_value = _permutation_p_value(observed, perm_stats)
        critical = float(np.quantile(perm_stats, 1.0 - self.alpha)) if perm_stats.size else 0.0

        return float(observed), float(p_value), float(critical)

    def detect_drift(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)

        # Fill history only; do not contaminate l_new during warmup.
        if len(self.l_hist) < self.hist_min_size:
            self.l_hist.append(z)
            self.last_score = 0.0
            self.last_p_value = 1.0
            self.last_statistic = 0.0
            self.last_critical_value = 0.0
            return False, [0.0, 1.0, 0.0]

        self.l_new.append(z)

        if len(self.l_new) < self.l_new_size:
    # Sample-and-hold: keep the most recent test result instead of zeroing.
            return False, [self.last_statistic, self.last_p_value, self.last_critical_value]

        X_hist = _stack_buffer(self.l_hist[-self.l_hist_size:])
        X_new = _stack_buffer(self.l_new)

        mmd2_val, p_value, critical = self._permutation_test(X_hist, X_new)

        self.last_statistic = float(mmd2_val)
        self.last_p_value = float(p_value)
        self.last_critical_value = float(critical)
        self.last_score = float(1.0 - p_value)

        is_drift = bool(p_value < self.alpha)

        self.l_hist.extend(self.l_new)
        if len(self.l_hist) > self.l_hist_size:
            self.l_hist = self.l_hist[-self.l_hist_size:]

        self.l_new = []

        return is_drift, [float(mmd2_val), float(p_value), float(critical)]

    def mmd2_between(self, X_ref, X_new):
        return self._mmd2(X_ref, X_new)



class EnergyTest:
    

    def __init__(
        self,
        l_hist_size: int,
        l_new_size: int,
        hist_min_size: int,
        lambda_energy: Optional[float] = None,
        alpha: Optional[float] = None,
        n_permutations: int = 199,
        random_state: int = 42,
    ):
        self.l_hist_size = int(max(1, l_hist_size))
        self.l_new_size = int(max(1, l_new_size))
        self.hist_min_size = int(max(1, hist_min_size))

        # Backward compatibility:
        # Older code passed lambda_energy as a fixed threshold. In the calibrated
        # version, if alpha is not explicitly supplied, lambda_energy is interpreted
        # as the p-value significance level when it is in (0, 1).
        if alpha is None:
            if lambda_energy is not None and 0.0 < float(lambda_energy) < 1.0:
                alpha = float(lambda_energy)
            else:
                alpha = 0.05

        self.alpha = float(alpha)

        self.n_permutations = _num_permutations_from_env(
            "DTSS_ENERGY_PERMUTATIONS",
            int(n_permutations),
        )
        self.rng = np.random.default_rng(int(random_state))

        self.l_hist: list[np.ndarray] = []
        self.l_new: list[np.ndarray] = []

        self.last_score: float = float("nan")
        self.last_p_value: float = 1.0
        self.last_statistic: float = 0.0
        self.last_critical_value: float = 0.0

    @staticmethod
    def _energy_from_distance(D: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
        idx_x = np.asarray(idx_x, dtype=int)
        idx_y = np.asarray(idx_y, dtype=int)

        m = idx_x.size
        n = idx_y.size

        if m < 2 or n < 2:
            return 0.0

        Dxx = D[np.ix_(idx_x, idx_x)]
        Dyy = D[np.ix_(idx_y, idx_y)]
        Dxy = D[np.ix_(idx_x, idx_y)]

        sum_xx = float(Dxx.sum() - np.trace(Dxx))
        sum_yy = float(Dyy.sum() - np.trace(Dyy))

        term_xy = 2.0 * float(Dxy.mean())
        term_xx = sum_xx / float(m * (m - 1))
        term_yy = sum_yy / float(n * (n - 1))

        energy = term_xy - term_xx - term_yy
        return float(energy)

    def _energy2(self, X, Y):
        X = _as_2d(np.asarray(X, dtype=float))
        Y = _as_2d(np.asarray(Y, dtype=float))
        if X.shape[0] < 2 or Y.shape[0] < 2:
            return 0.0

        Z = np.vstack([X, Y])
        D = cdist(Z, Z, metric="euclidean")

        m = X.shape[0]
        idx_x = np.arange(m)
        idx_y = np.arange(m, m + Y.shape[0])

        return self._energy_from_distance(D, idx_x, idx_y)

    def _permutation_test(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, float, float]:
        X = _as_2d(X)
        Y = _as_2d(Y)

        m = X.shape[0]
        n = Y.shape[0]

        if m < 2 or n < 2:
            return 0.0, 1.0, 0.0

        Z = np.vstack([X, Y])
        D = cdist(Z, Z, metric="euclidean")

        idx_x = np.arange(m)
        idx_y = np.arange(m, m + n)

        observed = self._energy_from_distance(D, idx_x, idx_y)

        perm_stats = np.zeros(self.n_permutations, dtype=float)
        all_idx = np.arange(m + n)

        for b in range(self.n_permutations):
            perm = self.rng.permutation(all_idx)
            px = perm[:m]
            py = perm[m:]
            perm_stats[b] = self._energy_from_distance(D, px, py)

        p_value = _permutation_p_value(observed, perm_stats)
        critical = float(np.quantile(perm_stats, 1.0 - self.alpha)) if perm_stats.size else 0.0

        return float(observed), float(p_value), float(critical)

    def detect_drift(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)

        # Fill history only; do not contaminate l_new during warmup.
        if len(self.l_hist) < self.hist_min_size:
            self.l_hist.append(z)
            self.last_score = 0.0
            self.last_p_value = 1.0
            self.last_statistic = 0.0
            self.last_critical_value = 0.0
            return False, [0.0, 1.0, 0.0]

        self.l_new.append(z)

        if len(self.l_new) < self.l_new_size:
        # Sample-and-hold: keep the most recent test result instead of zeroing.
            return False, [self.last_statistic, self.last_p_value, self.last_critical_value]

        X_hist = _stack_buffer(self.l_hist[-self.l_hist_size:])
        X_new = _stack_buffer(self.l_new)

        energy_val, p_value, critical = self._permutation_test(X_hist, X_new)

        self.last_statistic = float(energy_val)
        self.last_p_value = float(p_value)
        self.last_critical_value = float(critical)
        self.last_score = float(1.0 - p_value)

        is_drift = bool(p_value < self.alpha)

        self.l_hist.extend(self.l_new)
        if len(self.l_hist) > self.l_hist_size:
            self.l_hist = self.l_hist[-self.l_hist_size:]

        self.l_new = []

        return is_drift, [float(energy_val), float(p_value), float(critical)]

    def energy2_between(self, X_ref, X_new):
        return self._energy2(X_ref, X_new)



class KLDivergence:
    @staticmethod
    def get_kld(p, q):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        eps = 1e-12
        p = p + eps
        q = q + eps

        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def gaussian(mu0, Sigma0, mu1, Sigma1):
        mu0 = np.asarray(mu0, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        S0 = np.asarray(Sigma0, dtype=float)
        S1 = np.asarray(Sigma1, dtype=float)

        d = mu0.shape[0]
        eps = 1e-6
        S0 = S0 + eps * np.eye(d)
        S1 = S1 + eps * np.eye(d)

        invS1 = np.linalg.inv(S1)
        diff = (mu1 - mu0).reshape(-1, 1)

        term1 = np.log(np.linalg.det(S1) / np.linalg.det(S0))
        term2 = np.trace(invS1 @ S0)
        term3 = float(diff.T @ invS1 @ diff)

        kld = 0.5 * (term1 - d + term2 + term3)
        return float(kld)
