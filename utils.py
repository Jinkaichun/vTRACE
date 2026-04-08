"""
Utility functions for COVID-19 variant evolution modeling
Converted from MATLAB to Python
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import expit
from datetime import datetime, timedelta
import warnings

def trace_rinit(X_0, I_0, I_vac_0, J):
    """Initialize state vectors for particle filter"""
    X = np.tile(X_0[:, np.newaxis], (1, J))
    I_vac = np.tile(I_vac_0[:, np.newaxis], (1, J))
    I = np.tile(I_0[:, np.newaxis], (1, J))
    return X, I, I_vac

def trace_rmeas(X, time, eps2, covariate):
    """Measurement function with observation error"""
    # Find index for current time
    index = np.where(covariate['timeline'] == time)[0]
    if len(index) == 0:
        raise ValueError(f"Time {time} not found in covariate timeline")
    index = index[0]
    
    n = covariate['n'][index]  # total sequence number
    noise = np.random.randn(*X.shape)
    x_obs = X + (np.sqrt(X * (1 - X) / n) + eps2) * noise
    x_obs = np.maximum(x_obs, 0)
    x_obs = x_obs / x_obs.sum(axis=0, keepdims=True)
    return x_obs

def trace_dmeas(x_obs, X, time, eps2, covariate):
    """Measurement density function"""
    J = X.shape[1]
    
    # Find index for current time in observation times
    index_n = np.where(covariate['time_obs'] == time)[0]
    if len(index_n) == 0:
        return np.ones(J)  # Return uniform weights if time not found
    index_n = index_n[0]
    
    n = covariate['n'][index_n]  # total sequence number
    if n>10:
        # Calculate standard deviation for each variant
        std_dev = np.sqrt(X * (1 - X) / n) + eps2
        
        # Calculate likelihood
        x_obs_repeated = np.tile(x_obs[:, np.newaxis], (1, J))
        likli = norm.pdf(x_obs_repeated, X, std_dev)


        # Ignore lineages with very low or very high frequencies
        invalid_ind = np.where((x_obs < 0.005) | (x_obs > 0.995))[0]
        likli[invalid_ind, :] = 1
        
        # Replace zeros with smallest positive float
        likli[likli == 0] = np.finfo(float).eps
        
        # Product across variants
        likli = np.prod(likli, axis=0)
    else:
        likli=np.ones(J)
    return likli

def trace_rproc(X, I, I_vac, D, D_vac, B, k, r, gam, gam_vac, 
                lambda_, eps,P, time, covariate, deltat, freq_0=0.04):
    """Process model function"""
    n, J = X.shape
    m = D_vac.shape[1]
    
    mut = covariate['mut']
    N = covariate['N']
    vac = covariate['vac']
    start_rate = covariate['start_rate']
    
    # Find indices
    #index = np.where(covariate['timeline'] == time)[0][0]
    try:
        index = np.where(covariate['timeline'] == time)[0][0]
    except IndexError:
        breakpoint()
        raise
    index_last = np.where(covariate['timeline'] == time - deltat)[0][0]
    
    deltaN = N[index] - N[index_last]
    deltavac = vac[index, :] - vac[index_last, :]
    N_now = deltaN
    
    # 1) Decay
    I_inf_temp = I * np.exp(-lambda_ * deltat)
    I_vac_temp = I_vac * np.exp(-lambda_ * deltat)
    
    # 2) New immunity
    deltaI_inf = deltaN * X
    deltaI_vac = gam_vac * np.tile(deltavac[:, np.newaxis], (1, J))
    
    I_new = I_inf_temp+deltaI_inf
    I_vac_new = I_vac_temp +deltaI_vac
    
    # Calculate selection coefficients
    if np.isscalar(r) or len(r) == 1:
        C = np.exp(-D / r)
        S = C @ I
        C_vac = np.exp(-D_vac / r)
        S_vac = C_vac @ I_vac
        F = k * B[:, np.newaxis] - gam * S / P - gam * S_vac / P
    else:
        # Handle vector r
        r_rep = r.reshape(1, 1, -1)
        r_rep = np.tile(r_rep, (n, n, 1))
        C = np.exp(-D[:, :, np.newaxis] / r_rep)
        I_reshaped = I.reshape(n, 1, J)
        S = np.matmul(C.transpose(2, 0, 1), I_reshaped.transpose(2, 0, 1)).transpose(1, 2, 0)
        S = S.reshape(n, J)
        
        r_vac_rep = r.reshape(1, 1, -1)
        r_vac_rep = np.tile(r_vac_rep, (n, m, 1))
        C_vac = np.exp(-D_vac[:, :, np.newaxis] / r_vac_rep)
        I_vac_reshaped = I_vac.reshape(m, 1, J)
        S_vac = np.matmul(C_vac.transpose(2, 0, 1), I_vac_reshaped.transpose(2, 0, 1)).transpose(1, 2, 0)
        S_vac = S_vac.reshape(n, J)
        
        F = k * B[:, np.newaxis] - gam * S / P - gam * S_vac / P
    
    f = F - (F * X).sum(axis=0, keepdims=True)
    
    n_threhold=500
    # Bottleneck effect
    n_sample = np.ceil(eps * N_now+0.000001)
    if np.isscalar(n_sample):
        if n_sample > n_threhold:
            X = X + np.sqrt(X * (1 - X) / n_sample) * np.random.randn(n, J)
            X = np.maximum(X, 0)
            X = X / X.sum(axis=0, keepdims=True)
        else:
            X_num = np.zeros_like(X)
            if np.any(X > 1):
                print("Error: X contains values > 1.")
                breakpoint()  # 暂停执行
            if np.any(X < 0):
                print("Error: X contains values < 0.")
                breakpoint()
            if np.isnan(X).any():
                print("Error: X contains NaN values.")
                breakpoint()
            for j in range(J):
                X_num[:, j] = np.random.multinomial(n_sample, X[:, j])
            X = X_num / X_num.sum(axis=0, keepdims=True)
    else:
        if np.any(X > 1):
            print("Error: X contains values > 1.")
            breakpoint()  # 暂停执行
        if np.any(X < 0):
            print("Error: X contains values < 0.")
            breakpoint()
        if np.isnan(X).any():
            print("Error: X contains NaN values.")
            breakpoint()
        mask=n_sample > n_threhold  
        if np.any(mask):
            X[:,mask] = X[:,mask] + np.sqrt(X[:,mask] * (1 - X[:,mask]) / n_sample[mask]) * np.random.randn(n, np.sum(mask))
            X[:,mask] = np.maximum(X[:,mask], 0)
            X[:,mask] = X[:,mask] / X[:,mask].sum(axis=0, keepdims=True)
        if np.any(~mask):
            X_num = np.zeros((n, np.sum(~mask)))
            for j, col in enumerate(np.where(~mask)[0]):
                X_num[:, j] = np.random.multinomial(n_sample[col], X[:, col])
            X[:, ~mask] = X_num / X_num.sum(axis=0, keepdims=True)

    # X = X + np.sqrt(X * (1 - X) / n_sample) * np.random.randn(n, J)
    # X = np.maximum(X, 0)
    # X = X / X.sum(axis=0, keepdims=True)
    # Selection
    X = X * np.exp(f * deltat)
    X = X / X.sum(axis=0, keepdims=True)
    
    # Update immunity
    I = I_new
    I_vac = I_vac_new
    
    # Handle new variant emergence
    index_next = np.where(covariate['timeline'] == time + deltat)[0]
    if len(index_next) > 0:
        index_next = index_next[0]
        if mut[index_next] != 0:
            variant_idx = int(mut[index_next] - 1)  # Convert to 0-based index

            # 获取剩余行的索引
            other_indices = np.delete(np.arange(X.shape[0]), variant_idx)

            # 从原始矩阵中提取这些行
            X_other = X[other_indices, :]

            # 对这些行进行归一化（列方向上）
            X_other = X_other / X_other.sum(axis=0, keepdims=True)

            # 乘上比例 (1 - start_rate[variant_idx])
            adjustment = 1.0 - start_rate[variant_idx]
            X_other *= adjustment

            # 更新X矩阵
            X[variant_idx, :] = start_rate[variant_idx]
            X[other_indices, :] = X_other
    
    return f, X, I, I_vac

def birth_calculation(freq, start_freq, days):
    """Calculate birth index for variants"""
    judge = np.zeros_like(freq)
    T, n = freq.shape
    
    for i in range(1, days + 1):
        freq_i = np.roll(freq, i, axis=0)
        judge_i = freq_i > start_freq
        judge_i[:i, :] = 0
        judge = judge + judge_i
    
    birth_index = np.zeros(n, dtype=int)
    for k in range(n):
        indices = np.where(judge[:, k] == days)[0]
        if len(indices) > 0:
            birth_index[k] = indices[0] + 1 -days # Add 1 for 1-based indexing compatibility
    
    return birth_index


def death_calculation(freq, threshold):
    """
    Calculate the death index for each variant.
    
    Parameters:
    - freq: ndarray of shape (T, n), frequencies over time
    - threshold: float, frequency cutoff for "death"
    
    Returns:
    - death_index: ndarray of shape (n,), time index (1-based) when each variant 'dies'
    """
    T, n = freq.shape
    death_index = np.zeros(n, dtype=int)
    
    for k in range(n):
        freq_k = freq[:, k]
        peak_idx = np.argmax(freq_k)
        # Search after peak for first drop below threshold
        below_thresh = np.where(freq_k[peak_idx+1:] < threshold)[0]
        if len(below_thresh) > 0:
            death_index[k] = peak_idx + 1 + below_thresh[0] + 1  # +1 for offset, +1 for 1-based index
    
    return death_index


def fillNaN_columns(data):
    """Fill NaN values in columns"""
    data_interp = data.copy()
    rows, cols = data.shape
    
    for col in range(cols):
        current_col = data[:, col].copy()
        
        # Find first and last non-NaN indices
        non_nan_indices = np.where(~np.isnan(current_col))[0]
        
        if len(non_nan_indices) > 0:
            first_idx = non_nan_indices[0]
            last_idx = non_nan_indices[-1]
            
            # Fill beginning NaNs with 0
            current_col[:first_idx] = 0
            
            # Fill ending NaNs with last value
            current_col[last_idx + 1:] = current_col[last_idx]
            
            # Linear interpolation for middle NaNs
            if len(non_nan_indices) > 1:
                # Use pandas for easier interpolation
                series = pd.Series(current_col)
                series = series.interpolate(method='linear')
                current_col = series.values
        
        data_interp[:, col] = current_col
    
    return data_interp

def systematic_resampling(weights, J):
    """Systematic resampling for particle filter"""
    J = len(weights)
    # 均匀间隔采样点
    positions = (np.random.rand() + np.arange(J)) / J
    # 累积权重
    cdf = np.cumsum(weights)
    # searchsorted 返回第一个 cdf >= position 的索引
    return np.searchsorted(cdf, positions)

def pfilter(data, rproc, rinit, dmeas, covariate, params, tspan, deltat, J, 
            D, D_vac, B, X_0, I_0, I_vac_0, freq_0=0.04):
    """Particle filter for state estimation"""
    # Extract parameters
    P = covariate['P']
    k = params[0]
    r = params[1]
    gam = params[2]
    lambda_ = params[3]
    eps = params[4]
    eps2 = params[5]
    gam_vac = params[6]
    
    n_strain, n_vac = D_vac.shape
    t0, t_end = tspan
    timeline = np.arange(t0, t_end + deltat, deltat)
    n_time=len(timeline)
    # Initialize output arrays
    E_X = np.full((n_strain, len(timeline)), np.nan)
    E_I = np.full((n_strain, len(timeline)), np.nan)
    E_I_vac = np.full((n_vac, len(timeline)), np.nan)
    E_f = np.full((n_strain, len(timeline)), np.nan)

    # Quantile arrays
    Q_X_25 = np.full((n_strain, n_time), np.nan)
    Q_X_75 = np.full((n_strain, n_time), np.nan)
    Q_I_25 = np.full((n_strain, n_time), np.nan)
    Q_I_75 = np.full((n_strain, n_time), np.nan)
    Q_I_vac_25 = np.full((n_vac, n_time), np.nan)
    Q_I_vac_75 = np.full((n_vac, n_time), np.nan)
    Q_f_25 = np.full((n_strain, n_time), np.nan)
    Q_f_75 = np.full((n_strain, n_time), np.nan)
    
    # Initialize particles
    X, I, I_vac = rinit(X_0, I_0, I_vac_0, J)
    x_obs = data['x_obs']
    time_obs = data['time']
    con_likeli = 0
    
    for i, time in enumerate(timeline):
        E_X[:, i] = X.mean(axis=1)
        E_I[:, i] = I.mean(axis=1)
        E_I_vac[:, i] = I_vac.mean(axis=1)

        Q_X_25[:, i] = np.quantile(X, 0.05, axis=1)
        Q_X_75[:, i] = np.quantile(X, 0.95, axis=1)
        Q_I_25[:, i] = np.quantile(I, 0.05, axis=1)
        Q_I_75[:, i] = np.quantile(I, 0.95, axis=1)
        Q_I_vac_25[:, i] = np.quantile(I_vac, 0.05, axis=1)
        Q_I_vac_75[:, i] = np.quantile(I_vac, 0.95, axis=1)
        
        # Process step
        f_P, X_P, I_P, I_vac_P = rproc(X, I, I_vac, D, D_vac, B, 
                                       k, r, gam, gam_vac, lambda_, 
                                       eps, P, time, covariate, 
                                       deltat, freq_0)
        
        # Check if we have observation at next time
        time_ind = np.where(time_obs == time + deltat)[0]
        if len(time_ind) > 0:
            time_ind = time_ind[0]
            x = x_obs[:, time_ind]
            weights = dmeas(x, X_P, time + deltat, eps2, covariate)
            
            # Update likelihood
            if x.max() <= 1:
                con_likeli += np.log(np.nanmean(weights))
            
            # Resampling
            weights_norm = weights / weights.sum()
            ind = systematic_resampling(weights_norm, J)
            X = X_P[:, ind]
            I = I_P[:, ind]
            I_vac = I_vac_P[:, ind]
            f = f_P[:, ind]
        else:
            X = X_P
            I = I_P
            I_vac = I_vac_P
            f = f_P
        
        E_f[:, i] = f.mean(axis=1)
        Q_f_25[:, i] = np.quantile(f, 0.05, axis=1)
        Q_f_75[:, i] = np.quantile(f, 0.95, axis=1)

    return (con_likeli, timeline, 
        E_X, Q_X_25, Q_X_75, 
        E_I, Q_I_25, Q_I_75, 
        E_I_vac, Q_I_vac_25, Q_I_vac_75, 
        E_f, Q_f_25, Q_f_75)

def mif2(data, rproc, rinit, dmeas, covariate, params_guess, rw_sd, tspan, 
         deltat, J, M, a, D, D_vac, B, X_0, I_0, I_vac_0, freq_0=0.04):
    """Multiple Iterated Filtering (MIF2) for parameter estimation"""
    params_guess_trans = np.log(params_guess)
    params_swarm_trans = params_guess_trans.copy()
    n_par = len(params_guess_trans)
    params_swarm_iter = np.zeros((n_par, M))
    con_likeli_m = np.zeros(M)
    t0, t_end = tspan
    timeline = np.arange(t0, t_end + deltat, deltat)
    x_obs = data['x_obs']
    time_obs = data['time']
    params_swarm_trans = params_swarm_trans.reshape(-1,1) 
    P=covariate['P']
    for m in range(M):
        print(f"Iteration {m + 1}/{M}")
        rw_sd_m = rw_sd * a**(2 * m / 50)
        
        # Initialize particles
        X, I, I_vac = rinit(X_0, I_0, I_vac_0, J)
        
        for i, time in enumerate(timeline):
            # Add noise to parameters
            params_swarm_trans = params_swarm_trans + np.random.randn(n_par, J) * rw_sd_m[:, np.newaxis]
            params_swarm = np.exp(params_swarm_trans)
            
            # Extract parameters for each particle
            k = params_swarm[0, :]
            r = params_swarm[1, :]
            gam = params_swarm[2, :]
            lambda_ = params_swarm[3, :]
            eps = params_swarm[4, :]
            eps2 = params_swarm[5, :]
            gam_vac = params_swarm[6, :]

            
            # Process step
            _, X_P, I_P, I_vac_P = rproc(X, I, I_vac, D, D_vac, B, 
                                         k, r, gam, gam_vac, lambda_, 
                                         eps,  P, time, covariate, 
                                         deltat, freq_0)
            
            # Check for observation
            time_ind = np.where(time_obs == time + deltat)[0]
            if len(time_ind) > 0:
                time_ind = time_ind[0]
                x = x_obs[:, time_ind]
                weights = dmeas(x, X_P, time + deltat, eps2, covariate)
                weights_norm = weights / weights.sum()
                
                # Resample particles and parameters
                ind = systematic_resampling(weights_norm, J)
                X = X_P[:, ind]
                I = I_P[:, ind]
                I_vac = I_vac_P[:, ind]
                params_swarm_trans = params_swarm_trans[:, ind]
            else:
                X = X_P
                I = I_P
                I_vac = I_vac_P
        
        # Store results
        params_est_m = np.exp(params_swarm_trans.mean(axis=1))
        params_swarm_iter[:, m] = params_est_m
        
        # Evaluate likelihood with current estimate
        con_likeli_m[m], *_ = pfilter(data, rproc, rinit, dmeas, 
                                                  covariate, params_est_m, 
                                                  tspan, deltat, 2*J, D, 
                                                  D_vac, B, X_0, 
                                                  I_0, I_vac_0, freq_0)
        print(f"Log-likelihood: {con_likeli_m[m]}")
    
    # Find best parameters
    best_idx = np.argmax(con_likeli_m)
    params_est_max = params_swarm_iter[:, best_idx]
    params_est = params_est_m
    
    return params_est, params_swarm_iter, params_est_max, con_likeli_m


def mif2_multi(data_list, rproc, rinit, dmeas, covariate_list, params_guess, rw_sd,
               deltat, J, M, a, D, D_vac, B, X_0, I_0, I_vac_0, freq_0=0.04):
    """
    Multiple Iterated Filtering (MIF2) for multi-region parameter estimation.

    关键差异:
    - 每个地区有独立 t_end (存于 covariate_list[r]['tspan'][1])
      统一用最长 timeline, 并在 time 循环里仅对 time ≤ t_end_r 的
      地区执行过程 / 观测 / 重采样.
    """
    # ---------- 基本检查 ----------
    R = len(data_list)
    assert len(covariate_list) == R, "covariate_list 与 data_list 长度不一致"

    # ---------- 全局时间线 ----------
    # 假设 t0 在所有地区相同
    t0 = covariate_list[0]['tspan'][0]
    t_end_list = [cov['tspan'][1] for cov in covariate_list]
    t_end_max  = max(t_end_list)
    timeline = np.arange(t0, t_end_max + deltat, deltat)

    # ---------- 观测时间提前缓存 ----------
    obs_times_list = [data['time'] for data in data_list]

    # ---------- 参数群初始化 ----------
    params_guess_trans = np.log(params_guess)
    n_par = params_guess_trans.size
    params_swarm_trans = np.tile(params_guess_trans.reshape(-1, 1), (1, J))
    params_swarm_iter = np.zeros((n_par, M))
    con_likeli_m = np.zeros(M)

    # ---------- 主 MIF2 循环 ----------
    for m in range(M):
        print(f"Iteration {m + 1}/{M}")
        rw_sd_m = rw_sd * a ** (2 * m / 50)
        X = np.zeros((R, len(X_0), J))
        I = np.zeros((R, len(I_0), J))
        I_vac = np.zeros((R, len(I_vac_0), J))
        for r in range(R):
            X[r], I[r], I_vac[r] = rinit(X_0, I_0, I_vac_0, J)

        # --------- 时间推进 ---------
        for time in timeline:
            # ① 对参数加随机步
            params_swarm_trans += np.random.randn(n_par, J) * rw_sd_m[:, None]
            params_swarm = np.exp(params_swarm_trans)

            # ② 提取共享参数 (索引同旧版)
            sp0 = 3 * R
            k        = params_swarm[sp0]
            r_shared = params_swarm[sp0 + 1]
            lambda_  = params_swarm[sp0 + 2]
            gam_vac  = params_swarm[sp0 + 3]

            global_weights = np.zeros(J)
            has_obs = False

            # ③ 遍历地区
            for r in range(R):
                # 如果该地区已结束, 跳过
                if time > t_end_list[r]:
                    continue

                # 地区专属参数
                gam  = params_swarm[3 * r]
                eps  = params_swarm[3 * r + 1]
                eps2 = params_swarm[3 * r + 2]
                P    = covariate_list[r]['P']

                # 过程传播
                _, X_P, I_P, I_vac_P = rproc(
                    X[r], I[r], I_vac[r],
                    D, D_vac, B,
                    k, r_shared, 
                    gam, gam_vac, lambda_,
                    eps, P, time,
                    covariate_list[r],
                    deltat, freq_0
                )
                X[r], I[r], I_vac[r] = X_P, I_P, I_vac_P

                # 观测 & 权重 (仅当 next_time 仍在该地区时间段内)
                next_time = time + deltat
                if next_time <= t_end_list[r]:
                    obs_idx = np.where(obs_times_list[r] == next_time)[0]
                    if obs_idx.size:
                        x_obs = data_list[r]['x_obs'][:, obs_idx[0]]
                        w_r = dmeas(x_obs, X_P, next_time, eps2, covariate_list[r])
                        global_weights += w_r
                        has_obs = True

            # ④ 重采样
            if has_obs:
                w_norm = global_weights / global_weights.sum()
                idx = systematic_resampling(w_norm, J)
                params_swarm_trans = params_swarm_trans[:, idx]
                for r in range(R):
                    X[r]     = X[r][:, idx]
                    I[r]     = I[r][:, idx]
                    I_vac[r] = I_vac[r][:, idx]

        # --------- 存储参数估计 ---------
        params_est_m = np.exp(params_swarm_trans.mean(axis=1))
        params_swarm_iter[:, m] = params_est_m

        # --------- 似然评估 (各区独立 tspan) ---------
        total_ll = 0.0
        for r in range(R):
            gam  = params_est_m[3 * r]
            eps  = params_est_m[3 * r + 1]
            eps2 = params_est_m[3 * r + 2]

            k_val        = params_est_m[sp0]
            r_val        = params_est_m[sp0 + 1]
            lambda_val   = params_est_m[sp0 + 2]
            gam_vac_val  = params_est_m[sp0 + 3]
            P_val        = covariate_list[r]['P']

            params_r = np.array([k_val, r_val, gam, lambda_val,
                                 eps, eps2, gam_vac_val])

            # 注意: 使用该地区自己的 tspan
            tspan_r = covariate_list[r]['tspan']
            ll_r, *_ = pfilter(
                data_list[r], rproc, rinit, dmeas, covariate_list[r], params_r,
                tspan_r, deltat, 2 * J,
                D, D_vac, B,
                X_0, I_0, I_vac_0, freq_0
            )
            total_ll += ll_r
            print(f"  {region if 'region' in locals() else f'Region{r}'} log-likelihood: {ll_r:.2f}")

        con_likeli_m[m] = total_ll
        print(f"Total log-likelihood (iter {m + 1}): {total_ll:.2f}\n")

    # --------- 选择最佳参数 ---------
    best_idx = np.argmax(con_likeli_m)
    params_est_max = params_swarm_iter[:, best_idx]
    return params_est_max, params_swarm_iter, params_est_max, con_likeli_m



def fill_zero_rows(x_obs_ave: np.ndarray) -> np.ndarray:
    """
    x_obs_ave : ndarray shape (T, N)
        若某一整行全为 0，则：
        - 同时存在前/后非零行 → 线性插值
        - 只存在前或后非零行 → 直接用该行填充
    返回：
        填补后的新矩阵，shape 同输入
    """
    if not isinstance(x_obs_ave, np.ndarray):
        x_obs_ave = np.asarray(x_obs_ave)
    # 1. 识别“整行都是 0”的行
    zero_row_mask = (x_obs_ave == 0).all(axis=1)

    # 2. 转成 DataFrame，方便用 pandas 的 interpolate/ffill/bfill
    df = pd.DataFrame(x_obs_ave, copy=True)

    # 3. 仅将“整行 0”置为 NaN，其余保持原值
    df[zero_row_mask] = np.nan

    # 4. 先做按行（axis=0）的线性插值，再用前向/后向补齐端点
    df = (df
          .interpolate(method="linear", axis=0, limit_direction="both")
          .fillna(method="ffill")     # 前向填充，解决开头缺失
          .fillna(method="bfill"))    # 后向填充，解决末尾缺失

    return df.values

# Helper function to convert MATLAB datenum to Python datetime
def datenum_to_datetime(datenum):
    """Convert MATLAB datenum to Python datetime"""
    # MATLAB datenum starts from January 0, 0000
    # Python datetime starts from January 1, 0001
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)

def datetime_to_datenum(dt):
    """Convert Python datetime to MATLAB datenum"""
    mdn = dt + timedelta(days=366)
    frac_seconds = (dt - datetime(dt.year, dt.month, dt.day)).total_seconds() / (24 * 3600)
    return mdn.toordinal() + frac_seconds