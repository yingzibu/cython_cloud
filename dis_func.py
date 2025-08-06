"""
Update: 3/18/2025
        Task done:  
            1. Double checked the formula in f2 normal interval calculation
            2. Changed 
Update: 3/17/2025 
        Task done:  
            1. Updated the f2 incase n_t != n_r when bc/vc is activated
            Will turn bc/vc = False if n_t != n_r
            2. Updated the f2 normal and basic interval

Date:   3/11/2025
Author: Yingzi Bu
Aim:    methods for calculating dissolution similarity
Note:   
        Task:  
            # some task...

"""

import warnings
from tqdm import tqdm
# warnings.simplefilter(action='ignore', categrory=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from scipy.stats import norm


def apply_cutoff_rule(time_points:list, ref, test, cutoff=85):
    # Aim: to apply 85% rule. Only reserve one time point if both R and T release 85%
    # return updated time, ref, test 

    # ref_ test_ could be dataframe or list, need to check
    type_list = True
    if type(ref) in [pd.DataFrame, pd.Series]:  
        ref_  = list(ref[time_points].mean(axis=0)); type_list = False
    elif type(ref) == list: ref_ = ref
    
    if type(test) in [pd.DataFrame, pd.Series]: 
        test_ = list(test[time_points].mean(axis=0))
    elif type(test) == list: test_ = test

    idx_list = []
    first_85_appended = False
    times = []; time_dict = {}
    for i in time_points: 
        t = float(i); time_dict[t] = i; times.append(t)
    times = sorted(times)
    new_time_points = [time_dict[t] for t in times]
    assert set(new_time_points) == set(time_points)
    new_time_points = time_points

    for idx, (i, r, t) in enumerate(zip(time_points, ref_, test_)):
        if r > cutoff and t > cutoff: 
            if first_85_appended == False:
                idx_list.append(idx)
                first_85_appended = True
        else: idx_list.append(idx)
    times = list(np.take(time_points, idx_list))
    if type_list == False: # dfs
        ref_df = ref[times]
        test_df = test[times]
        return times, ref_df, test_df, idx_list
    else: # r and t are lists
        ref_ = list(np.take(ref, idx_list))
        test_= list(np.take(test, idx_list))
        return times, ref_, test_, idx_list




def cal_EDNE(ref_df, test_df, alpha_=0.05, delta_square=None, ver=False):
    if delta_square == None: delta_square = 99 * len(ref_df.columns)

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = ref_mean - test_mean

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    # assert m == n

    s_pool = ref_s / m + test_s / n
    u_ = norm.ppf(alpha_)

    t_up = diff.T@ diff - delta_square
    # import streamlit as st
    # st.write('diff', diff.T @ diff)
    # st.write('delta_square: ', delta_square)
    t_down = np.sqrt(4 * diff.T @ s_pool @ diff)
    t_EDNE = t_up / t_down
    if t_EDNE < u_: relation_ = '<'; conc = 'Similar'
    else: relation_ = '>='; conc = 'Dissimilar'

    temp_ = {}
    temp_['EDNE Test'] = [f'T_EDNE = {t_EDNE:.3f}',   u_,conc,
                          alpha_, delta_square]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic',  
                     'critical value', 'Conclusion', 'alpha', 'Delta^2'])
    if ver: print(temp_df)
    return temp_df


def cal_T_square(ref_df, test_df, alpha_=0.05, delta_square=None):
    if delta_square == None: delta_square = 0.74 ** 2

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = ref_mean - test_mean

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    assert m == n
    N = m + n
    P = len(ref_df.columns)


    s = (ref_s + test_s) / 2
    s_inv = np.linalg.inv(s)
    t_square = m * n / N * diff.T @ s_inv @ diff

    # Fcrit = scipy.stats.f.ppf(dfn=P, q=alpha_, dfd=N-P-1) previous
    # yet here we have noncentrality parameter, should use ncf function and param nc
    nc = delta_square * m * n / N
    Fcrit = scipy.stats.ncf.ppf(q=alpha_, dfn=P, dfd=N-P-1, nc=nc, loc=0, scale=1)
    adjusted_t_square = (N - 1 - P) / (N - 2) / P * t_square
    # C_ = (N-2) * P / (N - 1 - P) * Fcrit
    if adjusted_t_square < Fcrit: relation_ = '<'; conc = 'similar'
    else: relation_ = '>='; conc = 'dissimilar'

    temp_ = {}
    temp_['T^2-test'] = [f'| param*T2 = {adjusted_t_square:.3f}',
                         relation_,
                         f'[F_{alpha_}] = {Fcrit:.3f}', conc]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    print(temp_df)
    # print(f'test statistic    critical value    conclusion')
    # print(f'(N-p-1)/[p(N-2)] T^2 =  {adjusted_t_square:.3f}     {relation_}[F_{alpha_}] = {Fcrit:.3f}   {conc}')
    return adjusted_t_square


def cal_SE(ref_df, test_df, alpha_=0.05, delta_square=None):
    if delta_square == None: delta_square = 0.74 ** 2

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = (ref_mean - test_mean)

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    assert m == n
    N = m + n
    P = len(ref_df.columns)

    s_list = [ref_s.iloc[i,i] + test_s.iloc[i,i] for i in range(P)]
    hat_delta = diff.div(pd.Series(s_list, index=ref_df.columns), axis=0)
    hat_epsilon = np.multiply(hat_delta, hat_delta)

    t_up = 2 * hat_delta.T @ diff - delta_square
    s_pool = (ref_s/m + test_s/n)

    t_down = 16 * hat_delta.T @ s_pool @ hat_delta
    hat_pi_1 = 2 * np.multiply(ref_s, ref_s)
    hat_pi_2 = 2 * np.multiply(test_s, test_s)
    t_down += 4 * hat_epsilon.T @ (hat_pi_1 / m + hat_pi_2 / n) @ hat_epsilon
    t_se = t_up / np.sqrt(t_down)

    u_ = norm.ppf(alpha_)
    if t_se < u_: relation_ = '<'; conc = 'similar'
    else: relation_ = '>='; conc = 'dissimilar'

    temp_ = {}
    temp_['SE Test'] = [f'| T_SE = {t_se:.3f}',
                         relation_,
                         f'[u_{alpha_}] = {u_:.3f}', conc]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    print(temp_df)
    return t_se

def cal_GMD(ref_df, test_df, alpha_=0.05, delta_square=None):
    if delta_square == None: delta_square = 0.74 ** 2

    ref_mean = ref_df.mean(axis=0)
    test_mean = test_df.mean(axis=0)
    diff = (ref_mean - test_mean)

    ref_s = ref_df.cov()
    test_s = test_df.cov()

    m = len(ref_df); n = len(test_df)
    assert m == n
    N = m + n
    P = len(ref_df.columns)
    s_inv = np.linalg.inv(ref_s + test_s)
    hat_gamma = s_inv @ diff

    t_up = np.sqrt(N) * (2 * diff.T @ hat_gamma - delta_square)

    hat_v = 16 * hat_gamma @ (N/m * ref_s + N/n * test_s) @ hat_gamma
    total_sum = 0
    for i in range(P):
        for j in range(P):
            for k in range(P):
                for l in range(P):
                    total_sum += 4 * N / m * (
    ref_s.iloc[i,k] * ref_s.iloc[j,l] + ref_s.iloc[i,l] * ref_s.iloc[j,k]
                ) * hat_gamma[i] * hat_gamma[j] * hat_gamma[k] * hat_gamma[l]

                    total_sum += 4 * N / n * (
    test_s.iloc[i,k] * test_s.iloc[j,l] + test_s.iloc[i,l] * test_s.iloc[j,k]
            ) * hat_gamma[i] * hat_gamma[j] * hat_gamma[k] * hat_gamma[l]
    hat_v += total_sum
    t_gmd = t_up / np.sqrt(hat_v)
    t_gmd
    u_ = norm.ppf(alpha_)
    if t_gmd < u_: relation_ = '<'; conc = 'similar'
    else: relation_ = '>='; conc = 'dissimilar'

    temp_ = {}
    temp_['GMD Test'] = [f'| T_GMD = {t_gmd:.3f}',
                        relation_,
                        f'[u_{alpha_}] = {u_:.3f}', conc]
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
            columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    print(temp_df)
    return t_gmd

# careful since the position of ref and test should not be interchangeable like f2
def cal_f1(ref_df, test_df, ver=False): 
    
    # input is df
    if type(ref_df) in [pd.DataFrame, pd.Series]: 
        ref_mean = list(ref_df.mean(axis=0))
    # input is a list of mean
    elif type(ref_df) == list:       
        ref_mean = ref_df; bc, vc=False, False
    else: 
        if ver:
            print('unrecognized for ref data, type={type(ref_df)}'); return None

    if type(test_df) in [pd.DataFrame, pd.Series]:
        test_mean = list(test_df.mean(axis=0))
    elif type(test_df) == list:       test_mean = test_df; bc, vc=False, False
    else: 
        if ver: 
            print('unrecognized for ref data, type={type(test_df)}'); return None

    # P = len(ref_mean)
    try: 
        assert len(ref_mean) == len(test_mean)
    except:
        return ref_mean, test_mean
    sum_diff, sum_R = 0, 0
    for i, j in zip(ref_mean, test_mean):
        sum_diff += np.abs(i-j)
        sum_R += i
    f1 = sum_diff / sum_R * 100
    if ver: print(f'F1 value R & T: {f1:.3f}')
    # if ver: return f2, sum_variance, sum_diff_square
    return f1


def cal_f2(ref_df, test_df, bc=False, vc=False, ver=False, return_all=False): # implement bias corrected
    # input is df, could do bc vc
    if type(ref_df) == pd.DataFrame: ref_mean = list(ref_df.mean(axis=0))
    # input is a list of mean, cannot do vc, bias corrected
    elif type(ref_df) == list:       ref_mean = ref_df; bc, vc=False, False
    else: 
        if ver:
            print('unrecognized for ref data, type={type(ref_df)}'); return None

    if type(test_df) == pd.DataFrame: test_mean = list(test_df.mean(axis=0))
    elif type(test_df) == list:       test_mean = test_df; bc, vc=False, False
    else: 
        if ver: 
            print('unrecognized for ref data, type={type(test_df)}'); return None

    P = len(ref_mean) # number of time points
    assert len(ref_mean) == len(test_mean)
    sum_diff_square = 0
    for i, j in zip(ref_mean, test_mean):
        sum_diff_square += (i-j) ** 2

    sum_variance = 0
    if vc or bc: # will apply var or bias corrected only inf ref and test are dataframes
        try: 
            assert len(ref_df) == len(test_df)
            assert type(ref_df) == type(test_df) and type(ref_df) == pd.DataFrame
        except: 
            if ver: print("type of ref df: ",type(ref_df), 
                  'len ref:', len(ref_df), 'len test: ', len(test_df))
            # return None
            vc = False; bc = False

        ref_S = [i**2 for i in np.std(ref_df, ddof=1).tolist()]
        test_S= [i**2 for i in np.std(test_df, ddof=1).tolist()]
        if bc and vc == False:
            sum_variance =  np.sum(ref_S) + np.sum(test_S)
        if vc: # apply variance-corrected f2
            bc = True
            sum_variance = 0
            for rs, ts in zip(ref_S, test_S):
                sum_s = rs + ts
                w_t = 0.5 + ts / sum_s
                w_r = 0.5 + rs / sum_s
                sum_variance += w_t * ts + w_r * rs
        n = len(ref_df)   # number of units
        sum_variance /= n

    if sum_variance > sum_diff_square: # definitely applied bc or vc
        if vc: param_name = 'vc'; vc = False
        else: param_name = 'bc' ; bc = False

        if ver: print(f'var    >   sum(|t-r|^2), cannot apply {param_name}')
        if ver: print(f'{sum_variance:.3f} > {sum_diff_square:.3f}')
        return None
    # else: # 2 conditions, sum_variance=0, sum_variance \in (0, sum_diff_square)

        # reset bc = False, vc = Fal

    D = sum_diff_square - sum_variance

    f2 = 100 - 25 * np.log10(1+D/P)
    if ver: print(f'F2 value R & T: {f2:.3f} | bc: {bc} | vc: {vc}')
    # if ver: return f2, sum_variance, sum_diff_square
    if return_all:
        return f2, bc, vc
    else:
        return f2



# import streamlit as st
def cal_MSD(ref_df, test_df, tolerance_list=[10,11,13,15], 
            ver=False, ver_stat=False):
    
    # st.write(list(test_df.columns))
    try:
        assert list(test_df.columns) == list(ref_df.columns)
    except:
        if ver: 
            print(
            f'time diff| T: {list(test_df.columns)}| R: {list(ref_df.columns)}')
        return None
    time_points = list(test_df.columns)

    P = len(time_points)
    n_r = len(ref_df)
    n_t = len(test_df)

    # try: assert n == len(test_df)
    # except:
    #     if ver:
    #         print(f'ref units {n} are different from test units {len(test_df)}')
    #         print('Check data before cal MSD')
    #     return None

    S1 = ref_df.cov()  # time_num X time_num matrix
    S2 = test_df.cov() # time_num X time_num matrix 
    # Since S1 and S2 are the same dim, even different units, 
    # may still be able to calculate
    # import streamlit as st
    # st.write(S1)
    # st.write(S2)

    # S_pooled = (S1 + S2) / 2 
    S_pooled = (n_r - 1) * S1 + (n_t - 1) * S2
    S_pooled /= (n_r + n_t - 2)  
 
    # st.write(S_pooled)
    ref_mean = list(ref_df.mean(axis=0))
    test_mean = list(test_df.mean(axis=0))
    x2_x1 = [i-j for i, j in zip(test_mean, ref_mean)]
    a = np.array(x2_x1).reshape(len(time_points), 1)

    # if n_r == n_t: 
    # n = n_r
    # K = n**2/(2*n)* (2*n - P - 1) / ((2*n - 2) * P)
    # Fcrit = scipy.stats.f.ppf(q=1-.1, dfn=P, dfd=2*n-P-1)

    K = n_r * n_t / (n_r + n_t) * (n_r + n_t - P - 1) / ((n_r + n_t - 2) * P)
    Fcrit= scipy.stats.f.ppf(q=1-.1, dfn=P, dfd=n_r+n_t-P-1)

    spinv = np.linalg.inv(S_pooled.loc[time_points, time_points])
    D_M = np.sqrt(a.T @ spinv @ a)[0][0]
    # print('Mahalanobis distance (T & R):', D_M)

    bound1 = a @ (1 + np.sqrt(Fcrit/(K * a.T @ spinv @ a)))
    bound2 = a @ (1 - np.sqrt(Fcrit/(K * a.T @ spinv @ a)))
    # 90% CI of Mahalanobis distance:
    DM_1 = np.sqrt(bound1.T @ spinv @ bound1)[0][0]
    DM_2 = np.sqrt(bound2.T @ spinv @ bound2)[0][0]
    DM_upper = max(DM_1, DM_2)
    DM_lower = min(DM_1, DM_2)
    # if DM_lower < 0: DM_lower = 0

    # print('lower bound of DM:', DM_lower)
    # print('upper bound of DM:', DM_upper)
    

    # print('DM_upper | tolerance limit | conclusion')
    temp_ = {}
    # temp_['T^2-test'] = [f'| param*T2 = {adjusted_t_square:.3f}',
    #                      relation_,
    #                      f'[F_{alpha_}] = {Fcrit:.3f}', conc]
    # temp_df = pd.DataFrame.from_dict(temp_, orient='index',
    #         columns=['Test statistic', 'vs', 'critical value', 'conclusion'])
    # print(temp_df)


    for tolerance in tolerance_list:

        D_g = np.array([tolerance] * len(time_points)).reshape(
                len(time_points), 1)
        RD = np.sqrt(D_g.T @ spinv @ D_g)[0][0]

        if DM_upper <= RD: conc = 'Similar'
        else: conc = 'Dissimilar'
        temp_[f'{tolerance}% limit'] = [D_M, DM_lower, DM_upper, 
                                        RD, tolerance, conc]
            
        #     print(f'{DM_upper:.3f} \t <=  {RD:.3f}[{tolerance}%]     Similar')
        # else:
        #     print(f'{DM_upper:.3f} \t >   {RD:.3f} [{tolerance}%]    Dissimilar')
    temp_df = pd.DataFrame.from_dict(temp_, orient='index',
                                    columns = ['Dm', 'Dm_lower', 'Dm_upper',
                                        'Dm_max', 'limit (%)', 'Conclusion'])
    if ver: print(temp_df)
    if ver_stat: 
        stat_dict = {}
        cols_temp = ['P', 'K', 'F', 'MSD', 'Lower', 'Upper']
        stat_dict['stat'] = [P, K, Fcrit, D_M, DM_lower, DM_upper]
        stat_df = pd.DataFrame.from_dict(stat_dict, 
                                         orient='index', columns=cols_temp)
        if ver: print(stat_df.T)


    return temp_df
 
def jackknife_statistic(ref_df, test_df, type_jk='nt=nr', 
                        bc=False, vc=False, ver=False):
    nt = len(test_df)
    nr = len(ref_df)
    jk_list = []
    if type_jk == 'nt=nr':
        try: assert nt == nr
        except: return [] # Empty jk_list means nt != nr
        for i in range(nt):
            t = test_df.drop(i)
            r = ref_df.drop(i)
            f2 = cal_f2(t, r, bc=bc, vc=vc, ver=ver)
            jk_list.append(f2)
    elif type_jk == 'nt+nr':
        for i in range(nt):
            t = test_df.drop(i)
            f2 = cal_f2(t, ref_df, bc=bc, vc=vc, ver=ver)
            jk_list.append(f2)

        for i in range(nr):
            r = ref_df.drop(i)
            f2 = cal_f2(test_df, r, bc=bc, vc=vc, ver=ver)
            jk_list.append(f2)
    elif type_jk == 'nt*nr':
        for i in range(nt):
            t = test_df.drop(i)
            for j in range(nr):
                r = ref_df.drop(j)
                f2 = cal_f2(t, r, bc=bc, vc=vc, ver=ver)
                jk_list.append(f2)

    else:
        if ver: print("""type_jk should be one of['nt+nr', 'nt*nr', 'nt=nr']""")
        return
    return jk_list


def bootstrap_f2_list(test_df, ref_df, B=10000, bc=False, vc=False, 
                      ver=False, one_row=False):
    m = len(ref_df); n = len(test_df)
    n = min(m, n)
    f2_orig = cal_f2(test_df, ref_df, bc=bc, vc=vc, ver=ver)
    f2_estimates = []
    for i in tqdm(range(B), total=B, desc=f'bootstrap {B} samples'):
        r = ref_df.sample(n=n, replace=True) # resample with replacement
        t = test_df.sample(n=n, replace=True)
        if one_row: 
            r = r.sample(n=1, replace=True)
            t = t.sample(n=1, replace=True)
        f2 = cal_f2(t,r, bc=bc, vc=vc, ver=ver)
        
        if f2 == None: 
            return None, f2_orig
        f2_estimates.append(f2)
    f2_estimates.sort()
    assert len(f2_estimates) == B
    return f2_estimates, f2_orig

def BCa_jk(jk_list, f2_estimates, f2_orig, alpha_=0.05):
    if len(jk_list) == 0: # means n_t != n_r
        return None, None, None
    m = np.mean(jk_list) # mean of the jackknife statistics
    u, d = 0, 0
    for i in jk_list:
        diff = m - i
        u += diff**3
        d += diff**2
    a_hat = u / (d**1.5)
    a_hat /= 6
    f2_num = sum(i < f2_orig for i in f2_estimates)
    z0_hat = norm.ppf(f2_num/len(f2_estimates))
    z_alpha = norm.ppf(alpha_)
    z_1_alpha = norm.ppf(1-alpha_)

    def cal_alpha(z_, z0_hat=z0_hat, a_hat=a_hat):
        temp = z0_hat + z_
        temp1 = temp / (1-a_hat*temp)
        temp1 += z0_hat
        return norm.cdf(temp1)

    alpha_1 = cal_alpha(z_alpha)
    alpha_2 = cal_alpha(z_1_alpha)
    f2_L = np.percentile(np.array(f2_estimates), 100*alpha_1)
    f2_U = np.percentile(np.array(f2_estimates), 100*alpha_2)
    return m, f2_L, f2_U

def cal_bootf2(ref_df, test_df, B=1000, bc=False, vc=False, alpha_=0.05,
               type_jk='nt=nr', ver=False, one_row=False):
    result_here = {}
    f2_estimates, f2_orig = bootstrap_f2_list(ref_df, test_df, B=B, bc=bc, 
                                              vc=vc, ver=ver, one_row=one_row)
    if f2_estimates == None: return None
    bootstrap_mean = np.mean(f2_estimates)
    # def percent_interval(B=B, alpha_=alpha_, f2_estiates=f2_estimates,
    #                    f2_orig=f2_orig): 
    #     k = (B+1) * alpha_; k = int(k) # python starts from 0th 

    #     frac_part = norm.ppf(alpha_) - norm.ppf(k/(B+1))
    #     frac_part /=(norm.ppf((k+1)/(B+1)) - norm.ppf(k/(B+1)))
    #     k = k - 1 # python starts from 0th
    #     f2_here = f2_estimates[k] + \
    #                 frac_part * (f2_estiates[k+1] - f2_estiates[k])
    #     # f2_here = 2 * f2_orig - f2_here
    #     return f2_here
    
    # import streamlit as st
    # st.write('Ba', B * alpha_, ' | (B+1)a', (B+1) * alpha_)
    # st.write('at 500', f2_estimates[500],'| at 499:', f2_estimates[499])
    # st.write('B(1-a):', B * (1-alpha_), '| (B+1)(1-a): ',(B+1) * (1-alpha_))
    # st.write('at 9500', f2_estimates[9500],'| at 9499:', f2_estimates[9499])


    # st.write(f2_estimates[int(B * alpha_)]) # the same as f2_L_percent
    # st.write(f2_estimates[int(B * (1- alpha_))]) # the same as f2_U_percent
    # what if B * alpha is not an integer? 
    # f2_L = percent_interval(B=B, alpha_=alpha_)
    # f2_U = percent_interval(B=B, alpha_=1-alpha_)
    # result_here[f'f2, percentile interval_1'] = [f2_orig, bootstrap_mean,
    #                               f2_L, f2_U]
    f2_L_percent = np.percentile(np.array(f2_estimates), 100*alpha_/2)
    f2_U_percent = np.percentile(np.array(f2_estimates), 100*(1-alpha_/2))
    # print(m, f2_L, f2_U)

    result_here[f'f2, percentile interval'] = [f2_orig, bootstrap_mean,
                                  f2_L_percent, f2_U_percent]

    
    # normal interval
    E_B = bootstrap_mean - f2_orig
    V_B = 0
    for f in f2_estimates: V_B += (f - bootstrap_mean) ** 2
    V_B /= (B - 1)
    Z_ = norm.ppf(1-alpha_); temp_ = np.sqrt(V_B) * Z_
    f2_1 = f2_orig - E_B - temp_; f2_2 = f2_orig - E_B + temp_
    # import streamlit as st
    # st.write(f2_orig, E_B, temp_)
    f2_L = min(f2_1, f2_2); f2_U = max(f2_1, f2_2)
    result_here['f2, normal interval'] = [f2_orig, bootstrap_mean, 
                                              f2_L, f2_U]
    
    # basic interval 
    # select the (B+1) * alpha th f2 
    # since python starts from 0th, will select the (B+1) * alpha - 1 th f2
    # def get_kth_f2(B=B, alpha_=alpha_, f2_estimates=f2_estimates): 

    def basic_interval(B=B, alpha_=alpha_, f2_estimates=f2_estimates,
                       f2_orig=f2_orig): 
        k = (B+1) * alpha_; k = int(k) # python starts from 0th 

        frac_part = norm.ppf(alpha_) - norm.ppf(k/(B+1))
        frac_part /=(norm.ppf((k+1)/(B+1)) - norm.ppf(k/(B+1)))
        k = k - 1 # python starts from 0th
        f2_here = f2_estimates[k] + \
                    frac_part * (f2_estimates[k+1] - f2_estimates[k])
        f2_here = 2 * f2_orig - f2_here
        return f2_here
    
    f2_1 = basic_interval(alpha_=alpha_); f2_2 = basic_interval(alpha_=1-alpha_)
    f2_L = min(f2_1, f2_2); f2_U = max(f2_1, f2_2)
    result_here['f2, basic interval'] = [f2_orig, bootstrap_mean,
                                         f2_L, f2_U]
        
    
    if type_jk == 'all': type_jks = ['nt=nr', 'nt+nr', 'nt*nr']
    else: type_jks = [type_jk]

    try: 
        for type_jk in type_jks:
            jk_list = jackknife_statistic(ref_df, test_df, type_jk=type_jk, 
                                          bc=bc, vc=vc, ver=ver)
            m, f2_L, f2_U = BCa_jk(jk_list, f2_estimates, f2_orig, alpha_=alpha_)
            result_here[f'jackknife BCa {type_jk}'] =\
                     [m, bootstrap_mean, f2_L, f2_U]
    except:
        if ver:
            print(f'cannot calculate BCa_jk, f2_list uses one_row={one_row}')
        
    cols_here = ['sample mean', f'{B} bootstraps mean', 'CI_L', 'CI_U']
    result_df = pd.DataFrame.from_dict(result_here, orient='index', 
                                       columns=cols_here)
    if ver: print('\n'); print(result_df)
    return result_df




def output_for_R(data_here, file_name, folder_name=None, 
                 batch_name=['ref, test'], ver=False):
    data_sheet_1 = pd.DataFrame()
    data_sheet_1['Time'] = data_here.time_points
    time_points_all = data_here.time_points
    units = data_here.ref_df.shape[0]
    for i in range(units):
        data_sheet_1[f'Unit{i+1:02d}'] = list(data_here.ref_df.iloc[i])
    
    data_sheet_2 = pd.DataFrame()
    data_sheet_2['Time'] = data_here.time_points
    units = data_here.test_df.shape[0]
    for i in range(units):
        data_sheet_2[f'Unit{i+1:02d}'] = list(data_here.test_df.iloc[i])
    
    
    data_1 = data_here.ref_df.copy(); len_r = len(data_here.ref_df)
    # data_1['batch'] = pd.DataFrame([batch_name[0] for i in range(len(r))])
    data_1['tablet'] = pd.DataFrame([i for i in range(1, len_r+1)])
    data_1['type'] = pd.DataFrame(['R' for i in range(len_r)])
    
    data_2 = data_here.test_df.copy();len_t = len(data_here.test_df)
    # data_2['batch'] = pd.DataFrame([batch_name[1] for i in range(len(t))])
    data_2['tablet'] = pd.DataFrame([i for i in range(1, len_t+1)])
    data_2['type'] = pd.DataFrame(['T' for i in range(len_t)])
    
    data_df = pd.concat([data_1, data_2]).reset_index(drop=True)
    
    data_sheet_3 = pd.DataFrame()
    # data_sheet_3['batch'] = data_df['batch']
    data_sheet_3['type'] = data_df['type']
    data_sheet_3['tablet'] = data_df['tablet']
    data_sheet_3[time_points_all] = data_df[time_points_all]
    
    if folder_name == None: 
        final_file_name = file_name
    else:
        final_file_name = folder_name + '/' + file_name

    with pd.ExcelWriter(f'{final_file_name}.xlsx') as writer:
        data_sheet_1.to_excel(writer, sheet_name=batch_name[0], index=False)
        data_sheet_2.to_excel(writer, sheet_name=batch_name[1], index=False)
        data_sheet_3.to_excel(writer, sheet_name='data1', index=False)
    if ver: print('Write excel file:', f'{final_file_name}.xlsx')
    return data_sheet_1, data_sheet_2, data_sheet_3


def plot_data(time_points, ref_mean, test_mean, 
              xlabel='time in minutes', ylabel='% dissolved',
                rlabel='ref batch', tlabel='test batch',
                title='Mean distribution of the test and reference'):
    fig = plt.figure()
    # x_axis = time_points
    # def series2list(df_series):
    #     if type(df_series) == pd.Series: return df_series.tolist()
    #     if type(df_series) == list: return df_series
    # time_points = series2list(time_points)
    
    x_axis = [float(i) for i in time_points]
    test_axis = test_mean
    ref_axis = ref_mean
    plt.plot(x_axis, test_axis, 'x-', label=tlabel)
    plt.plot(x_axis, ref_axis, '.-', label=rlabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()
    return fig

def f2_from_diff(diff_in_percent): 
    f2_limit = (1 + diff_in_percent**2)**(-0.5)*100
    f2_limit = 50 * np.log10(f2_limit)
    return f2_limit


def f2_from_diff1(diff): 
    return 100 - 25 * np.log10( 1 + diff**2)

def f2_to_diff(f2_limit): 
    temp = (100 - f2_limit) / 25
    temp = 10 ** temp - 1
    return np.sqrt(temp)
