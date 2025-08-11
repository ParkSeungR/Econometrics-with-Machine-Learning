import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from linearmodels import IV2SLS, IV3SLS, IVSystemGMM, SUR

def parse_auto_equation_system(equations: dict):
    parsed = {}
    all_deps = list(equations.keys())
    rhs_vars = {}
    all_system_exog_candidates = set() 

    for dep, eq in equations.items():
        rhs = eq.split('~')[1].strip()
        rhs_list = [v.strip() for v in rhs.split('+')]
        rhs_vars[dep] = rhs_list

    for dep in equations:
        current_rhs = rhs_vars[dep]
        exog = [v for v in current_rhs if v not in all_deps] 
        endog = [v for v in current_rhs if v in all_deps and v != dep] 
        candidate_ivs = []
        for other_dep in equations:
            if other_dep == dep:
                continue
            for var in rhs_vars[other_dep]:
                if var not in current_rhs and var not in all_deps:
                    candidate_ivs.append(var)
        instr = sorted(list(set(candidate_ivs)))

        parsed[dep] = {'exog': exog, 'endog': endog, 'instr': instr}
        all_system_exog_candidates.update(exog)
        all_system_exog_candidates.update(instr)

    all_system_exog_variables = sorted(list(all_system_exog_candidates))
    return parsed, all_system_exog_variables


def Run_Estimate_auto_IV_system(data: pd.DataFrame, equations: dict):
    print("--- 1. 데이터 준비 ---")
    parsed_equations, all_system_exog_variables = parse_auto_equation_system(equations)
    used_vars = set()
    for dep, vdict in parsed_equations.items():
        used_vars.add(dep)
        used_vars.update(vdict['exog'])
        used_vars.update(vdict['endog'])
        used_vars.update(vdict['instr'])
    data = data[list(used_vars)].dropna()

    print(f"사용된 변수: {sorted(used_vars)}")
    print(f"전체 시스템의 외생변수 (개념적 IV 풀): {all_system_exog_variables}") 
    print("\n기술통계량:")
    display(data.describe())

    print("\n상관계수 행렬:")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    print("\n히스토그램 및 KDE:")
    n_cols = 3
    n_rows = int(np.ceil(len(data.columns) / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for i, col in enumerate(data.columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(data[col], kde=True)
        plt.title(f'Histogram and KDE of {col}')
    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("--- 2. SUR (OLS) 추정 ---")
    try:
        sur_eqs = {dep: f"{dep} ~ {' + '.join(info['exog'] + info['endog'])}" for dep, info in parsed_equations.items()}
        mod_sur = SUR.from_formula(sur_eqs, data)
        res_sur = mod_sur.fit(cov_type="unadjusted")
        print(res_sur)
    except Exception as e:
        print(f"SUR 오류: {e}")

    print("\n--- 3. 2SLS 추정 (개별 방정식 - from_formula) ---")
    
    for dep, info in parsed_equations.items():
        if not info['endog'] or not info['instr']:
            print(f"{dep}: 내생변수 또는 도구변수가 없어 2SLS (from_formula) 생략")
            continue
        try:
            rhs_parts = info['exog'][:]
            if info['endog'] and info['instr']:
                rhs_parts.append(f"[{' + '.join(info['endog'])} ~ {' + '.join(info['instr'])}]")
            
            # 상수항 1 + 제거, from_formula 기본 동작에 맡김
            formula_str = f"{dep} ~ {' + '.join(rhs_parts)}"
            
            mod = IV2SLS.from_formula(formula_str, data)
            res = mod.fit(cov_type="unadjusted") # cov_type 통일
            print(f"\n#### 2SLS 결과 ({dep} - from_formula) ####")
            print(res)
        except Exception as e:
            print(f"{dep}: 2SLS (from_formula) 오류: {e}")


    print("\n--- 4. 3SLS 추정 ---")
    try:
        eqs_3sls = {}
        for dep, info in parsed_equations.items():
            rhs_parts = info['exog'][:]
            if info['endog'] and info['instr']:
                rhs_parts.append(f"[{' + '.join(info['endog'])} ~ {' + '.join(info['instr'])}]")
            
            # 상수항 1 + 제거, from_formula 기본 동작에 맡김
            eqs_3sls[dep] = f"{dep} ~ {' + '.join(rhs_parts)}"

        print("\n3SLS 및 GMM에 사용될 공식:")
        for k, v in eqs_3sls.items():
            print(f"{k}: {v}")

        mod_3sls = IV3SLS.from_formula(eqs_3sls, data)
        res_3sls = mod_3sls.fit(cov_type="unadjusted") # cov_type 통일
        print(res_3sls)
    except Exception as e:
        print(f"3SLS 오류: {e}")

    print("\n--- 5. GMM 추정 ---")
    try:
        mod_gmm = IVSystemGMM.from_formula(eqs_3sls, data, weight_type="unadjusted") # weight_type도 통일
        res_gmm = mod_gmm.fit(cov_type="unadjusted", iter_limit=100) # cov_type 통일
        print(f"GMM iterations: {res_gmm.iterations}")
        print(res_gmm)
    except Exception as e:
        print(f"GMM 오류: {e}")

    print("\n" + "="*80)
    print("--- 자동 도구변수 기반 연립방정식 추정 완료 ---")
