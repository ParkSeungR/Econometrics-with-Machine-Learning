import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from linearmodels import IV2SLS, IV3SLS, IVSystemGMM, SUR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def parse_auto_equation_system(equations: dict):
    """
    방정식 시스템을 파싱하여 외생변수, 내생변수, 도구변수 목록을 자동으로 생성합니다.
    """
    parsed = {}
    all_deps = list(equations.keys())
    rhs_vars = {}
    all_system_exog_candidates = set() 

    # 1. 모든 방정식의 우변 변수 목록 추출
    for dep, eq in equations.items():
        rhs = eq.split('~')[1].strip()
        rhs_list = [v.strip() for v in rhs.split('+')]
        rhs_vars[dep] = rhs_list

    # 2. 각 방정식별 exog, endog, instr (후보) 구분
    for dep in equations:
        current_rhs = rhs_vars[dep]
        # 외생변수(exog): 우변에 있고, 다른 종속변수가 아닌 변수
        exog = [v for v in current_rhs if v not in all_deps] 
        # 내생변수(endog): 우변에 있고, 다른 방정식의 종속변수인 변수
        endog = [v for v in current_rhs if v in all_deps and v != dep] 
        
        # 도구변수 후보(candidate_ivs): 전체 시스템의 외생변수 중 해당 방정식의 exog/endog에 포함되지 않은 변수
        candidate_ivs = []
        for other_dep in equations:
            if other_dep == dep:
                continue
            for var in rhs_vars[other_dep]:
                if var not in current_rhs and var not in all_deps:
                    candidate_ivs.append(var)
        # 도구변수(instr): 전체 시스템에서 해당 방정식에 사용되지 않은 외생변수들 (후보 중복 제거 및 정렬)
        instr = sorted(list(set(candidate_ivs)))

        parsed[dep] = {'exog': exog, 'endog': endog, 'instr': instr}
        
        # 전체 시스템의 외생변수 후보 풀(pool) 업데이트
        all_system_exog_candidates.update(exog)
        all_system_exog_candidates.update(instr)

    all_system_exog_variables = sorted(list(all_system_exog_candidates))
    return parsed, all_system_exog_variables


def Run_Estimate_auto_IV_system(data: pd.DataFrame, equations: dict):
    """
    자동으로 파싱된 방정식을 사용하여 시스템 타입을 판별하고 적절한 추정을 수행합니다.
    - 연립방정식 모형 (내생변수 O): OLS → 2SLS → 3SLS → GMM
    - SUR 모형 (내생변수 X): OLS → SUR
    
    Parameters:
    -----------
    data : pandas.DataFrame
        분석할 데이터
    equations : dict
        {종속변수: "종속변수 ~ 독립변수1 + 독립변수2 + [다른 종속변수 (내생변수)]"} 형태의 딕셔너리
    """
    print("="*80)
    print("        자동 도구변수(IV)를 활용한 연립 방정식 시스템 추정 분석")
    print("="*80)
    
    print("\n--- 1. 데이터 전처리 및 탐색적 분석 ---")
    
    # 1. 데이터 파싱 및 정제
    parsed_equations, all_system_exog_variables = parse_auto_equation_system(equations)
    used_vars = set()
    for dep, vdict in parsed_equations.items():
        used_vars.add(dep)
        used_vars.update(vdict['exog'])
        used_vars.update(vdict['endog'])
        used_vars.update(vdict['instr'])
    data = data[list(used_vars)].dropna()

    print(f"사용 변수: {sorted(used_vars)}")
    print(f"전체 시스템 외생변수 (총 IV 풀): {all_system_exog_variables}") 
    print(f"결측치 제거 후 관측치 수: {len(data)}")
    
    # 시스템 타입 판별
    has_endogeneity = any(info['endog'] for info in parsed_equations.values())
    
    print("\n" + "="*80)
    if has_endogeneity:
        print("📌 시스템 타입: 연립방정식 모형 (Simultaneous Equation System)")
        print("   - 내생변수가 존재하여 IV 추정 방법이 필요합니다.")
        print("   - 추정 순서: OLS (비교 기준) → 2SLS → 3SLS → GMM")
    else:
        print("📌 시스템 타입: SUR 모형 (Seemingly Unrelated Regression)")
        print("   - 내생변수가 없고 오차항 간 상관관계만 존재합니다.")
        print("   - 추정 순서: OLS (개별 방정식) → SUR (효율성 개선)")
    print("="*80)
    
    # 각 방정식의 구조 출력
    print("\n방정식 구조:")
    for dep, info in parsed_equations.items():
        print(f"  {dep}:")
        print(f"    - 외생변수: {info['exog']}")
        if info['endog']:
            print(f"    - 내생변수: {info['endog']}")
        if info['instr']:
            print(f"    - 도구변수: {info['instr']}")
    
    # 2. 기초 통계량 및 시각화
    print("\n기초 통계량:")
    display(data.describe())

    print("\n상관계수 행렬:")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("상관계수 행렬")
    plt.show()

    print("\n히스토그램 및 KDE:")
    n_cols = 3
    n_rows = int(np.ceil(len(data.columns) / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for i, col in enumerate(data.columns):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(data[col], kde=True)
        plt.title(f'{col}의 분포 (Histogram and KDE)')
    plt.tight_layout()
    plt.show()

    # ========================================================================
    # 연립방정식 모형인 경우
    # ========================================================================
    if has_endogeneity:
        print("\n" + "="*80)
        print("--- 2. OLS 추정 (비교 기준선 - Baseline) ---")
        print("⚠️  주의: 내생변수를 고려하지 않은 OLS는 편향된 추정량을 제공합니다.")
        print("    IV 방법론(2SLS, 3SLS, GMM)과 비교하기 위한 기준선입니다.")
        print("="*80)
        
        for dep, info in parsed_equations.items():
            try:
                # 독립변수 리스트 (외생변수 + 내생변수)
                all_rhs = info['exog'] + info['endog']
                X = add_constant(data[all_rhs])
                y = data[dep]
                
                model_ols = OLS(y, X)
                result_ols = model_ols.fit()
                
                print(f"\n#### OLS 결과 ({dep}) ####")
                print(result_ols.summary())
            except Exception as e:
                print(f"{dep}: OLS 추정 실패: {e}")

        print("\n" + "="*80)
        print("--- 3. 2SLS 추정 (개별 방정식 - Two-Stage Least Squares) ---")
        print("="*80)
        
        for dep, info in parsed_equations.items():
            if not info['endog']:
                print(f"\n#### 2SLS 생략 ({dep}) ####")
                print(f"{dep}: 내생변수가 없으므로 2SLS 추정 생략 (OLS와 동일)")
                continue
                
            if not info['instr']:
                print(f"\n#### 2SLS 경고 ({dep}) ####")
                print(f"{dep}: 내생변수({', '.join(info['endog'])})는 있으나 도구변수가 없어 2SLS 추정 불가능")
                continue
                
            try:
                rhs_parts = info['exog'][:]
                rhs_parts.append(f"[{' + '.join(info['endog'])} ~ {' + '.join(info['instr'])}]")
                formula_str = f"{dep} ~ {' + '.join(rhs_parts)}"
                
                mod = IV2SLS.from_formula(formula_str, data)
                res = mod.fit(cov_type="unadjusted") 
                print(f"\n#### 2SLS 결과 ({dep}) ####")
                print(res)
            except Exception as e:
                print(f"{dep}: 2SLS 추정 실패: {e}")

        print("\n" + "="*80)
        print("--- 4. 3SLS 추정 (시스템 추정 - Three-Stage Least Squares) ---")
        print("="*80)
        try:
            eqs_3sls = {}
            for dep, info in parsed_equations.items():
                rhs_parts = info['exog'][:]
                
                if info['endog'] and info['instr']:
                    rhs_parts.append(f"[{' + '.join(info['endog'])} ~ {' + '.join(info['instr'])}]")
                
                eqs_3sls[dep] = f"{dep} ~ {' + '.join(rhs_parts)}"

            print("\n3SLS 방정식 정의:")
            for k, v in eqs_3sls.items():
                print(f"  {k}: {v}")

            mod_3sls = IV3SLS.from_formula(eqs_3sls, data)
            res_3sls = mod_3sls.fit(cov_type="unadjusted")
            print(f"\n{res_3sls}")
        except Exception as e:
            print(f"3SLS 추정 실패: {e}")

        print("\n" + "="*80)
        print("--- 5. GMM 추정 (Generalized Method of Moments) ---")
        print("="*80)
        try:
            mod_gmm = IVSystemGMM.from_formula(eqs_3sls, data, weight_type="unadjusted")
            res_gmm = mod_gmm.fit(cov_type="unadjusted", iter_limit=100)
            print(f"GMM 반복 횟수(iterations): {res_gmm.iterations}")
            print(f"\n{res_gmm}")
        except Exception as e:
            print(f"GMM 추정 실패: {e}")

    # ========================================================================
    # SUR 모형인 경우
    # ========================================================================
    else:
        print("\n" + "="*80)
        print("--- 2. OLS 추정 (개별 방정식 - Individual Equations) ---")
        print("="*80)
        
        for dep, info in parsed_equations.items():
            try:
                all_rhs = info['exog']
                X = add_constant(data[all_rhs])
                y = data[dep]
                
                model_ols = OLS(y, X)
                result_ols = model_ols.fit()
                
                print(f"\n#### OLS 결과 ({dep}) ####")
                print(result_ols.summary())
            except Exception as e:
                print(f"{dep}: OLS 추정 실패: {e}")

        print("\n" + "="*80)
        print("--- 3. SUR 추정 (Seemingly Unrelated Regression) ---")
        print("오차항 간 상관관계를 고려하여 효율성을 개선합니다.")
        print("="*80)
        try:
            sur_eqs = {dep: f"{dep} ~ {' + '.join(info['exog'])}" 
                       for dep, info in parsed_equations.items()}
            
            print("\nSUR 방정식 정의:")
            for k, v in sur_eqs.items():
                print(f"  {k}: {v}")
            
            mod_sur = SUR.from_formula(sur_eqs, data)
            res_sur = mod_sur.fit(cov_type="unadjusted")
            print(f"\n{res_sur}")
        except Exception as e:
            print(f"SUR 추정 실패: {e}")

    print("\n" + "="*80)
    if has_endogeneity:
        print("--- 연립방정식 시스템 추정 완료 ---")
        print("💡 해석 가이드:")
        print("   1. OLS: 내생성을 무시한 편향된 결과 (비교용)")
        print("   2. 2SLS: 개별 방정식의 일치추정량")
        print("   3. 3SLS: 시스템 전체를 고려한 효율적 추정량")
        print("   4. GMM: 일반화된 적률법 (이분산성에 강건)")
    else:
        print("--- SUR 시스템 추정 완료 ---")
        print("💡 해석 가이드:")
        print("   1. OLS: 개별 방정식 추정")
        print("   2. SUR: 오차항 상관관계를 고려한 효율적 추정")
    print("="*80)