import wooldridge as woo
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.font_manager as fm

def Run_Qualitative_model_analysis(data, formula):
    """
    정성적 종속변수 모형에 대한 종합적인 분석을 수행하는 함수.

    Args:
        data (pd.DataFrame): 분석에 사용할 데이터프레임.
        formula (str): 종속변수 ~ 독립변수1 + 독립변수2 + ... 형태의 회귀식.
                       예: 'inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6'
    """
    # --- 한글 폰트 설정 ---
    font_name = fm.FontProperties(fname=r'C:/Windows/Fonts/malgun.ttf').get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

    # 회귀식에서 종속변수와 독립변수 이름 추출
    dependent_var = formula.split('~')[0].strip()
    independent_vars_formula_str = formula.split('~')[1].strip()
    
    actual_independent_cols = []
    parts = [part.strip() for part in independent_vars_formula_str.split('+')]
    
    for part in parts:
        match_I_func = re.search(r'I\((.+?)\)', part)
        if match_I_func:
            inner_expr = match_I_func.group(1)
            variables_in_expr = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', inner_expr)
            actual_independent_cols.extend(variables_in_expr)
        else:
            actual_independent_cols.append(part)
            
    all_vars_for_analysis = list(set([dependent_var] + actual_independent_cols))
    
    existing_vars = [var for var in all_vars_for_analysis if var in data.columns]
    data_for_stats = data[existing_vars]

    print("--- 1. 기초 통계 분석 ---")
    print("\n[기초 통계량 (함수 추정에 사용된 변수만)]")
    if not data_for_stats.empty:
        print(data_for_stats.describe().round(4))
    else:
        print("경고: 회귀식에 해당하는 변수 중 데이터에서 찾을 수 있는 변수가 없습니다. 기초 통계 분석을 건너뜜니다.")

    print("\n[상관계수 행렬 (함수 추정에 사용된 변수만)]")
    if not data_for_stats.empty:
        print(data_for_stats.corr().round(4))
    else:
        print("경고: 회귀식에 해당하는 변수 중 데이터에서 찾을 수 있는 변수가 없습니다. 상관계수 행렬을 건너뜜니다.")

    # 히스토그램과 KDE
    if not data_for_stats.empty:
        plt.figure(figsize=(15, 10))
        num_cols_for_plot = len(data_for_stats.columns)
        if num_cols_for_plot > 0:
            num_rows = int(np.ceil(num_cols_for_plot / 3)) 
            for i, col in enumerate(data_for_stats.columns):
                plt.subplot(num_rows, 3, i + 1)
                sns.histplot(data_for_stats[col], kde=True)
                plt.title(f'Histogram and KDE for {col}')
            plt.tight_layout()
            plt.show()
        else:
            print("히스토그램 및 KDE를 그릴 변수가 없습니다.")
    else:
        print("히스토그램 및 KDE를 그릴 변수가 없습니다.")


    print("\n--- 2. 모형 추정 ---")
    results_lin = None
    results_logit = None
    results_probit = None

    print("\n[선형 확률모형(LPM) 추정 결과]")
    try:
        reg_lin = smf.ols(formula=formula, data=data)
        results_lin = reg_lin.fit(cov_type='HC3')
        print(results_lin.summary())
    except Exception as e:
        print(f"LPM 추정 중 오류 발생: {e}")

    print("\n[로짓 모형(Logit Model) 추정 결과]")
    try:
        reg_logit = smf.logit(formula=formula, data=data)
        results_logit = reg_logit.fit(disp=0)
        print(results_logit.summary())
    except Exception as e:
        print(f"로짓 모형 추정 중 오류 발생: {e}")

    print("\n[프로빗 모형(Probit Model) 추정 결과]")
    try:
        reg_probit = smf.probit(formula=formula, data=data)
        results_probit = reg_probit.fit(disp=0)
        print(results_probit.summary())
    except Exception as e:
        print(f"프로빗 모형 추정 중 오류 발생: {e}")

    print("\n--- 3. 한계효과 계산 ---")
    
    if results_lin and results_logit and results_probit:
        print("\n[평균 한계효과 (statsmodels 함수 이용)]")
        try:
            # .margeff 속성을 직접 사용하고, Series로 명시적 변환하여 index 속성 보장
            ape_logit_autom_series = pd.Series(results_logit.get_margeff().margeff)
            ape_probit_autom_series = pd.Series(results_probit.get_margeff().margeff)
            
            # Series의 인덱스가 이미 변수명이므로, 이를 직접 사용
            table_auto = pd.DataFrame({
                'APE_logit_autom': np.round(ape_logit_autom_series.values, 4),
                'APE_probit_autom': np.round(ape_probit_autom_series.values, 4)
            }, index=ape_logit_autom_series.index) # 인덱스를 변수명으로 설정
            
            print(table_auto)
        except Exception as e:
            print(f"statsmodels get_margeff() 사용 중 오류 발생: {e}")
    else:
        print("모형 추정 실패로 한계효과를 계산할 수 없습니다.")


    # 마지막 그래프 (관측치와 적합된 값 그래프 및 잔차 그래프)는 제거함.
    print("\n--- 분석 완료! ---")