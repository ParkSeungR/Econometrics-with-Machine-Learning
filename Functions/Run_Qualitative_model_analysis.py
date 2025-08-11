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
    ������ ���Ӻ��� ������ ���� �������� �м��� �����ϴ� �Լ�.

    Args:
        data (pd.DataFrame): �м��� ����� ������������.
        formula (str): ���Ӻ��� ~ ��������1 + ��������2 + ... ������ ȸ�ͽ�.
                       ��: 'inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6'
    """
    # --- �ѱ� ��Ʈ ���� ---
    font_name = fm.FontProperties(fname=r'C:/Windows/Fonts/malgun.ttf').get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False # ���̳ʽ� ��ȣ ���� ����

    # ȸ�ͽĿ��� ���Ӻ����� �������� �̸� ����
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

    print("--- 1. ���� ��� �м� ---")
    print("\n[���� ��跮 (�Լ� ������ ���� ������)]")
    if not data_for_stats.empty:
        print(data_for_stats.describe().round(4))
    else:
        print("���: ȸ�ͽĿ� �ش��ϴ� ���� �� �����Ϳ��� ã�� �� �ִ� ������ �����ϴ�. ���� ��� �м��� �ǳʍ��ϴ�.")

    print("\n[������ ��� (�Լ� ������ ���� ������)]")
    if not data_for_stats.empty:
        print(data_for_stats.corr().round(4))
    else:
        print("���: ȸ�ͽĿ� �ش��ϴ� ���� �� �����Ϳ��� ã�� �� �ִ� ������ �����ϴ�. ������ ����� �ǳʍ��ϴ�.")

    # ������׷��� KDE
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
            print("������׷� �� KDE�� �׸� ������ �����ϴ�.")
    else:
        print("������׷� �� KDE�� �׸� ������ �����ϴ�.")


    print("\n--- 2. ���� ���� ---")
    results_lin = None
    results_logit = None
    results_probit = None

    print("\n[���� Ȯ������(LPM) ���� ���]")
    try:
        reg_lin = smf.ols(formula=formula, data=data)
        results_lin = reg_lin.fit(cov_type='HC3')
        print(results_lin.summary())
    except Exception as e:
        print(f"LPM ���� �� ���� �߻�: {e}")

    print("\n[���� ����(Logit Model) ���� ���]")
    try:
        reg_logit = smf.logit(formula=formula, data=data)
        results_logit = reg_logit.fit(disp=0)
        print(results_logit.summary())
    except Exception as e:
        print(f"���� ���� ���� �� ���� �߻�: {e}")

    print("\n[���κ� ����(Probit Model) ���� ���]")
    try:
        reg_probit = smf.probit(formula=formula, data=data)
        results_probit = reg_probit.fit(disp=0)
        print(results_probit.summary())
    except Exception as e:
        print(f"���κ� ���� ���� �� ���� �߻�: {e}")

    print("\n--- 3. �Ѱ�ȿ�� ��� ---")
    
    if results_lin and results_logit and results_probit:
        print("\n[��� �Ѱ�ȿ�� (statsmodels �Լ� �̿�)]")
        try:
            # .margeff �Ӽ��� ���� ����ϰ�, Series�� ����� ��ȯ�Ͽ� index �Ӽ� ����
            ape_logit_autom_series = pd.Series(results_logit.get_margeff().margeff)
            ape_probit_autom_series = pd.Series(results_probit.get_margeff().margeff)
            
            # Series�� �ε����� �̹� �������̹Ƿ�, �̸� ���� ���
            table_auto = pd.DataFrame({
                'APE_logit_autom': np.round(ape_logit_autom_series.values, 4),
                'APE_probit_autom': np.round(ape_probit_autom_series.values, 4)
            }, index=ape_logit_autom_series.index) # �ε����� ���������� ����
            
            print(table_auto)
        except Exception as e:
            print(f"statsmodels get_margeff() ��� �� ���� �߻�: {e}")
    else:
        print("���� ���� ���з� �Ѱ�ȿ���� ����� �� �����ϴ�.")


    # ������ �׷��� (����ġ�� ���յ� �� �׷��� �� ���� �׷���)�� ������.
    print("\n--- �м� �Ϸ�! ---")