#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
�跮������ ���� �м� �Լ�
���ϸ�: Functions/econometric_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo
import statsmodels.stats.diagnostic as dg
import patsy as pt
import scipy.stats as stats
from stargazer.stargazer import Stargazer
import warnings
warnings.filterwarnings("ignore")

# �ѱ� ��Ʈ ���� (matplotlib���� �ѱ� ���� ����)
plt.rcParams['font.family'] = 'DejaVu Sans'
try:
    # Windows�� ���
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        # �ٸ� OS�� ���
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

def Run_Comprehensive_OLS_analysis(data, dependent_var, independent_vars, data_type='cross_section', alpha=0.05, output_format='text'):
    """
    �跮������ ���� �м� �Լ�
    
    Parameters:
    -----------
    data : pandas.DataFrame
        �м��� ������
    dependent_var : str
        ���Ӻ�����
    independent_vars : list
        ���������� ����Ʈ
    data_type : str, default='cross_section'
        ������ ���� ('cross_section' �Ǵ� 'time_series')
    alpha : float, default=0.05
        ���Ǽ���
    output_format : str, default='text'
        ��� ���� ('text', 'latex', 'html')
    
    Returns:
    --------
    dict: �м� ����� ���� ��ųʸ�
    """
    
    print("="*80)
    print("                   �跮������ ���� �м� ���")
    print("="*80)
    
    # 1. ���� ������ ����
    print("\n1. ������ �⺻ ����")
    print("-"*50)
    print(f"����ġ ��: {len(data)}")
    print(f"���Ӻ���: {dependent_var}")
    print(f"��������: {', '.join(independent_vars)}")
    print(f"������ ����: {data_type}")
    
    # �м��� ����� ������
    all_vars = [dependent_var] + independent_vars
    analysis_data = data[all_vars].dropna()
    
    print(f"����ġ ���� �� ����ġ ��: {len(analysis_data)}")
    
    # 2. ���� ��跮
    print("\n2. ���� ��跮")
    print("-"*50)
    desc_stats = analysis_data.describe().round(4)
    display(desc_stats)
    
    # 3. ������ ���
    print("\n3. ������ ���")
    print("-"*50)
    corr_matrix = analysis_data.corr().round(4)
    display(corr_matrix)
    
    # 4. �ð�ȭ
    print("\n4. ������ �ð�ȭ")
    print("-"*50)
    
    # ������׷�
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, var in enumerate(all_vars):
        if i < 4:
            axes[i].hist(analysis_data[var], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{var} Distribution')
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # ���Ӻ����� ���������� ������
    fig, axes = plt.subplots(1, len(independent_vars), figsize=(5*len(independent_vars), 4))
    if len(independent_vars) == 1:
        axes = [axes]
    
    for i, var in enumerate(independent_vars):
        axes[i].scatter(analysis_data[var], analysis_data[dependent_var], alpha=0.6)
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(dependent_var)
        axes[i].set_title(f'{dependent_var} vs {var}')
    
    plt.tight_layout()
    plt.show()
    
    # 5. OLS ȸ�ͺм�
    print("\n5. OLS ȸ�ͺм� ���")
    print("-"*50)
    
    # ȸ�ͽ� ����
    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
    
    # OLS ����
    ols_model = smf.ols(formula=formula, data=analysis_data)
    ols_results = ols_model.fit()
    
    display(ols_results.summary())
    
    results_dict = {
        'ols_results': ols_results,
        'formula': formula,
        'data': analysis_data
    }
    
    # 6. ���Լ� ����
    print("\n6. ������ ���Լ� ����")
    print("-"*50)
    
    residuals = ols_results.resid
    
    # Shapiro-Wilk ����
    sw_stat, sw_pval = stats.shapiro(residuals)
    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {sw_stat:.4f}")
    print(f"  p-value: {sw_pval:.4f}")
    print(f"  Result: {'Reject Normality' if sw_pval < alpha else 'Accept Normality'}")
    
    # Jarque-Bera ����
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    print(f"\nJarque-Bera Test:")
    print(f"  Statistic: {jb_stat:.4f}")
    print(f"  p-value: {jb_pval:.4f}")
    print(f"  Result: {'Reject Normality' if jb_pval < alpha else 'Accept Normality'}")
    
    # ���� ������׷�
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.tight_layout()
    plt.show()
    
    results_dict['normality_tests'] = {
        'shapiro_wilk': {'statistic': sw_stat, 'pvalue': sw_pval},
        'jarque_bera': {'statistic': jb_stat, 'pvalue': jb_pval}
    }
    
    # 7. �������� ���� ���� (RESET Test)
    print("\n7. �������� ���� ���� (RESET Test)")
    print("-"*50)
    
    try:
        reset_result = smo.reset_ramsey(res=ols_results, degree=3)
        reset_stat = reset_result.statistic
        reset_pval = reset_result.pvalue
        
        print(f"RESET Test (up to 3rd order):")
        print(f"  F-statistic: {reset_stat:.4f}")
        print(f"  p-value: {reset_pval:.4f}")
        print(f"  Result: {'Model Misspecification' if reset_pval < alpha else 'Model is Well Specified'}")
        
        results_dict['reset_test'] = {
            'statistic': reset_stat, 
            'pvalue': reset_pval
        }
    except:
        print("Cannot perform RESET test.")
        results_dict['reset_test'] = None
    
    # 8. ���߰����� ���� (VIF)
    print("\n8. ���߰����� ���� (VIF)")
    print("-"*50)
    
    if len(independent_vars) > 1:
        # VIF ����� ���� ��� ����
        y, X = pt.dmatrices(formula, data=analysis_data, return_type='dataframe')
        
        # VIF ���
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns[1:]  # ����� ����
        vif_data["VIF"] = [smo.variance_inflation_factor(X.values, i) 
                          for i in range(1, X.shape[1])]
        
        display(vif_data)
        
        # VIF �ؼ�
        high_vif = vif_data[vif_data['VIF'] > 10]
        if len(high_vif) > 0:
            print("Warning: High multicollinearity detected (VIF > 10):")
            print(high_vif['Variable'].tolist())
        else:
            print("No serious multicollinearity issues (all VIF < 10)")
            
        results_dict['vif'] = vif_data
    else:
        print("Single independent variable - multicollinearity test skipped.")
        results_dict['vif'] = None
    
    # 9. Ⱦ�ܸ� �ڷ��� ���: �̺л� ���� �� ��� ������
    if data_type == 'cross_section':
        print("\n9. �̺л꼺 ���� �� ��� ������")
        print("-"*50)
        
        # Breusch-Pagan ����
        y, X = pt.dmatrices(formula, data=analysis_data, return_type='dataframe')
        bp_result = dg.het_breuschpagan(ols_results.resid, X)
        bp_stat, bp_pval = bp_result[0], bp_result[1]
        
        print(f"Breusch-Pagan Test:")
        print(f"  LM statistic: {bp_stat:.4f}")
        print(f"  p-value: {bp_pval:.4f}")
        print(f"  Result: {'Heteroskedasticity Present' if bp_pval < alpha else 'Homoskedasticity Accepted'}")
        
        # White ����
        X_white = pd.DataFrame({
            'const': 1, 
            'fitted': ols_results.fittedvalues,
            'fitted_sq': ols_results.fittedvalues ** 2
        })
        white_result = dg.het_breuschpagan(ols_results.resid, X_white)
        white_stat, white_pval = white_result[0], white_result[1]
        
        print(f"\nWhite Test:")
        print(f"  LM statistic: {white_stat:.4f}")
        print(f"  p-value: {white_pval:.4f}")
        print(f"  Result: {'Heteroskedasticity Present' if white_pval < alpha else 'Homoskedasticity Accepted'}")
        
        # �̺л꼺 Ž�� ���� Ȯ��
        heteroskedastic = bp_pval < alpha or white_pval < alpha
        print(f"\n�̺л꼺 Ž�� ����: {heteroskedastic}")
        print(f"BP p-value < {alpha}: {bp_pval < alpha}")
        print(f"White p-value < {alpha}: {white_pval < alpha}")
        
        # ���� ����� ���� ����
        results_dict['heteroskedasticity_tests'] = {
            'bp_test': {'statistic': bp_stat, 'pvalue': bp_pval},
            'white_test': {'statistic': white_stat, 'pvalue': white_pval}
        }
        
        # ��� ������ ���� (�̺л꼺 Ž�� ���ο� ������� �����ؼ� ��)
        print(f"\n��� �������� �����մϴ�...")
        
        # White�� ���� ǥ�ؿ��� (�׻� ����)
        try:
            print("  - White ���� ǥ�ؿ��� ���� ��...")
            ols_robust = ols_model.fit(cov_type='HC3')
            results_dict['ols_robust'] = ols_robust
            print("  ? White ���� ǥ�ؿ��� �Ϸ�")
        except Exception as e:
            print(f"  ? White ���� ǥ�ؿ��� ����: {e}")
            results_dict['ols_robust'] = None
        
        # WLS (�����ּ��ڽ¹�) - �׻� ����
        try:
            print("  - WLS ���� ��...")
            # ������ �������� ����ġ ����
            abs_resid = np.abs(ols_results.resid)
            # 0���� ������ ����
            abs_resid = np.where(abs_resid < 1e-6, 1e-6, abs_resid)
            weights = 1 / abs_resid
            weights = weights / np.mean(weights)  # ǥ��ȭ
            
            wls_model = smf.wls(formula=formula, data=analysis_data, weights=weights)
            wls_results = wls_model.fit()
            results_dict['wls_results'] = wls_results
            print("  ? WLS ���� �Ϸ�")
        except Exception as e:
            print(f"  ? WLS ���� ����: {e}")
            results_dict['wls_results'] = None
        
        if heteroskedastic:
            print(f"\n? �̺л꼺�� Ž���Ǿ� ��� �������� �����մϴ�.")
        else:
            print("??  �̺л꼺�� Ž������ �ʾ�����, �񱳸� ���� ��� �������� �����߽��ϴ�.")
        
        # ��� Ȯ��
        print(f"\n���� ��� ���:")
        print(f"  - OLS: �Ϸ�")
        print(f"  - OLS (Robust SE): {'�Ϸ�' if results_dict.get('ols_robust') is not None else '����'}")
        print(f"  - WLS: {'�Ϸ�' if results_dict.get('wls_results') is not None else '����'}")
    
    # 10. �ð迭 �ڷ��� ���: �ڱ��� ���� �� ��� ������  
    elif data_type == 'time_series':
        print("\n9. �ڱ��� ���� �� ��� ������")
        print("-"*50)
        
        # Durbin-Watson ����
        dw_stat = sm.stats.stattools.durbin_watson(ols_results.resid)
        print(f"Durbin-Watson Test:")
        print(f"  DW statistic: {dw_stat:.4f}")
        
        if dw_stat < 1.5:
            dw_interpretation = "Positive autocorrelation suspected"
        elif dw_stat > 2.5:
            dw_interpretation = "Negative autocorrelation suspected"
        else:
            dw_interpretation = "No autocorrelation"
        print(f"  Result: {dw_interpretation}")
        
        # Breusch-Godfrey ����
        bg_result = dg.acorr_breusch_godfrey(ols_results, nlags=3)
        bg_stat, bg_pval = bg_result[2], bg_result[3]
        
        print(f"\nBreusch-Godfrey Test (3 lags):")
        print(f"  F-statistic: {bg_stat:.4f}")
        print(f"  p-value: {bg_pval:.4f}")
        print(f"  Result: {'Autocorrelation Present' if bg_pval < alpha else 'No Autocorrelation'}")
        
        # ARCH ȿ�� ���� (�̺л꼺 + �ڱ���)
        try:
            arch_result = dg.het_arch(ols_results.resid, nlags=3)
            arch_stat, arch_pval = arch_result[2], arch_result[3]
            print(f"\nARCH Test (3 lags):")
            print(f"  F-statistic: {arch_stat:.4f}")
            print(f"  p-value: {arch_pval:.4f}")
            print(f"  Result: {'ARCH Effects Present' if arch_pval < alpha else 'No ARCH Effects'}")
        except Exception as e:
            print(f"\nARCH Test failed: {e}")
            arch_pval = 1.0  # ���� ���н� ȿ�� �������� ����
        
        # ������ ���� (�⺻���� �ð迭 �м�)
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # ���Ӻ����� ���� ADF ����
            adf_result = adfuller(analysis_data[dependent_var].dropna(), autolag='AIC')
            adf_stat, adf_pval = adf_result[0], adf_result[1]
            
            print(f"\nAugmented Dickey-Fuller Test (���Ӻ���):")
            print(f"  ADF statistic: {adf_stat:.4f}")
            print(f"  p-value: {adf_pval:.4f}")
            print(f"  Result: {'Stationary' if adf_pval < alpha else 'Non-stationary (Unit Root)'}")
            
        except Exception as e:
            print(f"\nADF Test failed: {e}")
            adf_pval = 1.0
        
        # ���� Ž�� ���� Ȯ��
        autocorrelated = dw_stat < 1.5 or dw_stat > 2.5 or bg_pval < alpha
        arch_effects = arch_pval < alpha
        non_stationary = adf_pval >= alpha
        
        print(f"\n�ð迭 ���� Ž�� ���:")
        print(f"  �ڱ���: {autocorrelated}")
        print(f"  ARCH ȿ��: {arch_effects}")
        print(f"  ������: {non_stationary}")
        
        # ���� ����� ���� ����
        results_dict['time_series_tests'] = {
            'dw_test': {'statistic': dw_stat, 'interpretation': dw_interpretation},
            'bg_test': {'statistic': bg_stat, 'pvalue': bg_pval},
            'arch_test': {'statistic': arch_stat if 'arch_stat' in locals() else 'N/A', 
                         'pvalue': arch_pval},
            'adf_test': {'statistic': adf_stat if 'adf_stat' in locals() else 'N/A', 
                        'pvalue': adf_pval}
        }
        
        # ��� ������ ���� (���� Ž�� ���ο� ������� �����ؼ� ��)
        print(f"\n��� �������� �����մϴ�...")
        
        # 1. HAC ǥ�ؿ��� (Newey-West) - �׻� ����
        try:
            print("  - HAC ǥ�ؿ��� (Newey-West) ���� ��...")
            ols_hac = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': 2})
            results_dict['ols_hac'] = ols_hac
            print("  ? HAC ǥ�ؿ��� �Ϸ�")
        except Exception as e:
            print(f"  ? HAC ǥ�ؿ��� ����: {e}")
            results_dict['ols_hac'] = None
        
        # 2. Cochrane-Orcutt ������ - �׻� ����
        try:
            print("  - Cochrane-Orcutt ���� ��...")
            y, X = pt.dmatrices(formula, data=analysis_data, return_type='dataframe')
            corc_model = sm.GLSAR(y, X)
            corc_results = corc_model.iterative_fit(maxiter=100)
            results_dict['corc_results'] = corc_results
            print("  ? Cochrane-Orcutt ���� �Ϸ�")
        except Exception as e:
            print(f"  ? Cochrane-Orcutt ���� ����: {e}")
            results_dict['corc_results'] = None
        
        # 3. Prais-Winsten ������ - ���� �߰�
        try:
            print("  - Prais-Winsten ���� ��...")
            from statsmodels.tsa.arima_model import ARIMA
            from statsmodels.regression.linear_model import yule_walker
            
            # AR(1) ��� ����
            rho_yw = yule_walker(ols_results.resid, order=1)[0][0]
            
            # Prais-Winsten ��ȯ
            y_pw = analysis_data[dependent_var].copy()
            X_pw = analysis_data[independent_vars].copy()
            
            # ù ��° ����ġ ��ȯ: sqrt(1-rho^2)
            factor = np.sqrt(1 - rho_yw**2)
            y_pw.iloc[0] = y_pw.iloc[0] * factor
            X_pw.iloc[0] = X_pw.iloc[0] * factor
            
            # ������ ����ġ ��ȯ: Y_t - rho*Y_(t-1)
            for i in range(1, len(y_pw)):
                y_pw.iloc[i] = y_pw.iloc[i] - rho_yw * y_pw.iloc[i-1]
                X_pw.iloc[i] = X_pw.iloc[i] - rho_yw * X_pw.iloc[i-1]
            
            # ��ȯ�� �����ͷ� OLS ����
            pw_data = pd.concat([y_pw, X_pw], axis=1)
            pw_formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
            pw_model = smf.ols(formula=pw_formula, data=pw_data)
            pw_results = pw_model.fit()
            
            results_dict['prais_winsten_results'] = pw_results
            print("  ? Prais-Winsten ���� �Ϸ�")
        except Exception as e:
            print(f"  ? Prais-Winsten ���� ����: {e}")
            results_dict['prais_winsten_results'] = None
        
        # 4. GARCH(1,1) ���� (ARCH ȿ���� �ִ� ��� ����)
        if arch_effects:
            try:
                print("  - GARCH(1,1) ���� ��... (ARCH ȿ�� Ž����)")
                from arch import arch_model
                
                # GARCH �� ����
                garch_model = arch_model(analysis_data[dependent_var], 
                                       x=analysis_data[independent_vars],
                                       vol='GARCH', p=1, q=1)
                garch_results = garch_model.fit(disp='off')
                results_dict['garch_results'] = garch_results
                print("  ? GARCH(1,1) ���� �Ϸ�")
            except Exception as e:
                print(f"  ? GARCH(1,1) ���� ����: {e}")
                results_dict['garch_results'] = None
        else:
            results_dict['garch_results'] = None
        
        # ��� ���
        any_problems = autocorrelated or arch_effects or non_stationary
        if any_problems:
            print(f"\n? �ð迭 ������ Ž���Ǿ� ��� �������� �����մϴ�.")
            if autocorrelated:
                print("   �� �ڱ���: HAC ǥ�ؿ���, Cochrane-Orcutt, Prais-Winsten ����")
            if arch_effects:
                print("   �� ARCH ȿ��: GARCH �� ����")
            if non_stationary:
                print("   �� ������: ����(differencing) �Ǵ� ������ �м� ���")
        else:
            print("??  �ð迭 ������ Ž������ �ʾ�����, �񱳸� ���� ��� �������� �����߽��ϴ�.")
        
        # ��� Ȯ��
        print(f"\n���� ��� ���:")
        print(f"  - OLS: �Ϸ�")
        print(f"  - OLS (HAC SE): {'�Ϸ�' if results_dict.get('ols_hac') is not None else '����'}")
        print(f"  - Cochrane-Orcutt: {'�Ϸ�' if results_dict.get('corc_results') is not None else '����'}")
        print(f"  - Prais-Winsten: {'�Ϸ�' if results_dict.get('prais_winsten_results') is not None else '����'}")
        print(f"  - GARCH(1,1): {'�Ϸ�' if results_dict.get('garch_results') is not None else '�̽���/����'}")
    
    # 11. ��� ���ǥ ����
    print("\n10. ���� ��� ���ǥ")
    print("-"*50)
    
    # ��� �𵨵��� ����Ʈ�� ����
    models_to_display = [ols_results]
    model_names = ['OLS']
    
    print(f"�⺻ OLS �� �߰� �Ϸ�")
    
    # ������ ������ �߰� �𵨵�
    if data_type == 'cross_section':
        print("Ⱦ�ܸ� �ڷ� - �̺л� ���� �𵨵� Ȯ�� ��...")
        
        if 'ols_robust' in results_dict and results_dict['ols_robust'] is not None:
            models_to_display.append(results_dict['ols_robust'])
            model_names.append('OLS (Robust SE)')
            print("  ? OLS (Robust SE) �߰�")
        else:
            print("  ? OLS (Robust SE) ����")
            
        if 'wls_results' in results_dict and results_dict['wls_results'] is not None:
            models_to_display.append(results_dict['wls_results'])
            model_names.append('WLS')
            print("  ? WLS �߰�")
        else:
            print("  ? WLS ����")
    
    elif data_type == 'time_series':
        print("�ð迭 �ڷ� - �ڱ��� ���� �𵨵� Ȯ�� ��...")
        
        if 'ols_hac' in results_dict and results_dict['ols_hac'] is not None:
            models_to_display.append(results_dict['ols_hac'])
            model_names.append('OLS (HAC SE)')
            print("  ? OLS (HAC SE) �߰�")
        else:
            print("  ? OLS (HAC SE) ����")
            
        if 'corc_results' in results_dict and results_dict['corc_results'] is not None:
            models_to_display.append(results_dict['corc_results'])
            model_names.append('Cochrane-Orcutt')
            print("  ? Cochrane-Orcutt �߰�")
        else:
            print("  ? Cochrane-Orcutt ����")
            
        if 'prais_winsten_results' in results_dict and results_dict['prais_winsten_results'] is not None:
            models_to_display.append(results_dict['prais_winsten_results'])
            model_names.append('Prais-Winsten')
            print("  ? Prais-Winsten �߰�")
        else:
            print("  ? Prais-Winsten ����")
        
        # GARCH�� ������ �ٸ��Ƿ� �Ϲ����� ǥ�� �������� ����
        if 'garch_results' in results_dict and results_dict['garch_results'] is not None:
            print("  ??  GARCH ���� ���� ��µ�")
        else:
            print("  ? GARCH ����")
    
    print(f"\n���� �� ��: {len(models_to_display)}")
    print(f"���� �𵨸�: {model_names}")
    
    # ������ ���� ǥ ���� (�� ������)
    print("\n���� ǥ ������ �����մϴ�...")
    try:
        print("="*100)
        print("                              ���� ��� ���ǥ")
        print("="*100)
        
        # ��� ���� ����� DataFrame���� ����
        summary_data = []
        var_names = None
        
        for i, (model, name) in enumerate(zip(models_to_display, model_names)):
            try:
                print(f"  �� {i+1}: {name} ó�� ��...")
                
                # ��� ����
                coeffs = model.params
                std_errors = model.bse
                t_values = model.tvalues
                p_values = model.pvalues
                
                if i == 0:  # ù ��° �𵨿��� ������ ����
                    var_names = list(coeffs.index)
                    print(f"    ������: {var_names}")
                
                summary_data.append({
                    'Model': name,
                    'Coefficients': coeffs,
                    'Std_Errors': std_errors,
                    'T_Values': t_values,
                    'P_Values': p_values,
                    'R_squared': getattr(model, 'rsquared', 'N/A'),
                    'Adj_R_squared': getattr(model, 'rsquared_adj', 'N/A'),
                    'N_obs': int(model.nobs) if hasattr(model, 'nobs') else 'N/A'
                })
                print(f"    ? {name} ó�� �Ϸ�")
                
            except Exception as e:
                print(f"    ? �� {name} ó�� �� ����: {e}")
                continue
        
        print(f"\nó���� �� ��: {len(summary_data)}")
        
        # ǥ ���
        if summary_data and var_names:
            # ��� ���
            col_width = 20
            header = "Variable".ljust(15)
            for name in model_names[:len(summary_data)]:
                header += name.center(col_width)
            print(header)
            print("-" * len(header))
            
            # ��� ���
            for var in var_names:
                # ��� ��
                row = var.ljust(15)
                for data in summary_data:
                    try:
                        coeff = data['Coefficients'][var]
                        p_val = data['P_Values'][var]
                        
                        # ���Ǽ� ǥ��
                        sig = ""
                        if p_val < 0.01:
                            sig = "***"
                        elif p_val < 0.05:
                            sig = "**"
                        elif p_val < 0.1:
                            sig = "*"
                        
                        coeff_str = f"{coeff:.4f}{sig}"
                        row += f"{coeff_str}".center(col_width)
                        
                    except Exception as e:
                        print(f"      ���� {var} ó�� ����: {e}")
                        row += "N/A".center(col_width)
                print(row)
                
                # ǥ�ؿ��� ��
                se_row = "".ljust(15)
                for data in summary_data:
                    try:
                        se = data['Std_Errors'][var]
                        se_str = f"({se:.4f})"
                        se_row += f"{se_str}".center(col_width)
                    except:
                        se_row += "".center(col_width)
                print(se_row)
            
            print("-" * len(header))
            
            # �� ��跮
            stats_row = "R-squared".ljust(15)
            for data in summary_data:
                r2 = data['R_squared']
                if isinstance(r2, (int, float)):
                    stats_row += f"{r2:.4f}".center(col_width)
                else:
                    stats_row += f"{r2}".center(col_width)
            print(stats_row)
            
            adj_r2_row = "Adj R-squared".ljust(15)
            for data in summary_data:
                adj_r2 = data['Adj_R_squared']
                if isinstance(adj_r2, (int, float)):
                    adj_r2_row += f"{adj_r2:.4f}".center(col_width)
                else:
                    adj_r2_row += f"{adj_r2}".center(col_width)
            print(adj_r2_row)
            
            n_row = "Observations".ljust(15)
            for data in summary_data:
                n_obs = data['N_obs']
                n_row += f"{n_obs}".center(col_width)
            print(n_row)
            
            print("-" * len(header))
            print("Significance: *** p<0.01, ** p<0.05, * p<0.1")
            print("="*100)
            
            results_dict['manual_table'] = summary_data
            print("? ���� ǥ ���� �Ϸ�!")
            
        else:
            print("? ǥ ������ ���� �����Ͱ� �����մϴ�.")
            print("�⺻ ȸ�� ����� ���������� ����մϴ�:")
            for i, model in enumerate(models_to_display):
                print(f"\n{model_names[i]} Results:")
                print(model.summary().tables[1])
        
    except Exception as e:
        print(f"? ���� ǥ ���� �� ����: {e}")
        print("�⺻ ȸ�� ����� ���������� ����մϴ�:")
        for i, model in enumerate(models_to_display):
            print(f"\n{model_names[i]} Results:")
            try:
                print(model.summary().tables[1])
            except:
                print("�� ��� ��� ����")
    
    # Stargazer �õ� (�ɼ�) - �ܼ��ϰ� ������ ����
    print(f"\nStargazer ǥ ������ �õ��մϴ�...")
    stargazer_success = False
    
    try:
        from stargazer.stargazer import Stargazer
        stargazer = Stargazer(models_to_display)
        stargazer.custom_columns(model_names, [1]*len(model_names))
        stargazer.title("Econometric Analysis Results")
        stargazer.show_degrees_of_freedom(False)
        
        print("\n" + "="*100)
        print("                         Stargazer ���ǥ")
        print("="*100)
        
        # ��� ���ĺ� ó�� (�����ϰ�)
        if output_format.lower() == 'latex':
            latex_output = stargazer.render_latex()
            print(latex_output)
            stargazer_success = True
            
        elif output_format.lower() == 'html':
            html_output = stargazer.render_html()
            print(html_output)
            stargazer_success = True
            
        else:  # text (�⺻��) - �ܼ��� ����� ���
            print("�⺻ �ؽ�Ʈ ������ �������� �ʽ��ϴ�.")
            print("LaTeX �������� ����մϴ�:")
            print("-" * 80)
            latex_output = stargazer.render_latex()
            print(latex_output)
            print("-" * 80)
            stargazer_success = True
        
        results_dict['stargazer_table'] = stargazer
        results_dict['output_format'] = output_format
        
        if stargazer_success:
            print("? Stargazer ǥ ���� �Ϸ�!")
        
    except ImportError:
        print("? Stargazer ���̺귯���� ��ġ���� ����")
        print("   ��ġ ���: pip install stargazer")
    except Exception as e:
        print(f"? Stargazer ����: {e}")
    
    if not stargazer_success:
        print("?? ���� ���� ���� ǥ�� �����ϼ���.")
    
    # �߰�: ����ȭ�� ���ǥ ����
    print(f"\n" + "="*100)
    print("                           ����ȭ�� ���� ���")
    print("="*100)
    
    try:
        # �ٽ� ��跮�� ������ ����
        print(f"{'Model':<20} {'R��':<10} {'Adj R��':<10} {'N':<8} {'Key Variables'}")
        print("-" * 100)
        
        for i, (model, name) in enumerate(zip(models_to_display, model_names)):
            try:
                r2 = getattr(model, 'rsquared', 'N/A')
                adj_r2 = getattr(model, 'rsquared_adj', 'N/A')
                n_obs = int(model.nobs) if hasattr(model, 'nobs') else 'N/A'
                
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
                adj_r2_str = f"{adj_r2:.4f}" if isinstance(adj_r2, (int, float)) else str(adj_r2)
                
                # �ֿ� ������� ���Ǽ� Ȯ��
                sig_vars = []
                try:
                    for var, pval in model.pvalues.items():
                        if pval < 0.05 and var != 'Intercept':
                            if pval < 0.01:
                                sig_vars.append(f"{var}***")
                            elif pval < 0.05:
                                sig_vars.append(f"{var}**")
                except:
                    sig_vars = ["See table above"]
                
                key_vars = ", ".join(sig_vars[:3]) if sig_vars else "None significant"
                
                print(f"{name:<20} {r2_str:<10} {adj_r2_str:<10} {n_obs:<8} {key_vars}")
                
            except Exception as e:
                print(f"{name:<20} {'Error':<10} {'Error':<10} {'Error':<8} Error processing")
        
        print("-" * 100)
        print("*** p<0.01, ** p<0.05")
        print("="*100)
        
        # GARCH �� ���� ��� (�ð迭�� ���)
        if data_type == 'time_series' and 'garch_results' in results_dict and results_dict['garch_results'] is not None:
            print(f"\n" + "="*100)
            print("                           GARCH(1,1) �� ���")
            print("="*100)
            try:
                garch_res = results_dict['garch_results']
                print("��� ������ (Mean Equation):")
                print(garch_res.summary().tables[1])
                print("\n�л� ������ (Variance Equation):")
                print(garch_res.summary().tables[2])
                print("="*100)
            except Exception as e:
                print(f"GARCH ��� ��� ����: {e}")
        
        # �ð迭 ���� ��� ���
        if data_type == 'time_series' and 'time_series_tests' in results_dict:
            print(f"\n" + "="*100)
            print("                           �ð迭 ���� ��� ���")
            print("="*100)
            tests = results_dict['time_series_tests']
            
            print(f"{'Test':<25} {'Statistic':<15} {'P-value':<15} {'Result'}")
            print("-" * 100)
            
            # DW Test
            dw_stat = tests['dw_test']['statistic']
            dw_interp = tests['dw_test']['interpretation']
            print(f"{'Durbin-Watson':<25} {dw_stat:<15.4f} {'N/A':<15} {dw_interp}")
            
            # BG Test  
            bg_stat = tests['bg_test']['statistic']
            bg_pval = tests['bg_test']['pvalue']
            bg_result = "Autocorrelation" if bg_pval < 0.05 else "No Autocorrelation"
            print(f"{'Breusch-Godfrey':<25} {bg_stat:<15.4f} {bg_pval:<15.4f} {bg_result}")
            
            # ARCH Test
            if tests['arch_test']['statistic'] != 'N/A':
                arch_stat = tests['arch_test']['statistic']
                arch_pval = tests['arch_test']['pvalue']
                arch_result = "ARCH Effects" if arch_pval < 0.05 else "No ARCH Effects"
                print(f"{'ARCH':<25} {arch_stat:<15.4f} {arch_pval:<15.4f} {arch_result}")
            
            # ADF Test
            if tests['adf_test']['statistic'] != 'N/A':
                adf_stat = tests['adf_test']['statistic']
                adf_pval = tests['adf_test']['pvalue']
                adf_result = "Stationary" if adf_pval < 0.05 else "Non-stationary"
                print(f"{'ADF (Unit Root)':<25} {adf_stat:<15.4f} {adf_pval:<15.4f} {adf_result}")
            
            print("="*100)
        
    except Exception as e:
        print(f"����ȭ�� ��� ���� ����: {e}")
        
    print("\n? ��� �м��� �Ϸ�Ǿ����ϴ�!")
    
    if data_type == 'time_series':
        print("?? �ð迭 �м� ���:")
        print("   - �ֿ� ȸ�� ����� ���� '���� ǥ ����' ���� ����")
        print("   - GARCH �� ����� ���� ǥ�õ�")
        print("   - �ð迭 ���� ��� ��� Ȯ��")
    else:
        print("?? �ֿ� ����� ���� '���� ǥ ����' ������ �����ϼ���.")
    
    print("\n" + "="*80)
    print("                        Analysis Complete")
    print("="*80)
    
    return results_dict

print("�� �Լ��� ������ �跮������ ���� �м� �Լ��Դϴ�. ")
