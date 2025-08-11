# Libraries for the Analysis of Traditional Econometrics from A to Z
# Call this file "exec(open('Functions/Comprehensive_regression_analysis').read()"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import acf, pacf, adfuller
from scipy import stats
import statsmodels.stats.outliers_influence as smo # Corrected import for reset_ramsey

def Comprehensive_regression_analysis(data, y_var, X_vars, data_type=None):
    """
    ���� ��ɾ�� ȸ�� �м��� �ʿ��� ��� ����� ����ϴ� �Լ�.
    Ⱦ�ܸ� �Ǵ� �ð迭 �����Ϳ� ���� �̺л�/�ڱ��� ���� �� �ذ� �������� �����մϴ�.

    Args:
        data (pd.DataFrame): �м��� ��ü ������.
        y_var (str): ���� ������.
        X_vars (list): ���� ������ ����Ʈ.
        data_type (str, optional): �������� ������ ��������� �����մϴ�.
                                   'time_series' �Ǵ� 'cross_sectional'.
                                   None�� ��� �ε��� ������ ������� �ڵ� �����մϴ�.
                                   �⺻���� None.
    """

    y = data[y_var]
    X = data[X_vars]

    print("=" * 50)
    print("���� ȸ�� �м� ����")
    print("=" * 50)

    # 0. ������ ���� �ľ� (�ð迭 ����)
    if data_type is None:
        is_time_series = isinstance(data.index, pd.DatetimeIndex) and data.index.is_monotonic_increasing
        print("\n--- 0. ������ ���� �ľ� (�ڵ� ����) ---")
    elif data_type == 'time_series':
        is_time_series = True
        print("\n--- 0. ������ ���� �ľ� (����� ����: �ð迭) ---")
    elif data_type == 'cross_sectional':
        is_time_series = False
        print("\n--- 0. ������ ���� �ľ� (����� ����: Ⱦ�ܸ�) ---")
    else:
        raise ValueError("data_type�� 'time_series', 'cross_sectional' �Ǵ� None�̾�� �մϴ�.")
    
    if is_time_series:
        print("�����ʹ� �ð迭 �����ͷ� �ľǵ˴ϴ�.")
    else:
        print("�����ʹ� Ⱦ�ܸ� �����ͷ� �ľǵ˴ϴ�.")

    # 1. ������ �⺻ ����
    print("\n--- 1. ������ �⺻ ���� ---")
    print(data.info())
    print(f"\n���� ����: {y_var}")
    print(f"���� ����: {X_vars}")

    # 2. ���� ��跮
    print("\n--- 2. ���� ��跮 ---")
    print(data.describe())

    # 3. ������ ���
    print("\n--- 3. ������ ��� ---")
    print(data[X_vars + [y_var]].corr())
    # ������ ��� �׷��� ��� �κ� ����

    # 4. ������ �ð�ȭ
    print("\n--- 4. ������ �ð�ȭ ---")
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(X_vars + [y_var]):
        plt.subplot(1, len(X_vars) + 1, i + 1)
        sns.histplot(data[col], kde=True)
        plt.title(f'������׷� �� KDE: {col}')
    plt.tight_layout()
    plt.show()

    if is_time_series:
        print("\n�ð迭 ������: �ð� ��ȭ�� ���� �� �׷��� �߰�")
        plt.figure(figsize=(15, 6))
        for col in X_vars + [y_var]:
            plt.plot(data.index, data[col], label=col)
        plt.title("�ð� ��ȭ�� ���� ���� ����")
        plt.xlabel("�ð�")
        plt.ylabel("��")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 5. OLS ȸ�ͺм� ���
    print("\n--- 5. OLS ȸ�ͺм� ��� ---")
    X_const = add_constant(X)
    ols_model = sm.OLS(y, X_const)
    ols_results = ols_model.fit()
    print(ols_results.summary())

    # 6. ������ ���Լ� ����
    print("\n--- 6. ������ ���Լ� ���� ---")
    residuals = ols_results.resid

    shapiro_test = stats.shapiro(residuals)
    print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("�͹����� �Ⱒ: ������ ���� ������ ������ �ʽ��ϴ�.")
    else:
        print("�͹����� ä��: ������ ���� ������ �����ٰ� �� �� �ֽ��ϴ�.")

    jb_test = sm.stats.jarque_bera(residuals)
    print(f"Jarque-Bera Test: Statistic={jb_test[0]:.4f}, p-value={jb_test[1]:.4f}")
    if jb_test[1] < 0.05:
        print("�͹����� �Ⱒ: ������ ���� ������ ������ �ʽ��ϴ�.")
    else:
        print("�͹����� ä��: ������ ���� ������ �����ٰ� �� �� �ֽ��ϴ�.")

    fig = sm.qqplot(residuals, line='s')
    plt.title("���� QQ Plot")
    plt.show()

    # 7. �������� ���� ���� (RESET Test)
    print("\n--- 7. �������� ���� ���� (RESET Test) ---")
    try:
        reset_output = smo.reset_ramsey(res=ols_results, degree=3)
        
        fstat_reset = reset_output.statistic
        fpval_reset = reset_output.pvalue

        if not np.isnan(fstat_reset):
            print(f"RESET Test (Ramsey): F-statistic={fstat_reset:.4f}, p-value={fpval_reset:.4f}")
            if fpval_reset < 0.05:
                print("�͹����� �Ⱒ: ���� ���� ������ ������ �� �ֽ��ϴ� (Ramsey RESET test ���).")
            else:
                print("�͹����� ä��: ���� ���� ������ ���ٰ� �� �� �ֽ��ϴ� (Ramsey RESET test ���).")
        else:
            print("RESET Test ����� ǥ���� �� �����ϴ�.")
            
    except Exception as e:
        print(f"RESET Test ���� �� ���� �߻�: {e}")
        print("RESET Test�� �Ϻ� ��Ȳ���� ��ġ�� ������ ������ �߻��� �� �ֽ��ϴ� (��: ������ ũ��, ���߰�����).")


    # 8. ���߰����� ���� (VIF)
    print("\n--- 8. ���߰����� ���� (VIF) ---")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) 
                       for i in range(X_const.shape[1])]
    
    if 'const' in vif_data['Variable'].values:
        vif_data = vif_data[vif_data['Variable'] != 'const']

    print(vif_data)
    print("\n�Ϲ������� VIF ���� 10 �̻��̸� �ɰ��� ���߰������� �ǽ��մϴ�.")

    # 9. �ڱ��� ���� (�ð迭 �������� ��쿡��)
    if is_time_series:
        print("\n--- 9. �ڱ��� ���� (�ð迭 ������) ---")
        print("���� �׷��� (�ð��� ����)")
        plt.figure(figsize=(12, 5))
        plt.plot(data.index, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("�ð��� ���� ���� ����")
        plt.xlabel("�ð�")
        plt.ylabel("����")
        plt.grid(True)
        plt.show()

        # Durbin-Watson Test
        dw_test = durbin_watson(residuals)
        print(f"Durbin-Watson Test: Statistic={dw_test:.4f}")
        print("Durbin-Watson ��跮�� 2�� ������ �ڱ����� ������ ��Ÿ���ϴ�.")
        if dw_test < 1.5 or dw_test > 2.5:
            print("�ڱ����� ������ ���ɼ��� �ֽ��ϴ�.")

        # Breusch-Godfrey Test
        try:
            num_exog = X_const.shape[1]
            max_nlags = len(y) - num_exog - 1
            if max_nlags < 1: max_nlags = 1

            bg_test = sm.stats.acorr_breusch_godfrey(ols_results, nlags=min(int(len(y)/4), max_nlags))
            print(f"Breusch-Godfrey Test: LM Statistic={bg_test[0]:.4f}, p-value={bg_test[1]:.4f}")
            if bg_test[1] < 0.05:
                print("�͹����� �Ⱒ: �ڱ����� �����մϴ�.")
            else:
                print("�͹����� ä��: �ڱ����� ���ٰ� �� �� �ֽ��ϴ�.")
        except Exception as e:
            print(f"Breusch-Godfrey Test ���� �� ���� �߻�: {e}")
            print("����� ����ġ �� �Ǵ� lag ���� ������ ������ �߻��� �� �ֽ��ϴ�.")


        # Augmented Dickey-Fuller Test (������ ���� ����)
        print("\nAugmented Dickey-Fuller Test (������ ���� ����):")
        try:
            adf_test = adfuller(residuals)
            print(f"  ADF Statistic: {adf_test[0]:.4f}")
            print(f"  p-value: {adf_test[1]:.4f}")
            print("  Critical Values:")
            for key, value in adf_test[4].items():
                print(f"    {key}: {value:.4f}")
            if adf_test[1] < 0.05:
                print("�͹����� �Ⱒ: ������ ������ �����ϴ�.")
            else:
                print("�͹����� ä��: ������ �������� �����ϴ� (�������� ������ ���ɼ�).")
        except Exception as e:
            print(f"Augmented Dickey-Fuller Test ���� �� ���� �߻�: {e}")
            print("����� ����ġ �� ���� �Ǵ� ���� Ư�� ������ ������ �߻��� �� �ֽ��ϴ�.")

    # 10. �̺л� ���� (Ⱦ�ܸ� �������� ��쿡��)
    if not is_time_series:
        print("\n--- 10. �̺л� ���� (Ⱦ�ܸ� ������) ---")
        print("���� �׷��� (�������� ����)")
        plt.figure(figsize=(10, 6))
        plt.scatter(ols_results.fittedvalues, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("������")
        plt.ylabel("����")
        plt.title("�������� ���� ���� ������")
        plt.show()

        # Breusch-Pagan test
        try:
            bp_test = het_breuschpagan(residuals, X_const)
            print(f"Breusch-Pagan Test: LM Statistic={bp_test[0]:.4f}, p-value={bp_test[1]:.4f}")
            if bp_test[1] < 0.05:
                print("�͹����� �Ⱒ: �̺л꼺�� �����մϴ�.")
            else:
                print("�͹����� ä��: �̺л꼺�� ���ٰ� �� �� �ֽ��ϴ�.")
        except Exception as e:
            print(f"Breusch-Pagan Test ���� �� ���� �߻�: {e}")

        # White test
        try:
            white_test = het_white(residuals, X_const)
            print(f"White Test: LM Statistic={white_test[0]:.4f}, p-value={white_test[1]:.4f}")
            if white_test[1] < 0.05:
                print("�͹����� �Ⱒ: �̺л꼺�� �����մϴ�.")
            else:
                print("�͹����� ä��: �̺л꼺�� ���ٰ� �� �� �ֽ��ϴ�.")
        except Exception as e:
            print(f"White Test ���� �� ���� �߻�: {e}")
    else:
        print("\n--- 10. �̺л� ���� (�ð迭 �������� ��� �������� ����) ---")


    # 11. �ڱ��� �ذ��� ���� ������ (�ð迭 �������� ��쿡��)
    cochrane_orcutt_results = None
    prais_winsten_results = None # Prais-Winsten�� GLSAR�� �����Ͽ� ó��
    hac_se_results = None

    if is_time_series:
        print("\n--- 11. �ڱ��� �ذ��� ���� ������ (�ð迭 ������) ---")

        # Cochrane-Orcutt (sm.GLSAR ���)
        try:
            print("\n- Cochrane-Orcutt (GLSAR) ���� -")
            # GLSAR �� ����: (1,0)�� AR(1) ������ ����
            # X_const (����� ����)�� �����ؾ� ��
            glsar_model = sm.GLSAR(y, X_const, 1) # 1�� AR(1)�� �ǹ�
            # �ݺ� ���� (�ִ� 100ȸ, ���� ��� ���� 1e-8)
            cochrane_orcutt_results = glsar_model.iterative_fit(maxiter=100, tol=1e-8)
            print(cochrane_orcutt_results.summary())
            print(f"������ AR(1) ��� (rho): {glsar_model.rho[0]:.4f}")

        except Exception as e:
            print(f"Cochrane-Orcutt (GLSAR) ���� �� ���� �߻�: {e}")
            print("GLSAR ������ ������ Ư���̳� ���� ������ ������ �߻��� �� �ֽ��ϴ�.")


        # Prais-Winsten (sm.GLSAR ���)
        print("\n- Prais-Winsten (GLSAR) ���� -")
        try:
            # GLSAR�� �⺻������ ù ����ġ�� ��ȯ�ϴ� Prais-Winsten ����� ����մϴ�.
            # iterative_fit()�� ��ǻ� Prais-Winsten�� �ݺ����� rho ���� ������ �����մϴ�.
            # ���� ������ Prais-Winsten ���� ��������� ������ �ʿ䰡 �����ϴ�.
            # ���� Cochrane-Orcutt (GLSAR) ����� Prais-Winsten�� Ư���� �����մϴ�.
            print("Cochrane-Orcutt (GLSAR) ����� Prais-Winsten ������ Ư���� �����մϴ�.")
            print("Prais-Winsten�� Ưȭ�� ���� ������ �ʿ����� �ʽ��ϴ�. ���� GLSAR ����� �����ϼ���.")
        except Exception as e:
            print(f"Prais-Winsten ���� �� ���� �߻�: {e}")


        # HAC ǥ�ؿ��� (Newey-West)
        print("\n- HAC ǥ�ؿ��� (Newey-West) -")
        try:
            num_exog = X_const.shape[1]
            max_lags_hac = len(y) - num_exog -1
            if max_lags_hac < 1:
                max_lags_hac = 1

            hac_se_results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags_hac})
            print(hac_se_results.summary())
        except Exception as e:
            print(f"HAC ǥ�ؿ��� ���� �� ���� �߻�: {e}")

    else:
        print("\n--- 11. �ڱ��� �ذ��� ���� ������ (Ⱦ�ܸ� �������� ��� �������� ����) ---")

    # 12. �̺л� ���� �ذ� ������ (Ⱦ�ܸ� �������� ��쿡��)
    gls_results = None
    white_robust_results = None

    if not is_time_series:
        print("\n--- 12. �̺л� ���� �ذ� ������ (Ⱦ�ܸ� ������) ---")

        # �Ϲ�ȭ �ּ��ڽ¹�(Generalized Least Squares: GLS)
        try:
            print("\n- �Ϲ�ȭ �ּ��ڽ¹�(GLS) -")
            weights = 1.0 / (residuals**2 + 1e-8)
            gls_model = sm.WLS(y, X_const, weights=weights)
            gls_results = gls_model.fit()
            print(gls_results.summary())
        except Exception as e:
            print(f"GLS ���� �� ���� �߻�: {e}")

        # White��s Robust standard error
        print("\n- White��s Robust standard error -")
        try:
            white_robust_results = ols_model.fit(cov_type='HC3')
            print(white_robust_results.summary())
        except Exception as e:
            print(f"White's Robust standard error ���� �� ���� �߻�: {e}")
    else:
        print("\n--- 12. �̺л� ���� �ذ� ������ (�ð迭 �������� ��� �������� ����) ---")


    # 13. ���� ���� ��� ���ǥ
    print("\n" + "=" * 50)
    print("���� ���� ��� ���ǥ")
    print("=" * 50)

    if is_time_series:
        print("\n--- �ð迭 ������ ���� ������� ���ǥ ---")
        summary_dict_ts = {'OLS': ols_results}
        if cochrane_orcutt_results:
            summary_dict_ts['Cochrane-Orcutt (GLSAR)'] = cochrane_orcutt_results
        if hac_se_results:
            summary_dict_ts['HAC SE'] = hac_se_results

        from statsmodels.iolib.summary2 import summary_col
        try:
            results_table_ts = summary_col(list(summary_dict_ts.values()),
                                           model_names=list(summary_dict_ts.keys()),
                                           info_dict={'N':lambda x: f"{int(x.nobs)}",
                                                      'R2':lambda x: f"{x.rsquared:.4f}"})
            print(results_table_ts)
        except Exception as e:
            print(f"�ð迭 ������ ���� ��� ���ǥ ���� �� ���� �߻�: {e}")
            print("�Ϻ� ���� ����� ������� �� �ֽ��ϴ�.")

        print("\n(����: Cochrane-Orcutt (GLSAR)�� �ݺ� ������ �ڱ��� ���� ����Դϴ�.)")

    if not is_time_series:
        print("\n--- Ⱦ�ܸ� ������ ���� ������� ���ǥ ---")
        summary_dict_cs = {'OLS': ols_results}
        if gls_results:
            summary_dict_cs['GLS'] = gls_results
        if white_robust_results:
            summary_dict_cs['White\'s Robust SE'] = white_robust_results

        try:
            from statsmodels.iolib.summary2 import summary_col
            results_table_cs = summary_col(list(summary_dict_cs.values()),
                                           model_names=list(summary_dict_cs.keys()),
                                           info_dict={'N':lambda x: f"{int(x.nobs)}",
                                                      'R2':lambda x: f"{x.rsquared:.4f}"})
            print(results_table_cs)
        except Exception as e:
            print(f"Ⱦ�ܸ� ������ ���� ��� ���ǥ ���� �� ���� �߻�: {e}")
            print("�Ϻ� ���� ����� ������� �� �ֽ��ϴ�.")


    print("\n" + "=" * 50)
    print("�м� �Ϸ�")
    print("=" * 50)