import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import warnings

# VARResultsWrapper Ŭ������ ���� ����Ʈ
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

# ��� �޽��� ���� (����� ����� ����)
warnings.filterwarnings("ignore")

def Run_VAR_analysis(
    data: pd.DataFrame,
    endog_vars: list,
    exog_vars: list = None,
    # �Ʒ� �Ķ���͵��� ��� ����Ʈ ���� �����ϴ�.
    maxlags_order_selection: int = 8,
    ic_criterion: str = 'aic', # ���� ���� ����: 'aic', 'bic', 'fpe', 'hqic'
    diff_type: str = 'log_diff', # ����ȭ ���: 'none', 'diff', 'pct_change', 'log_diff'
    pct_change_periods: int = 4, # 'pct_change' ��� �� ���� �� �Ⱓ�� ������ (��: �б� �������� ���� ���� ��� 4)
    start_date: str = None, # �����ͼ��� Ư�� ��¥���� �����̽��� ��� (��: '1970-01-31')
    fixed_lag_order: int = None, # �������� VAR ������ ���� (None�̸� �ڵ� ����)
    nlags_irf: int = 10,
    nsteps_fevd: int = 20,
    nsteps_forecast: int = 20
):
    """
    VAR (Vector Autoregression) ������ ��ü �м� ������ �����ϴ� �Լ�.
    �����ͼ°� ����/�ܻ� ������ �ʼ��� �Է¹ް�, ������ �ɼ��� ����Ʈ ���� �����ϴ�.

    Args:
        data (pd.DataFrame): �ð迭 ������������. �ε����� datetime �����̾�� ��.
        endog_vars (list): VAR ������ ���Ե� ���� ����(���� ����) ����Ʈ.
        exog_vars (list, optional): VARX ������ ���Ե� �ܻ� ���� ����Ʈ. Defaults to None.
        maxlags_order_selection (int, optional): VAR ���� ������ ���� �ִ� ����. Defaults to 8.
        ic_criterion (str, optional): VAR ���� ���� ���� ('aic', 'bic', 'fpe', 'hqic'). Defaults to 'aic'.
        diff_type (str, optional): �ð迭 ����ȭ�� ���� ��ȯ ���.
                                    'none': ��ȯ ���� (���迭 ���)
                                    'diff': 1�� ����
                                    'pct_change': �ۼ�Ʈ ��ȭ
                                    'log_diff': �α� ���� (������)
                                    Defaults to 'log_diff'.
        pct_change_periods (int, optional): 'pct_change' ��� �� ���� �� �Ⱓ�� ������. Defaults to 4 (�б� ������ ���� ���� ���).
        start_date (str, optional): ��ȯ�� �����ͼ��� ���� ��¥�� �����մϴ� (��: '1970-01-31'). Defaults to None (�����̽� ����).
        fixed_lag_order (int, optional): ����ڰ� �������� VAR ������ ������ ����. None�̸� ic_criterion�� ���� �ڵ� ����. Defaults to None.
        nlags_irf (int, optional): ��ݹ��� �Լ��� �׸� ������ ��. Defaults to 10.
        nsteps_fevd (int, optional): ���� ���� �л� ���ظ� ���� �ܰ� ��. Defaults to 20.
        nsteps_forecast (int, optional): ������ ���� �ܰ� ��. Defaults to 20.

    Returns:
        results (statsmodels.tsa.vector_ar.var_model.VARResultsWrapper): VAR ���� ���� ��� ��ü.
    """

    print("--- VAR ���� �м� ���� ---")
    print(f"���� ����: {endog_vars}")
    if exog_vars:
        print(f"�ܻ� ����: {exog_vars}")
    print("-" * 30)

    # 0. ������ �غ� �� ��ó��
    required_cols = endog_vars + (exog_vars if exog_vars else [])
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"�����������ӿ� ���� �������� �����ϴ�: {missing_cols}")

    df = data[required_cols].copy()

    # 1. �ð迭�� ������ ���� �� ����ȭ
    print("\n1. �ð迭�� ������ ���� �� ����ȭ")
    print("���迭 �ð�ȭ (������ ���� Ȯ��):")
    df[endog_vars].plot(title="Original Series (Endogenous Variables)", figsize=(12, 6))
    plt.tight_layout()
    plt.show()
    print("���迭�� ���� �����ϴ� ������ �Ҿ����� �ð迭ó�� ���� �� �ֽ��ϴ�.")

    df_transformed = df.copy()
    if diff_type == 'diff':
        df_transformed = df.diff().dropna()
        print(f"-> 1�� ���� ({diff_type})�� ���� �ð迭 ����ȭ �õ�.")
    elif diff_type == 'pct_change':
        df_transformed = 100 * df.pct_change(periods=pct_change_periods).dropna()
        print(f"-> �ۼ�Ʈ ��ȭ ({diff_type}, periods={pct_change_periods})�� ���� �ð迭 ����ȭ �õ�.")
        print(f"   (����: periods�� �б�/���� �� ������ �ֱ�� ��ġ�ؾ� �մϴ�. ��: �б� �����ʹ� 4, ���� �����ʹ� 12)")
    elif diff_type == 'log_diff':
        df_transformed = np.log(df).diff().dropna() * 100
        print(f"-> �α� ���� ({diff_type}, ������ �ð迭)�� ���� �ð迭 ����ȭ �õ�.")
    else:
        print("-> �ð迭 ��ȯ�� �������� �ʰ� ���迭�� �����մϴ� (diff_type='none').")

    # �߰��� �κ�: Ư�� ��¥���� ������ �����̽�
    if start_date is not None:
        try:
            # pd.to_datetime���� ��¥ ���ڿ��� Datetime ��ü�� ��ȯ�Ͽ� ���
            df_transformed = df_transformed.loc[pd.to_datetime(start_date):]
            print(f"-> �����ͼ��� '{start_date}'���� �����ϵ��� �����̽��߽��ϴ�.")
        except KeyError:
            print(f"���: ������ ���� ��¥ '{start_date}' ������ �����Ͱ� �����ϴ�. �����̽��� ������� �ʾ��� �� �ֽ��ϴ�.")
        except Exception as e:
            print(f"���: start_date �����̽� �� ���� �߻�: {e}. �����̽��� �ǳʶݴϴ�.")


    # ��ȯ �� �����Ͱ� �ʹ� ª������ ���� �߻� ���ɼ� ����
    min_required_len = max(maxlags_order_selection, 1) + len(endog_vars) + 1 
    if len(df_transformed) < min_required_len:
        raise ValueError(f"������ ��ȯ �� ����ġ ���� �ʹ� �����ϴ� ({len(df_transformed)}��). VAR �� ������ ���� �ּ� ����ġ ���� �� {min_required_len}�� �Դϴ�. `diff_type`�� �����ϰų� �� �� �ð迭 �����͸� ����ϼ���.")

    print("\n��ȯ�� �ð迭 �ð�ȭ:")
    df_transformed[endog_vars].plot(title=f"Transformed Series ({diff_type})", figsize=(12, 6))
    plt.tight_layout()
    plt.show()
    print("���������� ���̴� ������ �ð迭�� Ư�� ������ ���ؿ��� �Ʒ����� �����ϴ� ������ �ð迭�� ����� ���Դϴ�.")
    print("\n--- ��ȯ�� ������ (Transformed Data) �̸����� ---")
    print(df_transformed.head())
    print("-" * 30)


    print("\nADF (Augmented Dickey-Fuller) ������ ����:")
    print("�͹����� (H0): �ð迭�� �������� �����Ѵ� (�Ҿ����ϴ�).")
    print("�븳���� (H1): �ð迭�� �������� �������� �ʴ´� (�����ϴ�).")
    print("������跮�� �Ӱ�ġ���� �۰� (�� ����), p-���� ���Ǽ���(��: 0.05)���� ������ H0�� �Ⱒ�ϰ� �������̶�� �Ǵ��մϴ�.")

    for col in endog_vars:
        print(f"\n--- {col} ---")
        adf_results_c = adfuller(df_transformed[col], regression='c')
        print(f"�����(drift) ���� ���� (regression='c'):")
        print(f"  ADF Statistic: {adf_results_c[0]:.4f}")
        print(f"  p-value: {adf_results_c[1]:.4f}")
        print(f"  Critical Values: {adf_results_c[4]}")
        if adf_results_c[1] <= 0.05:
            print("  => 5% ���Ǽ��ؿ��� �������� ���ٰ� �Ǵ� (������).")
        else:
            print("  => 5% ���Ǽ��ؿ��� �������� �ִٰ� �Ǵ� (�Ҿ���).")


    # 2. VAR ������ ���� ����
    print("\n\n2. VAR ������ ���� ����")
    if fixed_lag_order is not None:
        optimal_lag_order = fixed_lag_order
        print(f"����ڰ� ���� ���� {optimal_lag_order}�� �����߽��ϴ�.")
    else:
        model_for_order = VAR(df_transformed[endog_vars])
        print(f"�ִ� ���� {maxlags_order_selection}���� {ic_criterion.upper()} ���� ���� ����:")

        adjusted_maxlags = maxlags_order_selection
        if len(df_transformed) <= maxlags_order_selection:
            adjusted_maxlags = max(1, len(df_transformed) - 1)
            print(f"���: ������ ���̰� ª�� `maxlags_order_selection`({maxlags_order_selection})�� �ʹ� Ů�ϴ�. `maxlags`�� {adjusted_maxlags}�� �����մϴ�.")
        
        selected_order = model_for_order.select_order(maxlags=adjusted_maxlags)
            
        print(selected_order.summary())
        optimal_lag_order_by_ic = selected_order.selected_orders[ic_criterion]
        print(f"\n'{ic_criterion.upper()}' ���� ���� ����: {optimal_lag_order_by_ic}")
        
        optimal_lag_order = optimal_lag_order_by_ic

        if optimal_lag_order == 0:
            print("\n!!! ���: ���õ� ���� ������ 0�Դϴ�. �� ��� Granger �ΰ���, IRF, FEVD � ������ �߻��� �� �ֽ��ϴ�.")
            print("   `fixed_lag_order` �Ķ���͸� ����Ͽ� ������ �������� �����غ����� (��: `fixed_lag_order=1` �Ǵ� `2`).")
            print("   �Ǵ� `ic_criterion`�� 'bic' ������ �����Ͽ� �ٽ� �õ��غ�����.")
        else:
            print("���� ��跮�� ��� ������ ������ ������ ������ �������� �����Ƿ� �߰����� ���ܰ����� ���� ���� ������ �����ؾ� �մϴ�.")


    # 3. ���� ����
    print("\n\n3. VAR ���� ����")
    print(f"���õ� ���� ���� {optimal_lag_order}�� VAR ������ �����մϴ�.")
    if exog_vars:
        try:
            model = VARMAX(endog=df_transformed[endog_vars], order=(optimal_lag_order, 0), exog=df_transformed[exog_vars])
            results = model.fit(maxiter=1000, disp=False)
            print("VARX ������ �����Ǿ����ϴ�.")
        except Exception as e:
            print(f"VARX ���� ���� �� ���� �߻�: {e}")
            print("VARX ���� ��� VAR �������� ������ �õ��մϴ�.")
            model = VAR(df_transformed[endog_vars])
            results = model.fit(optimal_lag_order)
            print("VAR ������ �����Ǿ����ϴ�.")
    else:
        model = VAR(df_transformed[endog_vars])
        results = model.fit(optimal_lag_order)
        print("VAR ������ �����Ǿ����ϴ�.")

    actual_lag_order = results.k_ar if hasattr(results, 'k_ar') else optimal_lag_order
    print(f"**���� ������ ������ ���� (results.k_ar): {actual_lag_order}**")
    print("\n--- ���� ���� ��� ��� (Summary) ---")
    print(results.summary())
    print("\n���� �����ĺ� �Ķ���� ����ġ�� ���Ǽ� ������ ���� �򰡿��� �߿�õ��� �ʽ��ϴ�.")
    print("��� �׷��� �ΰ� ����, ��ݹ��� �Լ�, ���� ������ ���ؿ� ���� ��跮���� ���� ���˴ϴ�.")


    # 4. ����
    print("\n\n4. ���� ����")

    # 4-1. ������ �ڱ��� ���� ����
    print("\n4-1. ������ �ڱ��� ���� ���� (ACF Plots):")
    if isinstance(results, VARResultsWrapper):
        results.plot_acorr()
        plt.suptitle("ACF plots for residuals", y=1.02)
        plt.tight_layout()
        plt.show()
        print("ACF �÷Կ��� �ŷڱ���(����)�� ����� ���밡 �ִٸ� �ڱ��� ������ ���� �� �ֽ��ϴ�.")
        print("�������� �ڱ��� ������ �ִ��� ī������ ��跮�� ���Ǽ������� �Ǵ��� �� �ֽ��ϴ�.")
    else:
        print("VARMAX ��� ��ü������ plot_acorr()�� ���� �������� �ʽ��ϴ�. �������� ���� ACF�� Ȯ���ؾ� �մϴ�.")
        residuals = results.resid
        fig, axes = plt.subplots(len(endog_vars), 1, figsize=(10, 3 * len(endog_vars)))
        if len(endog_vars) == 1:
            axes = [axes]
        for i, var in enumerate(endog_vars):
            sm.graphics.tsa.plot_acf(residuals[var], lags=nlags_irf, ax=axes[i], title=f'Residuals ACF for {var}')
        plt.tight_layout()
        plt.show()


    # 4-2. ������ ���Լ� ���� (Jarque-Bera test)
    print("\n4-2. ������ ���Լ� ���� (Jarque-Bera test):")
    print("�͹����� (H0): �������� ���Ժ����� ������.")
    print("�븳���� (H1): �������� ���Ժ����� ������ �ʴ´�.")
    if isinstance(results, VARResultsWrapper):
        normality_test = results.test_normality()
        print(normality_test.summary())
        if normality_test.pvalue <= 0.05:
            print("=> 5% ���Ǽ��ؿ��� �͹������� �Ⱒ�մϴ�. �������� ���Ժ����� ������ ���� �� �ֽ��ϴ�.")
        else:
            print("=> 5% ���Ǽ��ؿ��� �͹������� �Ⱒ���� ���մϴ�. �������� ���Ժ����� �����ٰ� �� �� �ֽ��ϴ�.")
    else:
        print("VARMAX ��� ��ü������ test_normality()�� ���� �������� �ʽ��ϴ�. �������� ���� ���Լ��� �����ؾ� �մϴ�.")
        for col in endog_vars:
            print(f"\n--- {col} Residuals Normality Test ---")
            jb_test = sm.stats.stattools.jarque_bera(results.resid[col])
            print(f"  Jarque-Bera Statistic: {jb_test[0]:.4f}")
            print(f"  p-value: {jb_test[1]:.4f}")
            print(f"  Skewness: {jb_test[2]:.4f}")
            print(f"  Kurtosis: {jb_test[3]:.4f}")
            if jb_test[1] <= 0.05:
                print("  => 5% ���Ǽ��ؿ��� ���Ժ��� ������ �Ⱒ�մϴ�.")
            else:
                print("  => 5% ���Ǽ��ؿ��� ���Ժ��� ������ �Ⱒ���� ���մϴ�.")


    # 4-3. Granger �ΰ��� ����
    print("\n\n4-3. Granger �ΰ��� ����:")
    print("Granger �ΰ��� ������ ������ ���� ���� �ΰ����� ������ �м��մϴ�.")
    if optimal_lag_order == 0:
        print("  !!! ���: VAR ������ ������ 0�̹Ƿ� Granger �ΰ��� ������ ������ �� �����ϴ�.")
        print("  ������ 0�� ���������� ���� ���� �������� �ΰ� ���踦 �м��� �� �����ϴ�.")
    else:
        for target_var in endog_vars:
            other_vars = [v for v in endog_vars if v != target_var]
            if other_vars:
                try:
                    if isinstance(results, VARResultsWrapper):
                        causality_test = results.test_causality(target_var, other_vars, kind='f')
                        print(f"\n{other_vars}�� {target_var}�� Granger-�ΰ��ϴ��� ����:")
                        print(causality_test.summary())
                        if causality_test.pvalue <= 0.05:
                            print(f"  => 5% ���Ǽ��ؿ��� �͹������� �Ⱒ�մϴ�. {other_vars}�� {target_var}�� Granger-�ΰ��Ѵٰ� �� �� �ֽ��ϴ�.")
                        else:
                            print(f"  => 5% ���Ǽ��ؿ��� �͹������� �Ⱒ���� ���մϴ�. {other_vars}�� {target_var}�� Granger-�ΰ����� �ʴ´ٰ� �� �� �ֽ��ϴ�.")
                    else:
                        print(f"VARMAX ��� ��ü������ test_causality()�� ���� �������� �ʽ��ϴ�. �������� �׷��� �ΰ��� ������ �����ؾ� �մϴ�.")
                except Exception as e:
                    print(f"  {target_var}�� ���� Granger �ΰ��� ���� �� ���� �߻�: {e}")
            else:
                print(f"{target_var}�� �ٸ� ���� ������ �����Ƿ� Granger �ΰ��� ������ �ǳʶݴϴ�.")

    # 5. ��� ���� �Լ� (Impulse Response Function, IRF)
    print("\n\n5. ��� ���� �Լ� (IRF)")
    if optimal_lag_order == 0:
        print("  !!! ���: VAR ������ ������ 0�̹Ƿ� ��� ���� �Լ��� ����� �� �����ϴ�.")
        print("  ������ 0�� ���������� �������� ��� ȿ���� �м��� �� �����ϴ�.")
    else:
        print("��ݹ��� �Լ��� VAR �������� Ư�� �������� �����׿� ���� ����� �ٸ� ���� �������� �̷����� ��ġ�� ������ ���캾�ϴ�.")
        print("�������� 1 ǥ������ ��ݿ� ���� ���� ������ ����մϴ�.")

        try:
            irf = results.irf(nlags_irf)
            
            print("\n--- ������ȭ�� ��� ���� �Լ� (Non-Orthogonalized IRF) ---")
            irf.plot_cum_effects(orth=False)
            plt.suptitle("Cumulative responses (Non-Orthogonalized)", y=1.02)
            plt.tight_layout()
            plt.show()

            print("\n--- ����ȭ�� ��� ���� �Լ� (Orthogonalized IRF) ---")
            print("����ȭ�� ��ݹ��� �Լ��� ���� ������ �ΰ��մϴ�. �ͷ���Ű ���ظ� ���� �������ϴ�.")
            irf.plot(orth=True)
            plt.suptitle("Impulse responses (Orthogonalized)", y=1.02)
            plt.tight_layout()
            plt.show()
            print("��� ���� �Լ��� ���� ���� ���� ���� �ΰ����� �м��̳� ��å ȿ�� �м��� �� �� �ֽ��ϴ�.")

        except Exception as e:
            print(f"��� ���� �Լ� ��� �Ǵ� �÷��� �� ���� �߻�: {e}")
            print("VARMAX ������ ��� impulse_responses �޼��� ������ �ٸ� �� �ֽ��ϴ�.")
            if exog_vars:
                print("VARMAX ������ ��� impulse_responses�� VARMAX ��ü���� ���� ȣ���ؾ� �մϴ�.")
                print("�ڵ�ȭ�� ���� ù ��° ������ ���� ��ݸ� ���÷� �÷��մϴ�.")
                try:
                    impulse_array = np.zeros(len(endog_vars))
                    impulse_array[0] = 1
                    ax = results.impulse_responses(nlags_irf, orthogonalized=True, impulse=impulse_array).plot(figsize=(12,4))
                    ax.set(xlabel='t', title=f'Responses to a shock to `{endog_vars[0]}` (Orthogonalized)');
                    plt.tight_layout()
                    plt.show()
                except Exception as e_varmax_irf:
                    print(f"VARMAX ��ݹ��� �Լ� �÷��� �� ���� �߻�: {e_varmax_irf}")


    # 6. ���� ���� �л��� ���� (Forecast Error Variance Decomposition, FEVD)
    print("\n\n6. ���� ���� �л��� ���� (FEVD)")
    if optimal_lag_order == 0:
        print("  !!! ���: VAR ������ ������ 0�̹Ƿ� ���� ���� �л� ���ظ� ����� �� �����ϴ�.")
        print("  ������ 0�� ���������� ���� ������ �������� ���ظ� �� �� �����ϴ�.")
    else:
        print("FEVD�� �ð��� ��ȭ�� ���� ���Ӻ����� ���̰� ���� �ڱ� �ڽ��� �л�� �ٸ� ������ �л꿡 ���� ����Ǵ� ������ �����ݴϴ�.")
        try:
            fevd = results.fevd(nsteps_fevd)
            fevd.plot()
            plt.suptitle("Forecast Error Variance Decomposition (FEVD)", y=1.02)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"���� ���� �л� ���� ��� �Ǵ� �÷��� �� ���� �߻�: {e}")
            print("VARMAX ������ ��� fevd�� ���� �������� �ʰų� ������ �ٸ� �� �ֽ��ϴ�.")
            if exog_vars:
                print("VARMAX ������ fevd �޼��带 ���� �������� �ʽ��ϴ�. ���� ����� �ʿ��մϴ�.")


    # 7. ���� (Forecast)
    print("\n\n7. ����")
    if optimal_lag_order == 0:
        print("  !!! ���: VAR ������ ������ 0�̹Ƿ� ������ ����� �����ϱ� ��ƽ��ϴ�.")
        print("  ������ 0�� ������ ���� ������ �̷��� ������ ���� �ʴ´ٰ� �����մϴ�. �ܼ� ��� ������ ������ �� �ֽ��ϴ�.")
    else:
        print("VAR ������ �̿��� ���������� ���� ������ ���� ������ �� �� �ֽ��ϴ�.")
        try:
            lag_order_for_forecast = results.k_ar if hasattr(results, 'k_ar') else optimal_lag_order

            if exog_vars:
                print("VARX/VARMA ������ predict/forecast �޼��� ������ VAR�� �ٸ� �� �ֽ��ϴ�.")
                print("plot_forecast�� VARX/VARMA������ ��� �����մϴ�.")
                results.plot_forecast(steps=nsteps_forecast)
                plt.suptitle(f"Forecast for {nsteps_forecast} steps (VARX)", y=1.02)
                plt.tight_layout()
                plt.show()
            else:
                print("\n--- ���� �ð�ȭ ---")
                results.plot_forecast(steps=nsteps_forecast)
                plt.suptitle(f"Forecast for {nsteps_forecast} steps (VAR)", y=1.02)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"���� ��� �Ǵ� �÷��� �� ���� �߻�: {e}")

    print("\n--- VAR ���� �м� �Ϸ� ---")
    return results