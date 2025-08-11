import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from arch.unitroot import PhillipsPerron # PhillipsPerron import
import warnings

warnings.filterwarnings("ignore")

def Run_SARIMAX_analysis(
    data_df: pd.DataFrame,
    variable_name: str,
    freq: str,
    seasonal: bool = True,
    forecast_periods_validation: int = None,
    forecast_periods_exante: int = None
):
    """
    �ܺ��� �ð迭 �����Ϳ� ���� SARIMAX �� �м��� �����ϴ� �Լ�.

    Args:
        data_df (pd.DataFrame): �м��� �ð迭 �����Ͱ� ���Ե� ������������ (DatetimeIndex).
        variable_name (str): ������������ ������ �м��� �ð迭 ����(��)�� �̸�.
        freq (str): �������� �ֱ� ('M' for monthly, 'Q' for quarterly).
        seasonal (bool): ������ ������ �𵨿� ���Խ�ų�� ���� (pmdarima.auto_arima�� seasonal �ɼ�).
        forecast_periods_validation (int, optional): ������ ������ ���� ������ ��.
                                                     None�̸� freq�� ���� �⺻�� ���� (����: 24, �б⺰: 8).
        forecast_periods_exante (int, optional): ���� ������ ���� �Ⱓ ��.
                                                 None�̸� freq�� ���� �⺻�� ���� (����: 12, �б⺰: 12).
    """
    
    # ���õ� ������ �����Ͽ� pd.Series�� ��ȯ
    if variable_name not in data_df.columns:
        raise ValueError(f"'{variable_name}' ������ �����������ӿ� �������� �ʽ��ϴ�.")
    data = data_df[variable_name]
    
    # �ε����� DatetimeIndex���� Ȯ��
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("�������������� �ε����� DatetimeIndex���� �մϴ�.")

    print(f"--- �м� ����: '{variable_name}' ---")
    print("\n--- 1. Seasonal Decomposition (���� ����) ---")
    # ���� ���� (Multiplicative or Additive �ڵ� ����)
    decomposition_model = 'multiplicative' if data.mean() < data.std() else 'additive' # ������ �޸���ƽ
    decomposition = seasonal_decompose(data, model=decomposition_model, period=12 if freq == 'M' else 4)
    print(f"�ڵ����� '{decomposition_model}' ���� ���õǾ����ϴ�.")
    decomposition.plot().set_size_inches(12, 8)
    plt.suptitle(f'Seasonal Decomposition ({decomposition_model} Model) for {variable_name}', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    print("\n--- 2. Stationarity Tests (���� �׽�Ʈ) ---")
    # DF (Dickey-Fuller) Test (statsmodels.tsa.stattools.adfuller)
    df_test = adfuller(data)
    print(f"Dickey-Fuller Test (ADF):")
    print(f"  ADF Statistic: {df_test[0]:.4f}")
    print(f"  p-value: {df_test[1]:.4f}")
    print(f"  Critical Values: {df_test[4]}")
    if df_test[1] <= 0.05:
        print("  => �͹����� �Ⱒ: �ð迭�� ������ ���� �� �ֽ��ϴ� (Ʈ���峪 �������� ���� �� ����).")
    else:
        print("  => �͹����� ä��: �ð迭�� ���������� �� �ֽ��ϴ� (������ ���� ���ɼ�).")

    # KPSS Test (statsmodels.tsa.stattools.kpss)
    kpss_test = kpss(data, regression='c', nlags='auto') # level stationarity
    print(f"\nKPSS Test (Level Stationarity):")
    print(f"  KPSS Statistic: {kpss_test[0]:.4f}")
    print(f"  p-value: {kpss_test[1]:.4f}")
    print(f"  Critical Values: {kpss_test[3]}")
    if kpss_test[1] <= 0.05:
        print("  => �͹����� �Ⱒ: �ð迭�� ������ �����ϴ� (Ʈ���� ������ �� ����).")
    else:
        print("  => �͹����� ä��: �ð迭�� ������ ���� �� �ֽ��ϴ� (���� ����).")

    # PP (Phillips-Perron) Test
    print("\n--- Phillips-Perron (PP) Test ---")
    for tt in ['n','c','ct']:
        print(f"\nPhillips-Perron Test (trend='{tt}', test_type='tau'):")
        try:
            pp = PhillipsPerron(data, trend=tt, test_type='tau')
            print(pp.summary().as_text())
        except Exception as e:
            print(f"  PP Test with trend='{tt}' failed: {e}")
            print("  �� ������ �ַ� �������� ���̰� ª�ų�, Ư�� Ʈ���� �𵨿� �������� �� �߻��� �� �ֽ��ϴ�.")


    print("\n--- 3. Structural Change Check (���� ��ȭ üũ) ---")
    print("CUSUM �׽�Ʈ�� �� ������ �����ϴ� ���� �� �����մϴ�. SARIMAX �� ���� �� ������ ������ ���Դϴ�.")

    print("\n--- 4. SARIMAX Model Building (SARIMAX �� ����) ---")
    print(f"pmdarima.auto_arima�� ����Ͽ� ������ SARIMAX ���� Ž���մϴ�. (seasonal={seasonal})")

    # pmdarima�� auto_arima�� ����Ͽ� ������ (p,d,q)(P,D,Q)s �Ķ���� ã��
    model_auto_arima = auto_arima(
        data,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        m=12 if freq == 'M' else 4,
        seasonal=seasonal,
        d=None, D=None, # auto_arima�� �ڵ����� ����(d, D)�� �����ϵ��� None���� ����
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("\n--- Auto ARIMA Model Summary ---")
    print(model_auto_arima.summary())

    # CUSUM Test (�� ������ ����)
    print("\n--- 3. Structural Change Check (���� ��ȭ üũ) - �� ������ ���� ---")
    print("���� ������ �̿��Ͽ� CUSUM �׽�Ʈ�� �����մϴ�.")
    try:
        fitted_model_for_cusum = SARIMAX(
            data,
            order=model_auto_arima.order,
            seasonal_order=model_auto_arima.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        cusum_residuals = fitted_model_for_cusum.resid
        
        # CUSUM OLS ���� �׽�Ʈ
        cusum_test_results = breaks_cusumolsresid(cusum_residuals)
        print("CUSUM Test (on SARIMAX Residuals):")
        print(f"  Test Statistic: {cusum_test_results[0]:.4f}")
        print(f"  p-value: {cusum_test_results[1]:.4f}")
        if cusum_test_results[1] <= 0.05:
            print("  => �͹����� �Ⱒ: ������ ������ ��ȭ�� ������ ���ɼ��� �ֽ��ϴ�.")
        else:
            print("  => �͹����� ä��: ������ ��������� ���ǹ��� ������ ��ȭ�� �������� �ʾҽ��ϴ�.")
        
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(cusum_residuals - cusum_residuals.mean()))
        plt.title(f'CUSUM of SARIMAX Residuals for {variable_name}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Sum of Residuals')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"CUSUM �׽�Ʈ ���� �� ���� �߻�: {e}")
        print("CUSUM �׽�Ʈ�� Ư�� ���ǿ����� �����ϸ�, ���δ� ���� Ư���� ���� ������ �߻��� �� �ֽ��ϴ�.")


    print("\n--- 5. Model Validation (���� ������ ��) ---")
    if forecast_periods_validation is None:
        forecast_periods_validation = 24 if freq == 'M' else 8

    # �н� �� ���� ������ ����
    train = data[:-forecast_periods_validation]
    test = data[-forecast_periods_validation:]
    print(f"�����͸� {len(train)} (�н�)�� {len(test)} (����)���� �����߽��ϴ�.")

    # �н� �����ͷ� SARIMAX �� ������
    print("�н� �����ͷ� SARIMAX ���� �������մϴ�.")
    try:
        model_trained = SARIMAX(
            train,
            order=model_auto_arima.order,
            seasonal_order=model_auto_arima.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    except Exception as e:
        print(f"�� ������ �� ���� �߻�: {e}")
        print("�� �������� �����߽��ϴ�. �����ͳ� �Ķ���� ������ Ȯ�����ּ���.")
        return

    # ���� (out-of-sample prediction)
    start_index = len(train)
    end_index = len(data) - 1
    
    print(f"���� �Ⱓ ({test.index[0]} ~ {test.index[-1]})�� ���� ������ �����մϴ�.")
    try:
        predictions = model_trained.get_prediction(start=start_index, end=end_index, dynamic=False)
        pred_mean = predictions.predicted_mean
        pred_ci = predictions.conf_int()
    except Exception as e:
        print(f"���� ���� �� ���� �߻�: {e}")
        print("������ �����߽��ϴ�. ���̳� ������ ������ Ȯ�����ּ���.")
        return

    # ���� ��� �ð�ȭ
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Train Data')
    plt.plot(test.index, test, label='Actual Test Data')
    plt.plot(pred_mean.index, pred_mean, label='Predicted Test Data', color='red')
    plt.fill_between(pred_ci.index,
                     pred_ci.iloc[:, 0],
                     pred_ci.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
    plt.title(f'SARIMAX Model Validation for {variable_name}: Actual vs. Predicted')
    plt.xlabel('Date')
    plt.ylabel(variable_name)
    plt.legend()
    plt.grid(True)
    plt.show()

    # ������ �� ��ǥ (MSE, RMSE)
    mse = mean_squared_error(test, pred_mean)
    rmse = np.sqrt(mse)
    print(f"\n--- Model Performance on Validation Set for {variable_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    print("\n--- 6. Ex-ante Forecasting (���� ����) ---")
    if forecast_periods_exante is None:
        forecast_periods_exante = 12 if freq == 'M' else 12

    # �̷� ���� (ex-ante)
    print(f"�̷� {forecast_periods_exante} �Ⱓ�� ���� ���� ������ �����մϴ�.")
    try:
        # ��ü �����ͷ� �� ������ (���� �ֽ� ������ �ݿ��ϱ� ����)
        final_model = SARIMAX(
            data,
            order=model_auto_arima.order,
            seasonal_order=model_auto_arima.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        forecast_start_index = len(data)
        forecast_end_index = len(data) + forecast_periods_exante - 1
        
        # �̷� ��¥ �ε��� ����
        if freq == 'M':
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods_exante, freq='M')
        elif freq == 'Q':
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), periods=forecast_periods_exante, freq='Q')
        else:
            raise ValueError("Unsupported frequency. Use 'M' for monthly or 'Q' for quarterly.")

        forecast_results = final_model.get_prediction(start=forecast_start_index, end=forecast_end_index)
        forecast_mean = forecast_results.predicted_mean
        forecast_ci = forecast_results.conf_int()

        # forecast_mean�� �̷� ��¥ �ε��� �Ҵ�
        forecast_mean.index = future_dates
        forecast_ci.index = future_dates

    except Exception as e:
        print(f"���� ���� ���� �� ���� �߻�: {e}")
        print("���� ������ �����߽��ϴ�. ���̳� ������ ������ Ȯ�����ּ���.")
        return

    print("\n--- Ex-ante Forecast Values ---")
    print(pd.DataFrame({'Forecast': forecast_mean,
                        'Lower CI': forecast_ci.iloc[:, 0],
                        'Upper CI': forecast_ci.iloc[:, 1]}))

    # ���� ���� �ð�ȭ
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data, label='Historical Data')
    plt.plot(forecast_mean.index, forecast_mean, label='Ex-ante Forecast', color='green')
    plt.fill_between(forecast_ci.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1], color='lightgreen', alpha=0.3, label='Confidence Interval')
    plt.title(f'SARIMAX Ex-ante Forecast for {variable_name} ({forecast_periods_exante} Periods)')
    plt.xlabel('Date')
    plt.ylabel(variable_name)
    plt.legend()
    plt.grid(True)
    plt.show()