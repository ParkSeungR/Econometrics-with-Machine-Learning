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
    단변량 시계열 데이터에 대한 SARIMAX 모델 분석을 수행하는 함수.

    Args:
        data_df (pd.DataFrame): 분석할 시계열 데이터가 포함된 데이터프레임 (DatetimeIndex).
        variable_name (str): 데이터프레임 내에서 분석할 시계열 변수(열)의 이름.
        freq (str): 데이터의 주기 ('M' for monthly, 'Q' for quarterly).
        seasonal (bool): 계절성 요인을 모델에 포함시킬지 여부 (pmdarima.auto_arima의 seasonal 옵션).
        forecast_periods_validation (int, optional): 예측력 검증을 위한 데이터 수.
                                                     None이면 freq에 따라 기본값 설정 (월별: 24, 분기별: 8).
        forecast_periods_exante (int, optional): 사전 예측을 위한 기간 수.
                                                 None이면 freq에 따라 기본값 설정 (월별: 12, 분기별: 12).
    """
    
    # 선택된 변수만 추출하여 pd.Series로 변환
    if variable_name not in data_df.columns:
        raise ValueError(f"'{variable_name}' 변수가 데이터프레임에 존재하지 않습니다.")
    data = data_df[variable_name]
    
    # 인덱스가 DatetimeIndex인지 확인
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("데이터프레임의 인덱스는 DatetimeIndex여야 합니다.")

    print(f"--- 분석 변수: '{variable_name}' ---")
    print("\n--- 1. Seasonal Decomposition (계절 분해) ---")
    # 계절 분해 (Multiplicative or Additive 자동 검토)
    decomposition_model = 'multiplicative' if data.mean() < data.std() else 'additive' # 간단한 휴리스틱
    decomposition = seasonal_decompose(data, model=decomposition_model, period=12 if freq == 'M' else 4)
    print(f"자동으로 '{decomposition_model}' 모델이 선택되었습니다.")
    decomposition.plot().set_size_inches(12, 8)
    plt.suptitle(f'Seasonal Decomposition ({decomposition_model} Model) for {variable_name}', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    print("\n--- 2. Stationarity Tests (정상성 테스트) ---")
    # DF (Dickey-Fuller) Test (statsmodels.tsa.stattools.adfuller)
    df_test = adfuller(data)
    print(f"Dickey-Fuller Test (ADF):")
    print(f"  ADF Statistic: {df_test[0]:.4f}")
    print(f"  p-value: {df_test[1]:.4f}")
    print(f"  Critical Values: {df_test[4]}")
    if df_test[1] <= 0.05:
        print("  => 귀무가설 기각: 시계열은 정상성을 가질 수 있습니다 (트렌드나 단위근이 없을 수 있음).")
    else:
        print("  => 귀무가설 채택: 시계열은 비정상적일 수 있습니다 (단위근 존재 가능성).")

    # KPSS Test (statsmodels.tsa.stattools.kpss)
    kpss_test = kpss(data, regression='c', nlags='auto') # level stationarity
    print(f"\nKPSS Test (Level Stationarity):")
    print(f"  KPSS Statistic: {kpss_test[0]:.4f}")
    print(f"  p-value: {kpss_test[1]:.4f}")
    print(f"  Critical Values: {kpss_test[3]}")
    if kpss_test[1] <= 0.05:
        print("  => 귀무가설 기각: 시계열은 정상성이 없습니다 (트렌드 정상성일 수 있음).")
    else:
        print("  => 귀무가설 채택: 시계열은 정상성을 가질 수 있습니다 (수준 정상성).")

    # PP (Phillips-Perron) Test
    print("\n--- Phillips-Perron (PP) Test ---")
    for tt in ['n','c','ct']:
        print(f"\nPhillips-Perron Test (trend='{tt}', test_type='tau'):")
        try:
            pp = PhillipsPerron(data, trend=tt, test_type='tau')
            print(pp.summary().as_text())
        except Exception as e:
            print(f"  PP Test with trend='{tt}' failed: {e}")
            print("  이 오류는 주로 데이터의 길이가 짧거나, 특정 트렌드 모델에 부적합할 때 발생할 수 있습니다.")


    print("\n--- 3. Structural Change Check (구조 변화 체크) ---")
    print("CUSUM 테스트는 모델 잔차에 적용하는 것이 더 적절합니다. SARIMAX 모델 구축 후 잔차에 적용할 것입니다.")

    print("\n--- 4. SARIMAX Model Building (SARIMAX 모델 구축) ---")
    print(f"pmdarima.auto_arima를 사용하여 최적의 SARIMAX 모델을 탐색합니다. (seasonal={seasonal})")

    # pmdarima의 auto_arima를 사용하여 최적의 (p,d,q)(P,D,Q)s 파라미터 찾기
    model_auto_arima = auto_arima(
        data,
        start_p=1, start_q=1,
        max_p=5, max_q=5,
        m=12 if freq == 'M' else 4,
        seasonal=seasonal,
        d=None, D=None, # auto_arima가 자동으로 차분(d, D)을 결정하도록 None으로 설정
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("\n--- Auto ARIMA Model Summary ---")
    print(model_auto_arima.summary())

    # CUSUM Test (모델 잔차에 적용)
    print("\n--- 3. Structural Change Check (구조 변화 체크) - 모델 잔차에 적용 ---")
    print("모델의 잔차를 이용하여 CUSUM 테스트를 수행합니다.")
    try:
        fitted_model_for_cusum = SARIMAX(
            data,
            order=model_auto_arima.order,
            seasonal_order=model_auto_arima.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        cusum_residuals = fitted_model_for_cusum.resid
        
        # CUSUM OLS 잔차 테스트
        cusum_test_results = breaks_cusumolsresid(cusum_residuals)
        print("CUSUM Test (on SARIMAX Residuals):")
        print(f"  Test Statistic: {cusum_test_results[0]:.4f}")
        print(f"  p-value: {cusum_test_results[1]:.4f}")
        if cusum_test_results[1] <= 0.05:
            print("  => 귀무가설 기각: 잔차에 구조적 변화가 존재할 가능성이 있습니다.")
        else:
            print("  => 귀무가설 채택: 잔차에 통계적으로 유의미한 구조적 변화가 감지되지 않았습니다.")
        
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(cusum_residuals - cusum_residuals.mean()))
        plt.title(f'CUSUM of SARIMAX Residuals for {variable_name}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Sum of Residuals')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"CUSUM 테스트 실행 중 오류 발생: {e}")
        print("CUSUM 테스트는 특정 조건에서만 동작하며, 때로는 잔차 특성에 따라 오류가 발생할 수 있습니다.")


    print("\n--- 5. Model Validation (모형 예측력 평가) ---")
    if forecast_periods_validation is None:
        forecast_periods_validation = 24 if freq == 'M' else 8

    # 학습 및 검증 데이터 분할
    train = data[:-forecast_periods_validation]
    test = data[-forecast_periods_validation:]
    print(f"데이터를 {len(train)} (학습)과 {len(test)} (검증)으로 분할했습니다.")

    # 학습 데이터로 SARIMAX 모델 재적합
    print("학습 데이터로 SARIMAX 모델을 재적합합니다.")
    try:
        model_trained = SARIMAX(
            train,
            order=model_auto_arima.order,
            seasonal_order=model_auto_arima.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    except Exception as e:
        print(f"모델 재적합 중 오류 발생: {e}")
        print("모델 재적합이 실패했습니다. 데이터나 파라미터 설정을 확인해주세요.")
        return

    # 예측 (out-of-sample prediction)
    start_index = len(train)
    end_index = len(data) - 1
    
    print(f"검증 기간 ({test.index[0]} ~ {test.index[-1]})에 대한 예측을 수행합니다.")
    try:
        predictions = model_trained.get_prediction(start=start_index, end=end_index, dynamic=False)
        pred_mean = predictions.predicted_mean
        pred_ci = predictions.conf_int()
    except Exception as e:
        print(f"예측 수행 중 오류 발생: {e}")
        print("예측이 실패했습니다. 모델이나 데이터 설정을 확인해주세요.")
        return

    # 예측 결과 시각화
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

    # 예측력 평가 지표 (MSE, RMSE)
    mse = mean_squared_error(test, pred_mean)
    rmse = np.sqrt(mse)
    print(f"\n--- Model Performance on Validation Set for {variable_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    print("\n--- 6. Ex-ante Forecasting (사전 예측) ---")
    if forecast_periods_exante is None:
        forecast_periods_exante = 12 if freq == 'M' else 12

    # 미래 예측 (ex-ante)
    print(f"미래 {forecast_periods_exante} 기간에 대한 사전 예측을 수행합니다.")
    try:
        # 전체 데이터로 모델 재적합 (가장 최신 정보를 반영하기 위해)
        final_model = SARIMAX(
            data,
            order=model_auto_arima.order,
            seasonal_order=model_auto_arima.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        forecast_start_index = len(data)
        forecast_end_index = len(data) + forecast_periods_exante - 1
        
        # 미래 날짜 인덱스 생성
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

        # forecast_mean에 미래 날짜 인덱스 할당
        forecast_mean.index = future_dates
        forecast_ci.index = future_dates

    except Exception as e:
        print(f"사전 예측 수행 중 오류 발생: {e}")
        print("사전 예측이 실패했습니다. 모델이나 데이터 설정을 확인해주세요.")
        return

    print("\n--- Ex-ante Forecast Values ---")
    print(pd.DataFrame({'Forecast': forecast_mean,
                        'Lower CI': forecast_ci.iloc[:, 0],
                        'Upper CI': forecast_ci.iloc[:, 1]}))

    # 사전 예측 시각화
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