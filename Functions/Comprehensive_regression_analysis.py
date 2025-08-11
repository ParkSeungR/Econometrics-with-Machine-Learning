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
    단일 명령어로 회귀 분석에 필요한 모든 결과를 출력하는 함수.
    횡단면 또는 시계열 데이터에 따라 이분산/자기상관 검정 및 해결 추정법을 적용합니다.

    Args:
        data (pd.DataFrame): 분석할 전체 데이터.
        y_var (str): 종속 변수명.
        X_vars (list): 독립 변수명 리스트.
        data_type (str, optional): 데이터의 유형을 명시적으로 지정합니다.
                                   'time_series' 또는 'cross_sectional'.
                                   None인 경우 인덱스 유형을 기반으로 자동 감지합니다.
                                   기본값은 None.
    """

    y = data[y_var]
    X = data[X_vars]

    print("=" * 50)
    print("종합 회귀 분석 보고서")
    print("=" * 50)

    # 0. 데이터 유형 파악 (시계열 여부)
    if data_type is None:
        is_time_series = isinstance(data.index, pd.DatetimeIndex) and data.index.is_monotonic_increasing
        print("\n--- 0. 데이터 유형 파악 (자동 감지) ---")
    elif data_type == 'time_series':
        is_time_series = True
        print("\n--- 0. 데이터 유형 파악 (명시적 지정: 시계열) ---")
    elif data_type == 'cross_sectional':
        is_time_series = False
        print("\n--- 0. 데이터 유형 파악 (명시적 지정: 횡단면) ---")
    else:
        raise ValueError("data_type은 'time_series', 'cross_sectional' 또는 None이어야 합니다.")
    
    if is_time_series:
        print("데이터는 시계열 데이터로 파악됩니다.")
    else:
        print("데이터는 횡단면 데이터로 파악됩니다.")

    # 1. 데이터 기본 정보
    print("\n--- 1. 데이터 기본 정보 ---")
    print(data.info())
    print(f"\n종속 변수: {y_var}")
    print(f"독립 변수: {X_vars}")

    # 2. 기초 통계량
    print("\n--- 2. 기초 통계량 ---")
    print(data.describe())

    # 3. 상관계수 행렬
    print("\n--- 3. 상관계수 행렬 ---")
    print(data[X_vars + [y_var]].corr())
    # 상관계수 행렬 그래프 출력 부분 제거

    # 4. 데이터 시각화
    print("\n--- 4. 데이터 시각화 ---")
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(X_vars + [y_var]):
        plt.subplot(1, len(X_vars) + 1, i + 1)
        sns.histplot(data[col], kde=True)
        plt.title(f'히스토그램 및 KDE: {col}')
    plt.tight_layout()
    plt.show()

    if is_time_series:
        print("\n시계열 데이터: 시간 변화에 따른 선 그래프 추가")
        plt.figure(figsize=(15, 6))
        for col in X_vars + [y_var]:
            plt.plot(data.index, data[col], label=col)
        plt.title("시간 변화에 따른 변수 추이")
        plt.xlabel("시간")
        plt.ylabel("값")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 5. OLS 회귀분석 결과
    print("\n--- 5. OLS 회귀분석 결과 ---")
    X_const = add_constant(X)
    ols_model = sm.OLS(y, X_const)
    ols_results = ols_model.fit()
    print(ols_results.summary())

    # 6. 잔차의 정규성 검정
    print("\n--- 6. 잔차의 정규성 검정 ---")
    residuals = ols_results.resid

    shapiro_test = stats.shapiro(residuals)
    print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue < 0.05:
        print("귀무가설 기각: 잔차는 정규 분포를 따르지 않습니다.")
    else:
        print("귀무가설 채택: 잔차는 정규 분포를 따른다고 볼 수 있습니다.")

    jb_test = sm.stats.jarque_bera(residuals)
    print(f"Jarque-Bera Test: Statistic={jb_test[0]:.4f}, p-value={jb_test[1]:.4f}")
    if jb_test[1] < 0.05:
        print("귀무가설 기각: 잔차는 정규 분포를 따르지 않습니다.")
    else:
        print("귀무가설 채택: 잔차는 정규 분포를 따른다고 볼 수 있습니다.")

    fig = sm.qqplot(residuals, line='s')
    plt.title("잔차 QQ Plot")
    plt.show()

    # 7. 모형설정 오류 검정 (RESET Test)
    print("\n--- 7. 모형설정 오류 검정 (RESET Test) ---")
    try:
        reset_output = smo.reset_ramsey(res=ols_results, degree=3)
        
        fstat_reset = reset_output.statistic
        fpval_reset = reset_output.pvalue

        if not np.isnan(fstat_reset):
            print(f"RESET Test (Ramsey): F-statistic={fstat_reset:.4f}, p-value={fpval_reset:.4f}")
            if fpval_reset < 0.05:
                print("귀무가설 기각: 모형 설정 오류가 존재할 수 있습니다 (Ramsey RESET test 결과).")
            else:
                print("귀무가설 채택: 모형 설정 오류가 없다고 볼 수 있습니다 (Ramsey RESET test 결과).")
        else:
            print("RESET Test 결과를 표시할 수 없습니다.")
            
    except Exception as e:
        print(f"RESET Test 수행 중 오류 발생: {e}")
        print("RESET Test는 일부 상황에서 수치적 문제로 오류가 발생할 수 있습니다 (예: 데이터 크기, 다중공선성).")


    # 8. 다중공선성 검정 (VIF)
    print("\n--- 8. 다중공선성 검정 (VIF) ---")
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) 
                       for i in range(X_const.shape[1])]
    
    if 'const' in vif_data['Variable'].values:
        vif_data = vif_data[vif_data['Variable'] != 'const']

    print(vif_data)
    print("\n일반적으로 VIF 값이 10 이상이면 심각한 다중공선성을 의심합니다.")

    # 9. 자기상관 검정 (시계열 데이터일 경우에만)
    if is_time_series:
        print("\n--- 9. 자기상관 검정 (시계열 데이터) ---")
        print("잔차 그래프 (시간에 따른)")
        plt.figure(figsize=(12, 5))
        plt.plot(data.index, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("시간에 따른 잔차 추이")
        plt.xlabel("시간")
        plt.ylabel("잔차")
        plt.grid(True)
        plt.show()

        # Durbin-Watson Test
        dw_test = durbin_watson(residuals)
        print(f"Durbin-Watson Test: Statistic={dw_test:.4f}")
        print("Durbin-Watson 통계량은 2에 가까우면 자기상관이 없음을 나타냅니다.")
        if dw_test < 1.5 or dw_test > 2.5:
            print("자기상관이 존재할 가능성이 있습니다.")

        # Breusch-Godfrey Test
        try:
            num_exog = X_const.shape[1]
            max_nlags = len(y) - num_exog - 1
            if max_nlags < 1: max_nlags = 1

            bg_test = sm.stats.acorr_breusch_godfrey(ols_results, nlags=min(int(len(y)/4), max_nlags))
            print(f"Breusch-Godfrey Test: LM Statistic={bg_test[0]:.4f}, p-value={bg_test[1]:.4f}")
            if bg_test[1] < 0.05:
                print("귀무가설 기각: 자기상관이 존재합니다.")
            else:
                print("귀무가설 채택: 자기상관이 없다고 볼 수 있습니다.")
        except Exception as e:
            print(f"Breusch-Godfrey Test 수행 중 오류 발생: {e}")
            print("충분한 관측치 수 또는 lag 설정 문제로 오류가 발생할 수 있습니다.")


        # Augmented Dickey-Fuller Test (잔차의 정상성 검정)
        print("\nAugmented Dickey-Fuller Test (잔차의 정상성 검정):")
        try:
            adf_test = adfuller(residuals)
            print(f"  ADF Statistic: {adf_test[0]:.4f}")
            print(f"  p-value: {adf_test[1]:.4f}")
            print("  Critical Values:")
            for key, value in adf_test[4].items():
                print(f"    {key}: {value:.4f}")
            if adf_test[1] < 0.05:
                print("귀무가설 기각: 잔차는 정상성을 가집니다.")
            else:
                print("귀무가설 채택: 잔차는 비정상성을 가집니다 (단위근이 존재할 가능성).")
        except Exception as e:
            print(f"Augmented Dickey-Fuller Test 수행 중 오류 발생: {e}")
            print("충분한 관측치 수 부족 또는 잔차 특성 문제로 오류가 발생할 수 있습니다.")

    # 10. 이분산 검정 (횡단면 데이터일 경우에만)
    if not is_time_series:
        print("\n--- 10. 이분산 검정 (횡단면 데이터) ---")
        print("잔차 그래프 (예측값에 따른)")
        plt.figure(figsize=(10, 6))
        plt.scatter(ols_results.fittedvalues, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("예측값")
        plt.ylabel("잔차")
        plt.title("예측값에 따른 잔차 산점도")
        plt.show()

        # Breusch-Pagan test
        try:
            bp_test = het_breuschpagan(residuals, X_const)
            print(f"Breusch-Pagan Test: LM Statistic={bp_test[0]:.4f}, p-value={bp_test[1]:.4f}")
            if bp_test[1] < 0.05:
                print("귀무가설 기각: 이분산성이 존재합니다.")
            else:
                print("귀무가설 채택: 이분산성이 없다고 볼 수 있습니다.")
        except Exception as e:
            print(f"Breusch-Pagan Test 수행 중 오류 발생: {e}")

        # White test
        try:
            white_test = het_white(residuals, X_const)
            print(f"White Test: LM Statistic={white_test[0]:.4f}, p-value={white_test[1]:.4f}")
            if white_test[1] < 0.05:
                print("귀무가설 기각: 이분산성이 존재합니다.")
            else:
                print("귀무가설 채택: 이분산성이 없다고 볼 수 있습니다.")
        except Exception as e:
            print(f"White Test 수행 중 오류 발생: {e}")
    else:
        print("\n--- 10. 이분산 검정 (시계열 데이터의 경우 수행하지 않음) ---")


    # 11. 자기상관 해결을 위한 추정법 (시계열 데이터일 경우에만)
    cochrane_orcutt_results = None
    prais_winsten_results = None # Prais-Winsten은 GLSAR로 통합하여 처리
    hac_se_results = None

    if is_time_series:
        print("\n--- 11. 자기상관 해결을 위한 추정법 (시계열 데이터) ---")

        # Cochrane-Orcutt (sm.GLSAR 사용)
        try:
            print("\n- Cochrane-Orcutt (GLSAR) 추정 -")
            # GLSAR 모델 생성: (1,0)은 AR(1) 오차를 가정
            # X_const (상수항 포함)를 전달해야 함
            glsar_model = sm.GLSAR(y, X_const, 1) # 1은 AR(1)을 의미
            # 반복 추정 (최대 100회, 수렴 허용 오차 1e-8)
            cochrane_orcutt_results = glsar_model.iterative_fit(maxiter=100, tol=1e-8)
            print(cochrane_orcutt_results.summary())
            print(f"추정된 AR(1) 계수 (rho): {glsar_model.rho[0]:.4f}")

        except Exception as e:
            print(f"Cochrane-Orcutt (GLSAR) 추정 중 오류 발생: {e}")
            print("GLSAR 추정은 데이터 특성이나 수렴 문제로 오류가 발생할 수 있습니다.")


        # Prais-Winsten (sm.GLSAR 사용)
        print("\n- Prais-Winsten (GLSAR) 추정 -")
        try:
            # GLSAR은 기본적으로 첫 관측치도 변환하는 Prais-Winsten 방식을 사용합니다.
            # iterative_fit()이 사실상 Prais-Winsten의 반복적인 rho 추정 과정을 포함합니다.
            # 따라서 별도의 Prais-Winsten 모델을 명시적으로 생성할 필요가 없습니다.
            # 위의 Cochrane-Orcutt (GLSAR) 결과가 Prais-Winsten의 특성을 포함합니다.
            print("Cochrane-Orcutt (GLSAR) 결과가 Prais-Winsten 추정의 특성을 포함합니다.")
            print("Prais-Winsten에 특화된 별도 구현은 필요하지 않습니다. 위의 GLSAR 결과를 참조하세요.")
        except Exception as e:
            print(f"Prais-Winsten 추정 중 오류 발생: {e}")


        # HAC 표준오차 (Newey-West)
        print("\n- HAC 표준오차 (Newey-West) -")
        try:
            num_exog = X_const.shape[1]
            max_lags_hac = len(y) - num_exog -1
            if max_lags_hac < 1:
                max_lags_hac = 1

            hac_se_results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags_hac})
            print(hac_se_results.summary())
        except Exception as e:
            print(f"HAC 표준오차 추정 중 오류 발생: {e}")

    else:
        print("\n--- 11. 자기상관 해결을 위한 추정법 (횡단면 데이터의 경우 수행하지 않음) ---")

    # 12. 이분산 문제 해결 추정법 (횡단면 데이터일 경우에만)
    gls_results = None
    white_robust_results = None

    if not is_time_series:
        print("\n--- 12. 이분산 문제 해결 추정법 (횡단면 데이터) ---")

        # 일반화 최소자승법(Generalized Least Squares: GLS)
        try:
            print("\n- 일반화 최소자승법(GLS) -")
            weights = 1.0 / (residuals**2 + 1e-8)
            gls_model = sm.WLS(y, X_const, weights=weights)
            gls_results = gls_model.fit()
            print(gls_results.summary())
        except Exception as e:
            print(f"GLS 추정 중 오류 발생: {e}")

        # White’s Robust standard error
        print("\n- White’s Robust standard error -")
        try:
            white_robust_results = ols_model.fit(cov_type='HC3')
            print(white_robust_results.summary())
        except Exception as e:
            print(f"White's Robust standard error 추정 중 오류 발생: {e}")
    else:
        print("\n--- 12. 이분산 문제 해결 추정법 (시계열 데이터의 경우 수행하지 않음) ---")


    # 13. 종합 추정 결과 요약표
    print("\n" + "=" * 50)
    print("종합 추정 결과 요약표")
    print("=" * 50)

    if is_time_series:
        print("\n--- 시계열 데이터 관련 추정결과 요약표 ---")
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
            print(f"시계열 데이터 추정 결과 요약표 생성 중 오류 발생: {e}")
            print("일부 모델의 결과가 비어있을 수 있습니다.")

        print("\n(참고: Cochrane-Orcutt (GLSAR)는 반복 추정된 자기상관 보정 결과입니다.)")

    if not is_time_series:
        print("\n--- 횡단면 데이터 관련 추정결과 요약표 ---")
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
            print(f"횡단면 데이터 추정 결과 요약표 생성 중 오류 발생: {e}")
            print("일부 모델의 결과가 비어있을 수 있습니다.")


    print("\n" + "=" * 50)
    print("분석 완료")
    print("=" * 50)