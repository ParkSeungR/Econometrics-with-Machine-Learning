import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import warnings

# VARResultsWrapper 클래스를 직접 임포트
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

# 경고 메시지 무시 (깔끔한 출력을 위해)
warnings.filterwarnings("ignore")

def Run_VAR_analysis(
    data: pd.DataFrame,
    endog_vars: list,
    exog_vars: list = None,
    # 아래 파라미터들은 모두 디폴트 값을 가집니다.
    maxlags_order_selection: int = 8,
    ic_criterion: str = 'aic', # 차수 선택 기준: 'aic', 'bic', 'fpe', 'hqic'
    diff_type: str = 'log_diff', # 안정화 방법: 'none', 'diff', 'pct_change', 'log_diff'
    pct_change_periods: int = 4, # 'pct_change' 사용 시 과거 몇 기간과 비교할지 (예: 분기 데이터의 전년 동기 대비 4)
    start_date: str = None, # 데이터셋을 특정 날짜부터 슬라이싱할 경우 (예: '1970-01-31')
    fixed_lag_order: int = None, # 수동으로 VAR 시차를 지정 (None이면 자동 선택)
    nlags_irf: int = 10,
    nsteps_fevd: int = 20,
    nsteps_forecast: int = 20
):
    """
    VAR (Vector Autoregression) 모형의 전체 분석 절차를 수행하는 함수.
    데이터셋과 내생/외생 변수만 필수로 입력받고, 나머지 옵션은 디폴트 값을 가집니다.

    Args:
        data (pd.DataFrame): 시계열 데이터프레임. 인덱스는 datetime 형식이어야 함.
        endog_vars (list): VAR 모형에 포함될 내생 변수(종속 변수) 리스트.
        exog_vars (list, optional): VARX 모형에 포함될 외생 변수 리스트. Defaults to None.
        maxlags_order_selection (int, optional): VAR 차수 결정을 위한 최대 시차. Defaults to 8.
        ic_criterion (str, optional): VAR 차수 선택 기준 ('aic', 'bic', 'fpe', 'hqic'). Defaults to 'aic'.
        diff_type (str, optional): 시계열 안정화를 위한 변환 방법.
                                    'none': 변환 없음 (원계열 사용)
                                    'diff': 1차 차분
                                    'pct_change': 퍼센트 변화
                                    'log_diff': 로그 차분 (증가율)
                                    Defaults to 'log_diff'.
        pct_change_periods (int, optional): 'pct_change' 사용 시 과거 몇 기간과 비교할지. Defaults to 4 (분기 데이터 전년 동기 대비).
        start_date (str, optional): 변환된 데이터셋의 시작 날짜를 지정합니다 (예: '1970-01-31'). Defaults to None (슬라이싱 없음).
        fixed_lag_order (int, optional): 사용자가 수동으로 VAR 모형의 시차를 지정. None이면 ic_criterion에 따라 자동 선택. Defaults to None.
        nlags_irf (int, optional): 충격반응 함수를 그릴 시차의 수. Defaults to 10.
        nsteps_fevd (int, optional): 예측 오차 분산 분해를 위한 단계 수. Defaults to 20.
        nsteps_forecast (int, optional): 예측을 위한 단계 수. Defaults to 20.

    Returns:
        results (statsmodels.tsa.vector_ar.var_model.VARResultsWrapper): VAR 모형 추정 결과 객체.
    """

    print("--- VAR 모형 분석 시작 ---")
    print(f"내생 변수: {endog_vars}")
    if exog_vars:
        print(f"외생 변수: {exog_vars}")
    print("-" * 30)

    # 0. 데이터 준비 및 전처리
    required_cols = endog_vars + (exog_vars if exog_vars else [])
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"데이터프레임에 다음 변수들이 없습니다: {missing_cols}")

    df = data[required_cols].copy()

    # 1. 시계열의 안정성 검정 및 안정화
    print("\n1. 시계열의 안정성 검정 및 안정화")
    print("원계열 시각화 (안정성 육안 확인):")
    df[endog_vars].plot(title="Original Series (Endogenous Variables)", figsize=(12, 6))
    plt.tight_layout()
    plt.show()
    print("원계열은 점차 증가하는 형태의 불안정한 시계열처럼 보일 수 있습니다.")

    df_transformed = df.copy()
    if diff_type == 'diff':
        df_transformed = df.diff().dropna()
        print(f"-> 1차 차분 ({diff_type})을 통해 시계열 안정화 시도.")
    elif diff_type == 'pct_change':
        df_transformed = 100 * df.pct_change(periods=pct_change_periods).dropna()
        print(f"-> 퍼센트 변화 ({diff_type}, periods={pct_change_periods})를 통해 시계열 안정화 시도.")
        print(f"   (주의: periods는 분기/월간 등 데이터 주기와 일치해야 합니다. 예: 분기 데이터는 4, 월간 데이터는 12)")
    elif diff_type == 'log_diff':
        df_transformed = np.log(df).diff().dropna() * 100
        print(f"-> 로그 차분 ({diff_type}, 증가율 시계열)을 통해 시계열 안정화 시도.")
    else:
        print("-> 시계열 변환을 수행하지 않고 원계열로 진행합니다 (diff_type='none').")

    # 추가된 부분: 특정 날짜부터 데이터 슬라이싱
    if start_date is not None:
        try:
            # pd.to_datetime으로 날짜 문자열을 Datetime 객체로 변환하여 사용
            df_transformed = df_transformed.loc[pd.to_datetime(start_date):]
            print(f"-> 데이터셋을 '{start_date}'부터 시작하도록 슬라이싱했습니다.")
        except KeyError:
            print(f"경고: 지정된 시작 날짜 '{start_date}' 이후의 데이터가 없습니다. 슬라이싱이 적용되지 않았을 수 있습니다.")
        except Exception as e:
            print(f"경고: start_date 슬라이싱 중 오류 발생: {e}. 슬라이싱을 건너뜁니다.")


    # 변환 후 데이터가 너무 짧아지면 문제 발생 가능성 방지
    min_required_len = max(maxlags_order_selection, 1) + len(endog_vars) + 1 
    if len(df_transformed) < min_required_len:
        raise ValueError(f"데이터 변환 후 관측치 수가 너무 적습니다 ({len(df_transformed)}개). VAR 모델 추정을 위한 최소 관측치 수는 약 {min_required_len}개 입니다. `diff_type`을 변경하거나 더 긴 시계열 데이터를 사용하세요.")

    print("\n변환된 시계열 시각화:")
    df_transformed[endog_vars].plot(title=f"Transformed Series ({diff_type})", figsize=(12, 6))
    plt.tight_layout()
    plt.show()
    print("안정적으로 보이는 증가율 시계열은 특정 증가율 수준에서 아래위로 분포하는 안정적 시계열의 모습을 보입니다.")
    print("\n--- 변환된 데이터 (Transformed Data) 미리보기 ---")
    print(df_transformed.head())
    print("-" * 30)


    print("\nADF (Augmented Dickey-Fuller) 단위근 검정:")
    print("귀무가설 (H0): 시계열에 단위근이 존재한다 (불안정하다).")
    print("대립가설 (H1): 시계열에 단위근이 존재하지 않는다 (안정하다).")
    print("검정통계량이 임계치보다 작고 (더 음수), p-값이 유의수준(예: 0.05)보다 작으면 H0를 기각하고 안정적이라고 판단합니다.")

    for col in endog_vars:
        print(f"\n--- {col} ---")
        adf_results_c = adfuller(df_transformed[col], regression='c')
        print(f"상수항(drift) 포함 검정 (regression='c'):")
        print(f"  ADF Statistic: {adf_results_c[0]:.4f}")
        print(f"  p-value: {adf_results_c[1]:.4f}")
        print(f"  Critical Values: {adf_results_c[4]}")
        if adf_results_c[1] <= 0.05:
            print("  => 5% 유의수준에서 단위근이 없다고 판단 (안정적).")
        else:
            print("  => 5% 유의수준에서 단위근이 있다고 판단 (불안정).")


    # 2. VAR 모형의 차수 결정
    print("\n\n2. VAR 모형의 차수 결정")
    if fixed_lag_order is not None:
        optimal_lag_order = fixed_lag_order
        print(f"사용자가 고정 시차 {optimal_lag_order}를 지정했습니다.")
    else:
        model_for_order = VAR(df_transformed[endog_vars])
        print(f"최대 시차 {maxlags_order_selection}까지 {ic_criterion.upper()} 기준 차수 선택:")

        adjusted_maxlags = maxlags_order_selection
        if len(df_transformed) <= maxlags_order_selection:
            adjusted_maxlags = max(1, len(df_transformed) - 1)
            print(f"경고: 데이터 길이가 짧아 `maxlags_order_selection`({maxlags_order_selection})이 너무 큽니다. `maxlags`를 {adjusted_maxlags}로 조정합니다.")
        
        selected_order = model_for_order.select_order(maxlags=adjusted_maxlags)
            
        print(selected_order.summary())
        optimal_lag_order_by_ic = selected_order.selected_orders[ic_criterion]
        print(f"\n'{ic_criterion.upper()}' 기준 최적 시차: {optimal_lag_order_by_ic}")
        
        optimal_lag_order = optimal_lag_order_by_ic

        if optimal_lag_order == 0:
            print("\n!!! 경고: 선택된 최적 시차가 0입니다. 이 경우 Granger 인과성, IRF, FEVD 등에 문제가 발생할 수 있습니다.")
            print("   `fixed_lag_order` 파라미터를 사용하여 시차를 수동으로 지정해보세요 (예: `fixed_lag_order=1` 또는 `2`).")
            print("   또는 `ic_criterion`을 'bic' 등으로 변경하여 다시 시도해보세요.")
        else:
            print("여러 통계량이 모두 동일한 차수를 적절한 차수로 제시하지 않으므로 추가적인 진단과정을 통해 적정 차수를 결정해야 합니다.")


    # 3. 모형 추정
    print("\n\n3. VAR 모형 추정")
    print(f"선택된 최적 시차 {optimal_lag_order}로 VAR 모형을 추정합니다.")
    if exog_vars:
        try:
            model = VARMAX(endog=df_transformed[endog_vars], order=(optimal_lag_order, 0), exog=df_transformed[exog_vars])
            results = model.fit(maxiter=1000, disp=False)
            print("VARX 모형이 추정되었습니다.")
        except Exception as e:
            print(f"VARX 모형 추정 중 오류 발생: {e}")
            print("VARX 모형 대신 VAR 모형으로 추정을 시도합니다.")
            model = VAR(df_transformed[endog_vars])
            results = model.fit(optimal_lag_order)
            print("VAR 모형이 추정되었습니다.")
    else:
        model = VAR(df_transformed[endog_vars])
        results = model.fit(optimal_lag_order)
        print("VAR 모형이 추정되었습니다.")

    actual_lag_order = results.k_ar if hasattr(results, 'k_ar') else optimal_lag_order
    print(f"**실제 추정된 모형의 시차 (results.k_ar): {actual_lag_order}**")
    print("\n--- 모형 추정 결과 요약 (Summary) ---")
    print(results.summary())
    print("\n개별 방정식별 파라미터 추정치와 유의성 검정은 모형 평가에서 중요시되지 않습니다.")
    print("대신 그랜져 인과 검정, 충격반응 함수, 예측 오차의 분해와 같은 통계량들이 자주 사용됩니다.")


    # 4. 진단
    print("\n\n4. 모형 진단")

    # 4-1. 잔차의 자기상관 여부 검정
    print("\n4-1. 잔차의 자기상관 여부 검정 (ACF Plots):")
    if isinstance(results, VARResultsWrapper):
        results.plot_acorr()
        plt.suptitle("ACF plots for residuals", y=1.02)
        plt.tight_layout()
        plt.show()
        print("ACF 플롯에서 신뢰구간(점선)을 벗어나는 막대가 있다면 자기상관 문제가 있을 수 있습니다.")
        print("차수별로 자기상관 문제가 있는지 카이제곱 통계량과 유의수준으로 판단할 수 있습니다.")
    else:
        print("VARMAX 결과 객체에서는 plot_acorr()를 직접 지원하지 않습니다. 수동으로 잔차 ACF를 확인해야 합니다.")
        residuals = results.resid
        fig, axes = plt.subplots(len(endog_vars), 1, figsize=(10, 3 * len(endog_vars)))
        if len(endog_vars) == 1:
            axes = [axes]
        for i, var in enumerate(endog_vars):
            sm.graphics.tsa.plot_acf(residuals[var], lags=nlags_irf, ax=axes[i], title=f'Residuals ACF for {var}')
        plt.tight_layout()
        plt.show()


    # 4-2. 잔차의 정규성 검정 (Jarque-Bera test)
    print("\n4-2. 잔차의 정규성 검정 (Jarque-Bera test):")
    print("귀무가설 (H0): 오차항이 정규분포를 따른다.")
    print("대립가설 (H1): 오차항이 정규분포를 따르지 않는다.")
    if isinstance(results, VARResultsWrapper):
        normality_test = results.test_normality()
        print(normality_test.summary())
        if normality_test.pvalue <= 0.05:
            print("=> 5% 유의수준에서 귀무가설을 기각합니다. 오차항은 정규분포를 따르지 않을 수 있습니다.")
        else:
            print("=> 5% 유의수준에서 귀무가설을 기각하지 못합니다. 오차항은 정규분포를 따른다고 볼 수 있습니다.")
    else:
        print("VARMAX 결과 객체에서는 test_normality()를 직접 지원하지 않습니다. 수동으로 잔차 정규성을 검정해야 합니다.")
        for col in endog_vars:
            print(f"\n--- {col} Residuals Normality Test ---")
            jb_test = sm.stats.stattools.jarque_bera(results.resid[col])
            print(f"  Jarque-Bera Statistic: {jb_test[0]:.4f}")
            print(f"  p-value: {jb_test[1]:.4f}")
            print(f"  Skewness: {jb_test[2]:.4f}")
            print(f"  Kurtosis: {jb_test[3]:.4f}")
            if jb_test[1] <= 0.05:
                print("  => 5% 유의수준에서 정규분포 가설을 기각합니다.")
            else:
                print("  => 5% 유의수준에서 정규분포 가설을 기각하지 못합니다.")


    # 4-3. Granger 인과성 검정
    print("\n\n4-3. Granger 인과성 검정:")
    print("Granger 인과성 검정은 변수들 간의 실제 인과관계 방향을 분석합니다.")
    if optimal_lag_order == 0:
        print("  !!! 경고: VAR 모형의 시차가 0이므로 Granger 인과성 검정을 수행할 수 없습니다.")
        print("  시차가 0인 모형에서는 변수 간의 동태적인 인과 관계를 분석할 수 없습니다.")
    else:
        for target_var in endog_vars:
            other_vars = [v for v in endog_vars if v != target_var]
            if other_vars:
                try:
                    if isinstance(results, VARResultsWrapper):
                        causality_test = results.test_causality(target_var, other_vars, kind='f')
                        print(f"\n{other_vars}가 {target_var}를 Granger-인과하는지 검정:")
                        print(causality_test.summary())
                        if causality_test.pvalue <= 0.05:
                            print(f"  => 5% 유의수준에서 귀무가설을 기각합니다. {other_vars}는 {target_var}를 Granger-인과한다고 볼 수 있습니다.")
                        else:
                            print(f"  => 5% 유의수준에서 귀무가설을 기각하지 못합니다. {other_vars}는 {target_var}를 Granger-인과하지 않는다고 볼 수 있습니다.")
                    else:
                        print(f"VARMAX 결과 객체에서는 test_causality()를 직접 지원하지 않습니다. 수동으로 그랜져 인과성 검정을 수행해야 합니다.")
                except Exception as e:
                    print(f"  {target_var}에 대한 Granger 인과성 검정 중 오류 발생: {e}")
            else:
                print(f"{target_var}는 다른 내생 변수가 없으므로 Granger 인과성 검정을 건너뜁니다.")

    # 5. 충격 반응 함수 (Impulse Response Function, IRF)
    print("\n\n5. 충격 반응 함수 (IRF)")
    if optimal_lag_order == 0:
        print("  !!! 경고: VAR 모형의 시차가 0이므로 충격 반응 함수를 계산할 수 없습니다.")
        print("  시차가 0인 모형에서는 동태적인 충격 효과를 분석할 수 없습니다.")
    else:
        print("충격반응 함수는 VAR 모형에서 특정 방정식의 오차항에 대한 충격이 다른 내생 변수들의 미래값에 미치는 영향을 살펴봅니다.")
        print("오차항의 1 표준편차 충격에 대한 반응 정도를 계산합니다.")

        try:
            irf = results.irf(nlags_irf)
            
            print("\n--- 비직교화된 충격 반응 함수 (Non-Orthogonalized IRF) ---")
            irf.plot_cum_effects(orth=False)
            plt.suptitle("Cumulative responses (Non-Orthogonalized)", y=1.02)
            plt.tight_layout()
            plt.show()

            print("\n--- 직교화된 충격 반응 함수 (Orthogonalized IRF) ---")
            print("직교화된 충격반응 함수는 변수 순서에 민감합니다. 촐레스키 분해를 통해 구해집니다.")
            irf.plot(orth=True)
            plt.suptitle("Impulse responses (Orthogonalized)", y=1.02)
            plt.tight_layout()
            plt.show()
            print("충격 반응 함수를 통해 경제 변수 간의 인과관계 분석이나 정책 효과 분석을 할 수 있습니다.")

        except Exception as e:
            print(f"충격 반응 함수 계산 또는 플로팅 중 오류 발생: {e}")
            print("VARMAX 모형의 경우 impulse_responses 메서드 사용법이 다를 수 있습니다.")
            if exog_vars:
                print("VARMAX 모형의 경우 impulse_responses는 VARMAX 객체에서 직접 호출해야 합니다.")
                print("자동화를 위해 첫 번째 변수에 대한 충격만 예시로 플롯합니다.")
                try:
                    impulse_array = np.zeros(len(endog_vars))
                    impulse_array[0] = 1
                    ax = results.impulse_responses(nlags_irf, orthogonalized=True, impulse=impulse_array).plot(figsize=(12,4))
                    ax.set(xlabel='t', title=f'Responses to a shock to `{endog_vars[0]}` (Orthogonalized)');
                    plt.tight_layout()
                    plt.show()
                except Exception as e_varmax_irf:
                    print(f"VARMAX 충격반응 함수 플로팅 중 오류 발생: {e_varmax_irf}")


    # 6. 예측 오차 분산의 분해 (Forecast Error Variance Decomposition, FEVD)
    print("\n\n6. 예측 오차 분산의 분해 (FEVD)")
    if optimal_lag_order == 0:
        print("  !!! 경고: VAR 모형의 시차가 0이므로 예측 오차 분산 분해를 계산할 수 없습니다.")
        print("  시차가 0인 모형에서는 예측 오차의 동태적인 분해를 할 수 없습니다.")
    else:
        print("FEVD는 시간의 변화에 따라 종속변수의 변이가 과거 자기 자신의 분산과 다른 변수의 분산에 의해 설명되는 정도를 보여줍니다.")
        try:
            fevd = results.fevd(nsteps_fevd)
            fevd.plot()
            plt.suptitle("Forecast Error Variance Decomposition (FEVD)", y=1.02)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"예측 오차 분산 분해 계산 또는 플로팅 중 오류 발생: {e}")
            print("VARMAX 모형의 경우 fevd를 직접 지원하지 않거나 사용법이 다를 수 있습니다.")
            if exog_vars:
                print("VARMAX 모형은 fevd 메서드를 직접 지원하지 않습니다. 수동 계산이 필요합니다.")


    # 7. 예측 (Forecast)
    print("\n\n7. 예측")
    if optimal_lag_order == 0:
        print("  !!! 경고: VAR 모형의 시차가 0이므로 예측을 제대로 수행하기 어렵습니다.")
        print("  시차가 0인 모형은 과거 정보가 미래에 영향을 주지 않는다고 가정합니다. 단순 평균 예측과 유사할 수 있습니다.")
    else:
        print("VAR 모형을 이용한 예측에서는 사후 예측과 사전 예측을 할 수 있습니다.")
        try:
            lag_order_for_forecast = results.k_ar if hasattr(results, 'k_ar') else optimal_lag_order

            if exog_vars:
                print("VARX/VARMA 모형의 predict/forecast 메서드 사용법은 VAR과 다를 수 있습니다.")
                print("plot_forecast는 VARX/VARMA에서도 사용 가능합니다.")
                results.plot_forecast(steps=nsteps_forecast)
                plt.suptitle(f"Forecast for {nsteps_forecast} steps (VARX)", y=1.02)
                plt.tight_layout()
                plt.show()
            else:
                print("\n--- 예측 시각화 ---")
                results.plot_forecast(steps=nsteps_forecast)
                plt.suptitle(f"Forecast for {nsteps_forecast} steps (VAR)", y=1.02)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"예측 계산 또는 플로팅 중 오류 발생: {e}")

    print("\n--- VAR 모형 분석 완료 ---")
    return results