#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
계량경제학 종합 분석 함수
파일명: Functions/econometric_analysis.py
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

# 한글 폰트 설정 (matplotlib에서 한글 깨짐 방지)
plt.rcParams['font.family'] = 'DejaVu Sans'
try:
    # Windows의 경우
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    try:
        # 다른 OS의 경우
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        pass
plt.rcParams['axes.unicode_minus'] = False

def Run_Comprehensive_OLS_analysis(data, dependent_var, independent_vars, data_type='cross_section', alpha=0.05, output_format='text'):
    """
    계량경제학 종합 분석 함수
    
    Parameters:
    -----------
    data : pandas.DataFrame
        분석할 데이터
    dependent_var : str
        종속변수명
    independent_vars : list
        독립변수명 리스트
    data_type : str, default='cross_section'
        데이터 유형 ('cross_section' 또는 'time_series')
    alpha : float, default=0.05
        유의수준
    output_format : str, default='text'
        출력 형식 ('text', 'latex', 'html')
    
    Returns:
    --------
    dict: 분석 결과를 담은 딕셔너리
    """
    
    print("="*80)
    print("                   계량경제학 종합 분석 결과")
    print("="*80)
    
    # 1. 기초 데이터 정보
    print("\n1. 데이터 기본 정보")
    print("-"*50)
    print(f"관측치 수: {len(data)}")
    print(f"종속변수: {dependent_var}")
    print(f"독립변수: {', '.join(independent_vars)}")
    print(f"데이터 유형: {data_type}")
    
    # 분석에 사용할 변수들
    all_vars = [dependent_var] + independent_vars
    analysis_data = data[all_vars].dropna()
    
    print(f"결측치 제거 후 관측치 수: {len(analysis_data)}")
    
    # 2. 기초 통계량
    print("\n2. 기초 통계량")
    print("-"*50)
    desc_stats = analysis_data.describe().round(4)
    display(desc_stats)
    
    # 3. 상관계수 행렬
    print("\n3. 상관계수 행렬")
    print("-"*50)
    corr_matrix = analysis_data.corr().round(4)
    display(corr_matrix)
    
    # 4. 시각화
    print("\n4. 데이터 시각화")
    print("-"*50)
    
    # 히스토그램
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
    
    # 종속변수와 독립변수간 산점도
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
    
    # 5. OLS 회귀분석
    print("\n5. OLS 회귀분석 결과")
    print("-"*50)
    
    # 회귀식 구성
    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
    
    # OLS 추정
    ols_model = smf.ols(formula=formula, data=analysis_data)
    ols_results = ols_model.fit()
    
    display(ols_results.summary())
    
    results_dict = {
        'ols_results': ols_results,
        'formula': formula,
        'data': analysis_data
    }
    
    # 6. 정규성 검정
    print("\n6. 잔차의 정규성 검정")
    print("-"*50)
    
    residuals = ols_results.resid
    
    # Shapiro-Wilk 검정
    sw_stat, sw_pval = stats.shapiro(residuals)
    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {sw_stat:.4f}")
    print(f"  p-value: {sw_pval:.4f}")
    print(f"  Result: {'Reject Normality' if sw_pval < alpha else 'Accept Normality'}")
    
    # Jarque-Bera 검정
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    print(f"\nJarque-Bera Test:")
    print(f"  Statistic: {jb_stat:.4f}")
    print(f"  p-value: {jb_pval:.4f}")
    print(f"  Result: {'Reject Normality' if jb_pval < alpha else 'Accept Normality'}")
    
    # 잔차 히스토그램
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
    
    # 7. 모형설정 오류 검정 (RESET Test)
    print("\n7. 모형설정 오류 검정 (RESET Test)")
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
    
    # 8. 다중공선성 검정 (VIF)
    print("\n8. 다중공선성 검정 (VIF)")
    print("-"*50)
    
    if len(independent_vars) > 1:
        # VIF 계산을 위한 행렬 생성
        y, X = pt.dmatrices(formula, data=analysis_data, return_type='dataframe')
        
        # VIF 계산
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns[1:]  # 상수항 제외
        vif_data["VIF"] = [smo.variance_inflation_factor(X.values, i) 
                          for i in range(1, X.shape[1])]
        
        display(vif_data)
        
        # VIF 해석
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
    
    # 9. 횡단면 자료의 경우: 이분산 검정 및 대안 추정법
    if data_type == 'cross_section':
        print("\n9. 이분산성 검정 및 대안 추정법")
        print("-"*50)
        
        # Breusch-Pagan 검정
        y, X = pt.dmatrices(formula, data=analysis_data, return_type='dataframe')
        bp_result = dg.het_breuschpagan(ols_results.resid, X)
        bp_stat, bp_pval = bp_result[0], bp_result[1]
        
        print(f"Breusch-Pagan Test:")
        print(f"  LM statistic: {bp_stat:.4f}")
        print(f"  p-value: {bp_pval:.4f}")
        print(f"  Result: {'Heteroskedasticity Present' if bp_pval < alpha else 'Homoskedasticity Accepted'}")
        
        # White 검정
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
        
        # 이분산성 탐지 여부 확인
        heteroskedastic = bp_pval < alpha or white_pval < alpha
        print(f"\n이분산성 탐지 여부: {heteroskedastic}")
        print(f"BP p-value < {alpha}: {bp_pval < alpha}")
        print(f"White p-value < {alpha}: {white_pval < alpha}")
        
        # 검정 결과를 먼저 저장
        results_dict['heteroskedasticity_tests'] = {
            'bp_test': {'statistic': bp_stat, 'pvalue': bp_pval},
            'white_test': {'statistic': white_stat, 'pvalue': white_pval}
        }
        
        # 대안 추정법 실행 (이분산성 탐지 여부와 관계없이 실행해서 비교)
        print(f"\n대안 추정법을 실행합니다...")
        
        # White의 강건 표준오차 (항상 실행)
        try:
            print("  - White 강건 표준오차 추정 중...")
            ols_robust = ols_model.fit(cov_type='HC3')
            results_dict['ols_robust'] = ols_robust
            print("  ? White 강건 표준오차 완료")
        except Exception as e:
            print(f"  ? White 강건 표준오차 실패: {e}")
            results_dict['ols_robust'] = None
        
        # WLS (가중최소자승법) - 항상 실행
        try:
            print("  - WLS 추정 중...")
            # 잔차의 절댓값으로 가중치 생성
            abs_resid = np.abs(ols_results.resid)
            # 0으로 나누기 방지
            abs_resid = np.where(abs_resid < 1e-6, 1e-6, abs_resid)
            weights = 1 / abs_resid
            weights = weights / np.mean(weights)  # 표준화
            
            wls_model = smf.wls(formula=formula, data=analysis_data, weights=weights)
            wls_results = wls_model.fit()
            results_dict['wls_results'] = wls_results
            print("  ? WLS 추정 완료")
        except Exception as e:
            print(f"  ? WLS 추정 실패: {e}")
            results_dict['wls_results'] = None
        
        if heteroskedastic:
            print(f"\n? 이분산성이 탐지되어 대안 추정법을 권장합니다.")
        else:
            print("??  이분산성이 탐지되지 않았지만, 비교를 위해 대안 추정법도 실행했습니다.")
        
        # 결과 확인
        print(f"\n추정 결과 요약:")
        print(f"  - OLS: 완료")
        print(f"  - OLS (Robust SE): {'완료' if results_dict.get('ols_robust') is not None else '실패'}")
        print(f"  - WLS: {'완료' if results_dict.get('wls_results') is not None else '실패'}")
    
    # 10. 시계열 자료의 경우: 자기상관 검정 및 대안 추정법  
    elif data_type == 'time_series':
        print("\n9. 자기상관 검정 및 대안 추정법")
        print("-"*50)
        
        # Durbin-Watson 검정
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
        
        # Breusch-Godfrey 검정
        bg_result = dg.acorr_breusch_godfrey(ols_results, nlags=3)
        bg_stat, bg_pval = bg_result[2], bg_result[3]
        
        print(f"\nBreusch-Godfrey Test (3 lags):")
        print(f"  F-statistic: {bg_stat:.4f}")
        print(f"  p-value: {bg_pval:.4f}")
        print(f"  Result: {'Autocorrelation Present' if bg_pval < alpha else 'No Autocorrelation'}")
        
        # ARCH 효과 검정 (이분산성 + 자기상관)
        try:
            arch_result = dg.het_arch(ols_results.resid, nlags=3)
            arch_stat, arch_pval = arch_result[2], arch_result[3]
            print(f"\nARCH Test (3 lags):")
            print(f"  F-statistic: {arch_stat:.4f}")
            print(f"  p-value: {arch_pval:.4f}")
            print(f"  Result: {'ARCH Effects Present' if arch_pval < alpha else 'No ARCH Effects'}")
        except Exception as e:
            print(f"\nARCH Test failed: {e}")
            arch_pval = 1.0  # 검정 실패시 효과 없음으로 간주
        
        # 단위근 검정 (기본적인 시계열 분석)
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # 종속변수에 대한 ADF 검정
            adf_result = adfuller(analysis_data[dependent_var].dropna(), autolag='AIC')
            adf_stat, adf_pval = adf_result[0], adf_result[1]
            
            print(f"\nAugmented Dickey-Fuller Test (종속변수):")
            print(f"  ADF statistic: {adf_stat:.4f}")
            print(f"  p-value: {adf_pval:.4f}")
            print(f"  Result: {'Stationary' if adf_pval < alpha else 'Non-stationary (Unit Root)'}")
            
        except Exception as e:
            print(f"\nADF Test failed: {e}")
            adf_pval = 1.0
        
        # 문제 탐지 여부 확인
        autocorrelated = dw_stat < 1.5 or dw_stat > 2.5 or bg_pval < alpha
        arch_effects = arch_pval < alpha
        non_stationary = adf_pval >= alpha
        
        print(f"\n시계열 문제 탐지 결과:")
        print(f"  자기상관: {autocorrelated}")
        print(f"  ARCH 효과: {arch_effects}")
        print(f"  비정상성: {non_stationary}")
        
        # 검정 결과를 먼저 저장
        results_dict['time_series_tests'] = {
            'dw_test': {'statistic': dw_stat, 'interpretation': dw_interpretation},
            'bg_test': {'statistic': bg_stat, 'pvalue': bg_pval},
            'arch_test': {'statistic': arch_stat if 'arch_stat' in locals() else 'N/A', 
                         'pvalue': arch_pval},
            'adf_test': {'statistic': adf_stat if 'adf_stat' in locals() else 'N/A', 
                        'pvalue': adf_pval}
        }
        
        # 대안 추정법 실행 (문제 탐지 여부와 관계없이 실행해서 비교)
        print(f"\n대안 추정법을 실행합니다...")
        
        # 1. HAC 표준오차 (Newey-West) - 항상 실행
        try:
            print("  - HAC 표준오차 (Newey-West) 추정 중...")
            ols_hac = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': 2})
            results_dict['ols_hac'] = ols_hac
            print("  ? HAC 표준오차 완료")
        except Exception as e:
            print(f"  ? HAC 표준오차 실패: {e}")
            results_dict['ols_hac'] = None
        
        # 2. Cochrane-Orcutt 추정법 - 항상 실행
        try:
            print("  - Cochrane-Orcutt 추정 중...")
            y, X = pt.dmatrices(formula, data=analysis_data, return_type='dataframe')
            corc_model = sm.GLSAR(y, X)
            corc_results = corc_model.iterative_fit(maxiter=100)
            results_dict['corc_results'] = corc_results
            print("  ? Cochrane-Orcutt 추정 완료")
        except Exception as e:
            print(f"  ? Cochrane-Orcutt 추정 실패: {e}")
            results_dict['corc_results'] = None
        
        # 3. Prais-Winsten 추정법 - 새로 추가
        try:
            print("  - Prais-Winsten 추정 중...")
            from statsmodels.tsa.arima_model import ARIMA
            from statsmodels.regression.linear_model import yule_walker
            
            # AR(1) 계수 추정
            rho_yw = yule_walker(ols_results.resid, order=1)[0][0]
            
            # Prais-Winsten 변환
            y_pw = analysis_data[dependent_var].copy()
            X_pw = analysis_data[independent_vars].copy()
            
            # 첫 번째 관측치 변환: sqrt(1-rho^2)
            factor = np.sqrt(1 - rho_yw**2)
            y_pw.iloc[0] = y_pw.iloc[0] * factor
            X_pw.iloc[0] = X_pw.iloc[0] * factor
            
            # 나머지 관측치 변환: Y_t - rho*Y_(t-1)
            for i in range(1, len(y_pw)):
                y_pw.iloc[i] = y_pw.iloc[i] - rho_yw * y_pw.iloc[i-1]
                X_pw.iloc[i] = X_pw.iloc[i] - rho_yw * X_pw.iloc[i-1]
            
            # 변환된 데이터로 OLS 추정
            pw_data = pd.concat([y_pw, X_pw], axis=1)
            pw_formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
            pw_model = smf.ols(formula=pw_formula, data=pw_data)
            pw_results = pw_model.fit()
            
            results_dict['prais_winsten_results'] = pw_results
            print("  ? Prais-Winsten 추정 완료")
        except Exception as e:
            print(f"  ? Prais-Winsten 추정 실패: {e}")
            results_dict['prais_winsten_results'] = None
        
        # 4. GARCH(1,1) 추정 (ARCH 효과가 있는 경우 권장)
        if arch_effects:
            try:
                print("  - GARCH(1,1) 추정 중... (ARCH 효과 탐지됨)")
                from arch import arch_model
                
                # GARCH 모델 설정
                garch_model = arch_model(analysis_data[dependent_var], 
                                       x=analysis_data[independent_vars],
                                       vol='GARCH', p=1, q=1)
                garch_results = garch_model.fit(disp='off')
                results_dict['garch_results'] = garch_results
                print("  ? GARCH(1,1) 추정 완료")
            except Exception as e:
                print(f"  ? GARCH(1,1) 추정 실패: {e}")
                results_dict['garch_results'] = None
        else:
            results_dict['garch_results'] = None
        
        # 결과 요약
        any_problems = autocorrelated or arch_effects or non_stationary
        if any_problems:
            print(f"\n? 시계열 문제가 탐지되어 대안 추정법을 권장합니다.")
            if autocorrelated:
                print("   → 자기상관: HAC 표준오차, Cochrane-Orcutt, Prais-Winsten 권장")
            if arch_effects:
                print("   → ARCH 효과: GARCH 모델 권장")
            if non_stationary:
                print("   → 비정상성: 차분(differencing) 또는 공적분 분석 고려")
        else:
            print("??  시계열 문제가 탐지되지 않았지만, 비교를 위해 대안 추정법도 실행했습니다.")
        
        # 결과 확인
        print(f"\n추정 결과 요약:")
        print(f"  - OLS: 완료")
        print(f"  - OLS (HAC SE): {'완료' if results_dict.get('ols_hac') is not None else '실패'}")
        print(f"  - Cochrane-Orcutt: {'완료' if results_dict.get('corc_results') is not None else '실패'}")
        print(f"  - Prais-Winsten: {'완료' if results_dict.get('prais_winsten_results') is not None else '실패'}")
        print(f"  - GARCH(1,1): {'완료' if results_dict.get('garch_results') is not None else '미실행/실패'}")
    
    # 11. 결과 요약표 생성
    print("\n10. 추정 결과 요약표")
    print("-"*50)
    
    # 결과 모델들을 리스트로 정리
    models_to_display = [ols_results]
    model_names = ['OLS']
    
    print(f"기본 OLS 모델 추가 완료")
    
    # 데이터 유형별 추가 모델들
    if data_type == 'cross_section':
        print("횡단면 자료 - 이분산 관련 모델들 확인 중...")
        
        if 'ols_robust' in results_dict and results_dict['ols_robust'] is not None:
            models_to_display.append(results_dict['ols_robust'])
            model_names.append('OLS (Robust SE)')
            print("  ? OLS (Robust SE) 추가")
        else:
            print("  ? OLS (Robust SE) 없음")
            
        if 'wls_results' in results_dict and results_dict['wls_results'] is not None:
            models_to_display.append(results_dict['wls_results'])
            model_names.append('WLS')
            print("  ? WLS 추가")
        else:
            print("  ? WLS 없음")
    
    elif data_type == 'time_series':
        print("시계열 자료 - 자기상관 관련 모델들 확인 중...")
        
        if 'ols_hac' in results_dict and results_dict['ols_hac'] is not None:
            models_to_display.append(results_dict['ols_hac'])
            model_names.append('OLS (HAC SE)')
            print("  ? OLS (HAC SE) 추가")
        else:
            print("  ? OLS (HAC SE) 없음")
            
        if 'corc_results' in results_dict and results_dict['corc_results'] is not None:
            models_to_display.append(results_dict['corc_results'])
            model_names.append('Cochrane-Orcutt')
            print("  ? Cochrane-Orcutt 추가")
        else:
            print("  ? Cochrane-Orcutt 없음")
            
        if 'prais_winsten_results' in results_dict and results_dict['prais_winsten_results'] is not None:
            models_to_display.append(results_dict['prais_winsten_results'])
            model_names.append('Prais-Winsten')
            print("  ? Prais-Winsten 추가")
        else:
            print("  ? Prais-Winsten 없음")
        
        # GARCH는 구조가 다르므로 일반적인 표에 포함하지 않음
        if 'garch_results' in results_dict and results_dict['garch_results'] is not None:
            print("  ??  GARCH 모델은 별도 출력됨")
        else:
            print("  ? GARCH 없음")
    
    print(f"\n최종 모델 수: {len(models_to_display)}")
    print(f"최종 모델명: {model_names}")
    
    # 강제로 수동 표 생성 (더 안정적)
    print("\n수동 표 생성을 시작합니다...")
    try:
        print("="*100)
        print("                              추정 결과 요약표")
        print("="*100)
        
        # 모든 모델의 결과를 DataFrame으로 정리
        summary_data = []
        var_names = None
        
        for i, (model, name) in enumerate(zip(models_to_display, model_names)):
            try:
                print(f"  모델 {i+1}: {name} 처리 중...")
                
                # 계수 추출
                coeffs = model.params
                std_errors = model.bse
                t_values = model.tvalues
                p_values = model.pvalues
                
                if i == 0:  # 첫 번째 모델에서 변수명 추출
                    var_names = list(coeffs.index)
                    print(f"    변수명: {var_names}")
                
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
                print(f"    ? {name} 처리 완료")
                
            except Exception as e:
                print(f"    ? 모델 {name} 처리 중 오류: {e}")
                continue
        
        print(f"\n처리된 모델 수: {len(summary_data)}")
        
        # 표 출력
        if summary_data and var_names:
            # 헤더 출력
            col_width = 20
            header = "Variable".ljust(15)
            for name in model_names[:len(summary_data)]:
                header += name.center(col_width)
            print(header)
            print("-" * len(header))
            
            # 계수 출력
            for var in var_names:
                # 계수 행
                row = var.ljust(15)
                for data in summary_data:
                    try:
                        coeff = data['Coefficients'][var]
                        p_val = data['P_Values'][var]
                        
                        # 유의성 표시
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
                        print(f"      변수 {var} 처리 오류: {e}")
                        row += "N/A".center(col_width)
                print(row)
                
                # 표준오차 행
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
            
            # 모델 통계량
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
            print("? 수동 표 생성 완료!")
            
        else:
            print("? 표 생성을 위한 데이터가 부족합니다.")
            print("기본 회귀 결과를 개별적으로 출력합니다:")
            for i, model in enumerate(models_to_display):
                print(f"\n{model_names[i]} Results:")
                print(model.summary().tables[1])
        
    except Exception as e:
        print(f"? 수동 표 생성 중 오류: {e}")
        print("기본 회귀 결과를 개별적으로 출력합니다:")
        for i, model in enumerate(models_to_display):
            print(f"\n{model_names[i]} Results:")
            try:
                print(model.summary().tables[1])
            except:
                print("모델 요약 출력 실패")
    
    # Stargazer 시도 (옵션) - 단순하고 안전한 버전
    print(f"\nStargazer 표 생성을 시도합니다...")
    stargazer_success = False
    
    try:
        from stargazer.stargazer import Stargazer
        stargazer = Stargazer(models_to_display)
        stargazer.custom_columns(model_names, [1]*len(model_names))
        stargazer.title("Econometric Analysis Results")
        stargazer.show_degrees_of_freedom(False)
        
        print("\n" + "="*100)
        print("                         Stargazer 결과표")
        print("="*100)
        
        # 출력 형식별 처리 (간단하게)
        if output_format.lower() == 'latex':
            latex_output = stargazer.render_latex()
            print(latex_output)
            stargazer_success = True
            
        elif output_format.lower() == 'html':
            html_output = stargazer.render_html()
            print(html_output)
            stargazer_success = True
            
        else:  # text (기본값) - 단순한 방법만 사용
            print("기본 텍스트 형식은 지원하지 않습니다.")
            print("LaTeX 형식으로 출력합니다:")
            print("-" * 80)
            latex_output = stargazer.render_latex()
            print(latex_output)
            print("-" * 80)
            stargazer_success = True
        
        results_dict['stargazer_table'] = stargazer
        results_dict['output_format'] = output_format
        
        if stargazer_success:
            print("? Stargazer 표 생성 완료!")
        
    except ImportError:
        print("? Stargazer 라이브러리가 설치되지 않음")
        print("   설치 방법: pip install stargazer")
    except Exception as e:
        print(f"? Stargazer 오류: {e}")
    
    if not stargazer_success:
        print("?? 위의 수동 생성 표를 참고하세요.")
    
    # 추가: 간소화된 요약표 제공
    print(f"\n" + "="*100)
    print("                           간소화된 최종 요약")
    print("="*100)
    
    try:
        # 핵심 통계량만 간단히 정리
        print(f"{'Model':<20} {'R²':<10} {'Adj R²':<10} {'N':<8} {'Key Variables'}")
        print("-" * 100)
        
        for i, (model, name) in enumerate(zip(models_to_display, model_names)):
            try:
                r2 = getattr(model, 'rsquared', 'N/A')
                adj_r2 = getattr(model, 'rsquared_adj', 'N/A')
                n_obs = int(model.nobs) if hasattr(model, 'nobs') else 'N/A'
                
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
                adj_r2_str = f"{adj_r2:.4f}" if isinstance(adj_r2, (int, float)) else str(adj_r2)
                
                # 주요 계수들의 유의성 확인
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
        
        # GARCH 모델 별도 출력 (시계열의 경우)
        if data_type == 'time_series' and 'garch_results' in results_dict and results_dict['garch_results'] is not None:
            print(f"\n" + "="*100)
            print("                           GARCH(1,1) 모델 결과")
            print("="*100)
            try:
                garch_res = results_dict['garch_results']
                print("평균 방정식 (Mean Equation):")
                print(garch_res.summary().tables[1])
                print("\n분산 방정식 (Variance Equation):")
                print(garch_res.summary().tables[2])
                print("="*100)
            except Exception as e:
                print(f"GARCH 결과 출력 실패: {e}")
        
        # 시계열 검정 결과 요약
        if data_type == 'time_series' and 'time_series_tests' in results_dict:
            print(f"\n" + "="*100)
            print("                           시계열 검정 결과 요약")
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
        print(f"간소화된 요약 생성 실패: {e}")
        
    print("\n? 모든 분석이 완료되었습니다!")
    
    if data_type == 'time_series':
        print("?? 시계열 분석 결과:")
        print("   - 주요 회귀 결과는 위의 '수동 표 생성' 섹션 참고")
        print("   - GARCH 모델 결과는 별도 표시됨")
        print("   - 시계열 검정 결과 요약 확인")
    else:
        print("?? 주요 결과는 위의 '수동 표 생성' 섹션을 참고하세요.")
    
    print("\n" + "="*80)
    print("                        Analysis Complete")
    print("="*80)
    
    return results_dict

print("이 함수는 전통적 계량경제학 종합 분석 함수입니다. ")
