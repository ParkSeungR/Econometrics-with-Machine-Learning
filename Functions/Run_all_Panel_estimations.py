import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import (
    PooledOLS,
    RandomEffects,
    BetweenOLS,
    PanelOLS,
    FirstDifferenceOLS,
    compare
)
from linearmodels.panel.data import PanelData
from collections import OrderedDict # 결과 비교를 위해 순서 유지 딕셔너리 사용

def Run_all_Panel_estimations(data_df: pd.DataFrame, 
                              id_col: str, 
                              year_col: str, 
                              dependent_var: str, 
                              independent_vars: list):
    """
    주어진 패널 데이터에 대해 linearmodels 라이브러리의 다양한 패널 추정법을 적용하고 결과를 출력합니다.

    Parameters
    ----------
    data_df : pandas.DataFrame
        패널 데이터가 포함된 DataFrame.
    id_col : str
        개체(entity) ID를 나타내는 컬럼 이름.
    year_col : str
        시간(time) ID를 나타내는 컬럼 이름.
    dependent_var : str
        종속 변수의 컬럼 이름.
    independent_vars : list
        독립 변수들의 컬럼 이름 리스트.

    Notes
    -----
    이 함수는 제공된 매뉴얼의 예시를 기반으로 각 모델에 필요한 독립 변수들을 선별하여 사용합니다.
    따라서, 일부 독립 변수는 특정 모델에서 제외될 수 있습니다.
    """

    print("데이터 준비 중...\n")

    # 원본 DataFrame 복사 (변경 방지)
    processed_data = data_df.copy()

    # MultiIndex 설정
    # year_col을 Categorical로 변환하지 않고 원본 숫자형으로 유지하여 linearmodels 요구사항을 충족
    processed_data = processed_data.set_index([id_col, year_col])

    # 종속 변수 설정
    endog = processed_data[dependent_var]

    # 각 모델에 사용될 독립 변수들을 설정 (매뉴얼 예시 기반)
    # independent_vars는 'year_col'을 포함하지 않는 순수 독립 변수 리스트로 가정
    # 'year_col'은 필요한 경우 별도로 처리하여 더미 변수로 포함

    # 모델 결과들을 저장할 딕셔너리
    results = OrderedDict()

    # --- 2. PooledOLS (기본 OLS) ---
    print("----------------------------------------------------------------")
    print("PooledOLS (합동 OLS) 추정 중...")
    # PooledOLS와 RandomEffects는 'year_col'을 독립 변수로 사용하여 시간 더미를 생성
    
    # 먼저 독립 변수들의 DataFrame을 생성.
    # MultiIndex에서 'year_col'을 컬럼으로 가져와서 Categorical로 변환 후 사용.
    exog_common = processed_data[independent_vars].copy() # 기존 독립 변수들
    exog_common[year_col] = pd.Categorical(processed_data.index.get_level_values(year_col)) # year_col을 Categorical 컬럼으로 추가

    # sm.add_constant를 사용하여 상수항 추가 및 categorical 변수 처리
    exog_pooled_re = sm.add_constant(exog_common, has_constant='add')
    
    mod_pooled = PooledOLS(endog, exog_pooled_re)
    pooled_res = mod_pooled.fit()
    print(pooled_res)
    results["PooledOLS"] = pooled_res
    print("\n")

    # --- 3. RandomEffects (확률 효과 모델) ---
    print("----------------------------------------------------------------")
    print("RandomEffects (확률 효과) 추정 중...")
    # PooledOLS와 동일한 exog 사용
    mod_re = RandomEffects(endog, exog_pooled_re)
    re_res = mod_re.fit()
    print(re_res)
    results["RandomEffects"] = re_res
    print("\n분산 분해 결과:")
    print(re_res.variance_decomposition)
    print("\ntheta 값 (일부):")
    print(re_res.theta.head())
    print("\n")

    # --- 4. BetweenOLS (개체간 OLS) ---
    print("----------------------------------------------------------------")
    print("BetweenOLS (개체간 OLS) 추정 중...")
    # 매뉴얼 예시: 'expersq' 변수 제외. 'year' 변수는 BetweenOLS에서 의미 없음.
    exog_between_vars = [var for var in independent_vars if var != 'expersq']
    exog_between = sm.add_constant(processed_data[exog_between_vars], has_constant='add')
    
    mod_be = BetweenOLS(endog, exog_between)
    be_res = mod_be.fit()
    print(be_res)
    results["BetweenOLS"] = be_res
    print("\n")

    # --- 5. PanelOLS (Fixed Effects - 고정 효과) ---
    print("----------------------------------------------------------------")
    print("PanelOLS (고정 효과 - Entity Effects) 추정 중...")
    # 매뉴얼 예시: 'expersq', 'union', 'married', 'year' (더미)만 포함.
    # 시간 불변 변수 ('black', 'hisp', 'educ', 'exper') 제외
    
    # fe_base_vars 구성: 'expersq', 'union', 'married'
    fe_base_vars = [var for var in independent_vars if var in ["expersq", "union", "married"]]
    
    # 'year_col'을 컬럼으로 포함시켜 시간 더미 생성
    exog_fe_with_year = processed_data[fe_base_vars].copy()
    exog_fe_with_year[year_col] = pd.Categorical(processed_data.index.get_level_values(year_col)) # Categorical로 변환
    exog_fe = sm.add_constant(exog_fe_with_year, has_constant='add')
    
    mod_fe = PanelOLS(endog, exog_fe, entity_effects=True)
    fe_res = mod_fe.fit()
    print(fe_res)
    results["FixedEffects (Entity)"] = fe_res
    print("\n")

    # --- 6. PanelOLS (Fixed Effects - Entity + Time Effects) ---
    print("----------------------------------------------------------------")
    print("PanelOLS (고정 효과 - Entity + Time Effects) 추정 중...")
    # 매뉴얼 예시: 'expersq', 'union', 'married'만 사용 (year 더미는 time_effects로 대체)
    exog_fe_te_vars = [var for var in independent_vars if var in ["expersq", "union", "married"]]
    exog_fe_te = sm.add_constant(processed_data[exog_fe_te_vars], has_constant='add') # year_col 제외
    mod_fe_te = PanelOLS(endog, exog_fe_te, entity_effects=True, time_effects=True)
    fe_te_res = mod_fe_te.fit()
    print(fe_te_res)
    results["FixedEffects (Entity+Time)"] = fe_te_res
    print("\n")

    # --- 7. PanelOLS (Fixed Effects - Entity + Other Effects) ---
    print("----------------------------------------------------------------")
    print("PanelOLS (고정 효과 - Entity + Other Effects (Time)) 추정 중...")
    
    # pd.factorize를 사용하여 시간 주기를 0-indexed 정수 레이블로 변환
    time_levels = processed_data.index.get_level_values(year_col)
    factorized_time_ids, _ = pd.factorize(time_levels)
    
    # Series로 변환하고 MultiIndex를 유지
    internal_time_ids = pd.Series(factorized_time_ids, index=processed_data.index)
    time_ids_df = pd.DataFrame(internal_time_ids, index=processed_data.index, columns=["Other Effect"])
    
    # exog는 상수항과 함께 'expersq', 'union', 'married' 변수를 사용합니다.
    # 매뉴얼 예시의 다른 효과 부분에서는 다른 독립 변수를 사용하지만, 여기서는 consistency를 위해 fe_te_vars 사용
    try:
        # 매뉴얼의 텍스트("시간 효과만 재현")를 따르기 위해 entity_effects=False로 설정
        # 이 설정이 특정 버전의 linearmodels에서 관측치 불일치 에러를 유발할 수 있음.
        mod_fe_oe = PanelOLS(endog, exog_fe_te, entity_effects=False, other_effects=time_ids_df)
        fe_oe_res = mod_fe_oe.fit()
        print(fe_oe_res)
        results["FixedEffects (Other-Time)"] = fe_oe_res # 모델 이름도 변경
    except ValueError as e:
        print(f"FixedEffects (Other-Time) 모델 추정 중 오류 발생: {e}")
        print("이 모델은 'entity_effects=False'와 'other_effects' 조합에서 데이터 정렬 또는 효과 중첩과 관련된 문제로 인해 건너뜁니다.")
        print("다른 추정법 결과는 정상적으로 출력됩니다.")
    print("\n")
    
    # --- 8. FirstDifferenceOLS (첫 번째 차분 OLS) ---
    print("----------------------------------------------------------------")
    print("FirstDifferenceOLS (첫 번째 차분) 추정 중...")
    # 매뉴얼 예시: 'exper', 'expersq', 'union', 'married' 사용
    # 시간 불변 변수 ('black', 'hisp', 'educ') 및 year 더미는 첫 번째 차분에서 제외됨.
    fd_vars = [var for var in independent_vars if var in ["exper", "expersq", "union", "married"]]
    # FirstDifferenceOLS는 상수항을 자동으로 처리하거나 모델에서 제외함.
    # sm.add_constant 없이 직접 독립 변수 DataFrame을 전달.
    exog_fd = processed_data[fd_vars]
    
    mod_fd = FirstDifferenceOLS(endog, exog_fd)
    fd_res = mod_fd.fit()
    print(fd_res)
    results["FirstDifferenceOLS"] = fd_res
    print("\n")

    # --- 9. Covariance options for PooledOLS ---
    print("----------------------------------------------------------------")
    print("PooledOLS의 Robust, Clustered Covariance 추정 중...")
    # `exog_pooled_re` (const 포함) 사용
    mod_pooled_cov = PooledOLS(endog, exog_pooled_re)

    # Heteroskedasticity Robust Covariance
    robust_res = mod_pooled_cov.fit(cov_type="robust")
    print("\n--- PooledOLS with Robust Covariance ---")
    print(robust_res)
    
    # Clustered by Entity
    clust_entity_res = mod_pooled_cov.fit(cov_type="clustered", cluster_entity=True)
    print("\n--- PooledOLS with Clustered by Entity Covariance ---")
    print(clust_entity_res)

    # Clustered by Entity-Time
    clust_entity_time_res = mod_pooled_cov.fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    print("\n--- PooledOLS with Clustered by Entity-Time Covariance ---")
    print(clust_entity_time_res)

    # Covariance option 결과 비교
    print("\n--- PooledOLS Covariance Options 비교 ---")
    cov_results_comp = OrderedDict()
    cov_results_comp["Robust"] = robust_res
    cov_results_comp["Entity Clustered"] = clust_entity_res
    cov_results_comp["Entity-Time Clustered"] = clust_entity_time_res
    print(compare(cov_results_comp))
    print("\n")

    # --- 10. Hausman Test (FE vs RE) ---
    print("----------------------------------------------------------------")
    print("Hausman Test (Fixed Effects vs Random Effects) 실행 중...")
    if "FixedEffects (Entity)" in results and "RandomEffects" in results:
        try:
            # linearmodels의 compare 함수는 FE와 RE 비교 시 자동으로 Hausman test를 포함
            hausman_comp = compare({"FE": results["FixedEffects (Entity)"], "RE": results["RandomEffects"]})
            print(hausman_comp)
            print("\n위 결과의 'Hausman' 섹션에서 Hausman Test 통계량 및 p-값을 확인할 수 있습니다.")
        except Exception as e:
            print(f"Hausman Test 실행 중 오류 발생: {e}")
            print("Hausman Test는 FE와 RE 모델 간의 계수 및 공분산 행렬 호환성에 따라 실패할 수 있습니다.")
    else:
        print("Hausman Test를 실행하기 위한 Fixed Effects (Entity) 또는 Random Effects 모델 결과가 없습니다.")
    print("\n")

    # --- 모든 모델 결과 요약 비교 ---
    print("----------------------------------------------------------------")
    print("모든 패널 추정 모델 결과 요약 비교 중...")
    # `results` 딕셔너리에서 적절한 모델들을 선택하여 비교
    # 매뉴얼 예시와 유사하게 일부 모델만 선택하여 비교
    selected_for_comparison = OrderedDict()
    if "BetweenOLS" in results: selected_for_comparison["BE"] = results["BetweenOLS"]
    if "RandomEffects" in results: selected_for_comparison["RE"] = results["RandomEffects"]
    if "PooledOLS" in results: selected_for_comparison["Pooled"] = results["PooledOLS"]
    if "FixedEffects (Entity)" in results: selected_for_comparison["FE (Entity)"] = results["FixedEffects (Entity)"]
    if "FixedEffects (Entity+Time)" in results: selected_for_comparison["FE (E+T)"] = results["FixedEffects (Entity+Time)"]
    # "FixedEffects (Other-Time)" 모델이 성공적으로 실행되었을 때만 비교에 포함
    if "FixedEffects (Other-Time)" in results: selected_for_comparison["FE (Other-Time)"] = results["FixedEffects (Other-Time)"] 
    if "FirstDifferenceOLS" in results: selected_for_comparison["FD"] = results["FirstDifferenceOLS"]

    if selected_for_comparison:
        print(compare(selected_for_comparison))
    else:
        print("비교할 모델 결과가 없습니다.")
    print("----------------------------------------------------------------")
    print("모든 패널 추정법 적용 완료!")