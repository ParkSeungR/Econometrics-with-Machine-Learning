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
from collections import OrderedDict # ��� �񱳸� ���� ���� ���� ��ųʸ� ���

def Run_all_Panel_estimations(data_df: pd.DataFrame, 
                              id_col: str, 
                              year_col: str, 
                              dependent_var: str, 
                              independent_vars: list):
    """
    �־��� �г� �����Ϳ� ���� linearmodels ���̺귯���� �پ��� �г� �������� �����ϰ� ����� ����մϴ�.

    Parameters
    ----------
    data_df : pandas.DataFrame
        �г� �����Ͱ� ���Ե� DataFrame.
    id_col : str
        ��ü(entity) ID�� ��Ÿ���� �÷� �̸�.
    year_col : str
        �ð�(time) ID�� ��Ÿ���� �÷� �̸�.
    dependent_var : str
        ���� ������ �÷� �̸�.
    independent_vars : list
        ���� �������� �÷� �̸� ����Ʈ.

    Notes
    -----
    �� �Լ��� ������ �Ŵ����� ���ø� ������� �� �𵨿� �ʿ��� ���� �������� �����Ͽ� ����մϴ�.
    ����, �Ϻ� ���� ������ Ư�� �𵨿��� ���ܵ� �� �ֽ��ϴ�.
    """

    print("������ �غ� ��...\n")

    # ���� DataFrame ���� (���� ����)
    processed_data = data_df.copy()

    # MultiIndex ����
    # year_col�� Categorical�� ��ȯ���� �ʰ� ���� ���������� �����Ͽ� linearmodels �䱸������ ����
    processed_data = processed_data.set_index([id_col, year_col])

    # ���� ���� ����
    endog = processed_data[dependent_var]

    # �� �𵨿� ���� ���� �������� ���� (�Ŵ��� ���� ���)
    # independent_vars�� 'year_col'�� �������� �ʴ� ���� ���� ���� ����Ʈ�� ����
    # 'year_col'�� �ʿ��� ��� ������ ó���Ͽ� ���� ������ ����

    # �� ������� ������ ��ųʸ�
    results = OrderedDict()

    # --- 2. PooledOLS (�⺻ OLS) ---
    print("----------------------------------------------------------------")
    print("PooledOLS (�յ� OLS) ���� ��...")
    # PooledOLS�� RandomEffects�� 'year_col'�� ���� ������ ����Ͽ� �ð� ���̸� ����
    
    # ���� ���� �������� DataFrame�� ����.
    # MultiIndex���� 'year_col'�� �÷����� �����ͼ� Categorical�� ��ȯ �� ���.
    exog_common = processed_data[independent_vars].copy() # ���� ���� ������
    exog_common[year_col] = pd.Categorical(processed_data.index.get_level_values(year_col)) # year_col�� Categorical �÷����� �߰�

    # sm.add_constant�� ����Ͽ� ����� �߰� �� categorical ���� ó��
    exog_pooled_re = sm.add_constant(exog_common, has_constant='add')
    
    mod_pooled = PooledOLS(endog, exog_pooled_re)
    pooled_res = mod_pooled.fit()
    print(pooled_res)
    results["PooledOLS"] = pooled_res
    print("\n")

    # --- 3. RandomEffects (Ȯ�� ȿ�� ��) ---
    print("----------------------------------------------------------------")
    print("RandomEffects (Ȯ�� ȿ��) ���� ��...")
    # PooledOLS�� ������ exog ���
    mod_re = RandomEffects(endog, exog_pooled_re)
    re_res = mod_re.fit()
    print(re_res)
    results["RandomEffects"] = re_res
    print("\n�л� ���� ���:")
    print(re_res.variance_decomposition)
    print("\ntheta �� (�Ϻ�):")
    print(re_res.theta.head())
    print("\n")

    # --- 4. BetweenOLS (��ü�� OLS) ---
    print("----------------------------------------------------------------")
    print("BetweenOLS (��ü�� OLS) ���� ��...")
    # �Ŵ��� ����: 'expersq' ���� ����. 'year' ������ BetweenOLS���� �ǹ� ����.
    exog_between_vars = [var for var in independent_vars if var != 'expersq']
    exog_between = sm.add_constant(processed_data[exog_between_vars], has_constant='add')
    
    mod_be = BetweenOLS(endog, exog_between)
    be_res = mod_be.fit()
    print(be_res)
    results["BetweenOLS"] = be_res
    print("\n")

    # --- 5. PanelOLS (Fixed Effects - ���� ȿ��) ---
    print("----------------------------------------------------------------")
    print("PanelOLS (���� ȿ�� - Entity Effects) ���� ��...")
    # �Ŵ��� ����: 'expersq', 'union', 'married', 'year' (����)�� ����.
    # �ð� �Һ� ���� ('black', 'hisp', 'educ', 'exper') ����
    
    # fe_base_vars ����: 'expersq', 'union', 'married'
    fe_base_vars = [var for var in independent_vars if var in ["expersq", "union", "married"]]
    
    # 'year_col'�� �÷����� ���Խ��� �ð� ���� ����
    exog_fe_with_year = processed_data[fe_base_vars].copy()
    exog_fe_with_year[year_col] = pd.Categorical(processed_data.index.get_level_values(year_col)) # Categorical�� ��ȯ
    exog_fe = sm.add_constant(exog_fe_with_year, has_constant='add')
    
    mod_fe = PanelOLS(endog, exog_fe, entity_effects=True)
    fe_res = mod_fe.fit()
    print(fe_res)
    results["FixedEffects (Entity)"] = fe_res
    print("\n")

    # --- 6. PanelOLS (Fixed Effects - Entity + Time Effects) ---
    print("----------------------------------------------------------------")
    print("PanelOLS (���� ȿ�� - Entity + Time Effects) ���� ��...")
    # �Ŵ��� ����: 'expersq', 'union', 'married'�� ��� (year ���̴� time_effects�� ��ü)
    exog_fe_te_vars = [var for var in independent_vars if var in ["expersq", "union", "married"]]
    exog_fe_te = sm.add_constant(processed_data[exog_fe_te_vars], has_constant='add') # year_col ����
    mod_fe_te = PanelOLS(endog, exog_fe_te, entity_effects=True, time_effects=True)
    fe_te_res = mod_fe_te.fit()
    print(fe_te_res)
    results["FixedEffects (Entity+Time)"] = fe_te_res
    print("\n")

    # --- 7. PanelOLS (Fixed Effects - Entity + Other Effects) ---
    print("----------------------------------------------------------------")
    print("PanelOLS (���� ȿ�� - Entity + Other Effects (Time)) ���� ��...")
    
    # pd.factorize�� ����Ͽ� �ð� �ֱ⸦ 0-indexed ���� ���̺�� ��ȯ
    time_levels = processed_data.index.get_level_values(year_col)
    factorized_time_ids, _ = pd.factorize(time_levels)
    
    # Series�� ��ȯ�ϰ� MultiIndex�� ����
    internal_time_ids = pd.Series(factorized_time_ids, index=processed_data.index)
    time_ids_df = pd.DataFrame(internal_time_ids, index=processed_data.index, columns=["Other Effect"])
    
    # exog�� ����װ� �Բ� 'expersq', 'union', 'married' ������ ����մϴ�.
    # �Ŵ��� ������ �ٸ� ȿ�� �κп����� �ٸ� ���� ������ ���������, ���⼭�� consistency�� ���� fe_te_vars ���
    try:
        # �Ŵ����� �ؽ�Ʈ("�ð� ȿ���� ����")�� ������ ���� entity_effects=False�� ����
        # �� ������ Ư�� ������ linearmodels���� ����ġ ����ġ ������ ������ �� ����.
        mod_fe_oe = PanelOLS(endog, exog_fe_te, entity_effects=False, other_effects=time_ids_df)
        fe_oe_res = mod_fe_oe.fit()
        print(fe_oe_res)
        results["FixedEffects (Other-Time)"] = fe_oe_res # �� �̸��� ����
    except ValueError as e:
        print(f"FixedEffects (Other-Time) �� ���� �� ���� �߻�: {e}")
        print("�� ���� 'entity_effects=False'�� 'other_effects' ���տ��� ������ ���� �Ǵ� ȿ�� ��ø�� ���õ� ������ ���� �ǳʶݴϴ�.")
        print("�ٸ� ������ ����� ���������� ��µ˴ϴ�.")
    print("\n")
    
    # --- 8. FirstDifferenceOLS (ù ��° ���� OLS) ---
    print("----------------------------------------------------------------")
    print("FirstDifferenceOLS (ù ��° ����) ���� ��...")
    # �Ŵ��� ����: 'exper', 'expersq', 'union', 'married' ���
    # �ð� �Һ� ���� ('black', 'hisp', 'educ') �� year ���̴� ù ��° ���п��� ���ܵ�.
    fd_vars = [var for var in independent_vars if var in ["exper", "expersq", "union", "married"]]
    # FirstDifferenceOLS�� ������� �ڵ����� ó���ϰų� �𵨿��� ������.
    # sm.add_constant ���� ���� ���� ���� DataFrame�� ����.
    exog_fd = processed_data[fd_vars]
    
    mod_fd = FirstDifferenceOLS(endog, exog_fd)
    fd_res = mod_fd.fit()
    print(fd_res)
    results["FirstDifferenceOLS"] = fd_res
    print("\n")

    # --- 9. Covariance options for PooledOLS ---
    print("----------------------------------------------------------------")
    print("PooledOLS�� Robust, Clustered Covariance ���� ��...")
    # `exog_pooled_re` (const ����) ���
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

    # Covariance option ��� ��
    print("\n--- PooledOLS Covariance Options �� ---")
    cov_results_comp = OrderedDict()
    cov_results_comp["Robust"] = robust_res
    cov_results_comp["Entity Clustered"] = clust_entity_res
    cov_results_comp["Entity-Time Clustered"] = clust_entity_time_res
    print(compare(cov_results_comp))
    print("\n")

    # --- 10. Hausman Test (FE vs RE) ---
    print("----------------------------------------------------------------")
    print("Hausman Test (Fixed Effects vs Random Effects) ���� ��...")
    if "FixedEffects (Entity)" in results and "RandomEffects" in results:
        try:
            # linearmodels�� compare �Լ��� FE�� RE �� �� �ڵ����� Hausman test�� ����
            hausman_comp = compare({"FE": results["FixedEffects (Entity)"], "RE": results["RandomEffects"]})
            print(hausman_comp)
            print("\n�� ����� 'Hausman' ���ǿ��� Hausman Test ��跮 �� p-���� Ȯ���� �� �ֽ��ϴ�.")
        except Exception as e:
            print(f"Hausman Test ���� �� ���� �߻�: {e}")
            print("Hausman Test�� FE�� RE �� ���� ��� �� ���л� ��� ȣȯ���� ���� ������ �� �ֽ��ϴ�.")
    else:
        print("Hausman Test�� �����ϱ� ���� Fixed Effects (Entity) �Ǵ� Random Effects �� ����� �����ϴ�.")
    print("\n")

    # --- ��� �� ��� ��� �� ---
    print("----------------------------------------------------------------")
    print("��� �г� ���� �� ��� ��� �� ��...")
    # `results` ��ųʸ����� ������ �𵨵��� �����Ͽ� ��
    # �Ŵ��� ���ÿ� �����ϰ� �Ϻ� �𵨸� �����Ͽ� ��
    selected_for_comparison = OrderedDict()
    if "BetweenOLS" in results: selected_for_comparison["BE"] = results["BetweenOLS"]
    if "RandomEffects" in results: selected_for_comparison["RE"] = results["RandomEffects"]
    if "PooledOLS" in results: selected_for_comparison["Pooled"] = results["PooledOLS"]
    if "FixedEffects (Entity)" in results: selected_for_comparison["FE (Entity)"] = results["FixedEffects (Entity)"]
    if "FixedEffects (Entity+Time)" in results: selected_for_comparison["FE (E+T)"] = results["FixedEffects (Entity+Time)"]
    # "FixedEffects (Other-Time)" ���� ���������� ����Ǿ��� ���� �񱳿� ����
    if "FixedEffects (Other-Time)" in results: selected_for_comparison["FE (Other-Time)"] = results["FixedEffects (Other-Time)"] 
    if "FirstDifferenceOLS" in results: selected_for_comparison["FD"] = results["FirstDifferenceOLS"]

    if selected_for_comparison:
        print(compare(selected_for_comparison))
    else:
        print("���� �� ����� �����ϴ�.")
    print("----------------------------------------------------------------")
    print("��� �г� ������ ���� �Ϸ�!")