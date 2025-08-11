conda update scipy numpy


import numpy as np
import pandas as pd
from scipy.linalg import matrix_rank

def check_identification_complete(equations_dict, data=None):
    """
    연립방정식의 위수조건과 계수조건을 모두 확인하는 함수
    
    Parameters:
    equations_dict: dict - linearmodels 형식의 방정식
    data: pandas.DataFrame (optional) - 계수조건 확인용 데이터
    
    Returns:
    dict: 위수조건과 계수조건 결과
    """
    
    # 1. 변수 파싱 (이전과 동일)
    all_variables = set()
    endogenous_vars = set()
    
    for eq_name, formula in equations_dict.items():
        dependent = formula.split('~')[0].strip()
        endogenous_vars.add(dependent)
        
        rhs = formula.split('~')[1]
        
        if '[' in rhs and ']' in rhs:
            endog_part = rhs.split('[')[1].split(']')[0]
            endog_var = endog_part.split('~')[0].strip()
            endogenous_vars.add(endog_var)
            
            instruments = endog_part.split('~')[1].replace('+', ' ').split()
            for var in instruments:
                var = var.strip()
                if var not in ['1', '']:
                    all_variables.add(var)
        
        exog_part = rhs.split('[')[0] if '[' in rhs else rhs
        exog_vars = exog_part.replace('+', ' ').split()
        for var in exog_vars:
            var = var.strip()
            if var not in ['1', '']:
                all_variables.add(var)
    
    exogenous_vars = all_variables - endogenous_vars
    
    # 2. 각 방정식별 분석
    results = {}
    
    for eq_name, formula in equations_dict.items():
        dependent = formula.split('~')[0].strip()
        rhs = formula.split('~')[1]
        
        # 변수 분류
        included_endogenous = [dependent]
        included_exogenous = []
        instruments = []
        
        if '[' in rhs and ']' in rhs:
            endog_part = rhs.split('[')[1].split(']')[0]
            endog_var = endog_part.split('~')[0].strip()
            included_endogenous.append(endog_var)
            
            # 도구변수 추출
            instr_vars = endog_part.split('~')[1].replace('+', ' ').split()
            for var in instr_vars:
                var = var.strip()
                if var not in ['1', '']:
                    instruments.append(var)
            
            exog_part = rhs.split('[')[0]
        else:
            exog_part = rhs
        
        if exog_part.strip():
            exog_list = exog_part.replace('+', ' ').split()
            for var in exog_list:
                var = var.strip()
                if var in exogenous_vars:
                    included_exogenous.append(var)
        
        # 위수조건 (Order Condition) 검사
        total_exog = len(exogenous_vars)
        included_exog_count = len(included_exogenous)
        excluded_exog_count = total_exog - included_exog_count
        included_endog_count = len(included_endogenous)
        
        order_requirement = included_endog_count - 1
        order_condition = excluded_exog_count >= order_requirement
        
        # 식별 상태
        if excluded_exog_count < order_requirement:
            status = "Underidentified"
        elif excluded_exog_count == order_requirement:
            status = "Exactly identified"
        else:
            status = "Overidentified"
        
        # 계수조건 (Rank Condition) 검사
        rank_condition = None
        rank_details = "데이터가 필요함"
        
        if data is not None:
            try:
                rank_condition, rank_details = check_rank_condition(
                    data, included_endogenous, included_exogenous, 
                    instruments, exogenous_vars
                )
            except Exception as e:
                rank_details = f"계수조건 확인 오류: {str(e)}"
        
        results[eq_name] = {
            'order_condition': {
                'satisfied': order_condition,
                'excluded_exog': excluded_exog_count,
                'required': order_requirement,
                'status': status
            },
            'rank_condition': {
                'satisfied': rank_condition,
                'details': rank_details
            },
            'variables': {
                'included_endogenous': included_endogenous,
                'included_exogenous': included_exogenous,
                'instruments': instruments
            }
        }
    
    return results

def check_rank_condition(data, included_endogenous, included_exogenous, instruments, all_exogenous):
    """
    계수조건 확인 (실제 데이터 필요)
    """
    try:
        # 축약형 방정식을 위한 행렬 구성
        # Z: 모든 외생변수 (도구변수 포함)
        # X: 이 방정식의 설명변수들
        
        all_exog_list = list(all_exogenous)
        
        # 데이터에서 필요한 변수들이 있는지 확인
        missing_vars = set(all_exog_list) - set(data.columns)
        if missing_vars:
            return None, f"데이터에 없는 변수: {missing_vars}"
        
        # 축약형 계수 행렬 구성을 위한 간단한 근사
        Z = data[all_exog_list].values
        
        # 제외된 변수들로 부분 행렬 구성
        excluded_vars = list(set(all_exog_list) - set(included_exogenous))
        
        if len(excluded_vars) == 0:
            return False, "제외된 외생변수가 없음"
        
        Z_excluded = data[excluded_vars].values
        
        # Rank 계산
        required_rank = len(included_endogenous) - 1
        actual_rank = matrix_rank(Z_excluded)
        
        rank_satisfied = actual_rank >= required_rank
        
        details = f"행렬 rank: {actual_rank}, 필요 rank: {required_rank}"
        
        return rank_satisfied, details
        
    except Exception as e:
        return None, f"계수조건 계산 오류: {str(e)}"

def print_complete_identification_results(results):
    """위수조건과 계수조건 결과를 모두 출력"""
    print("=" * 70)
    print("연립방정식 완전 식별조건 검사 결과")
    print("=" * 70)
    
    for eq_name, result in results.items():
        print(f"\n방정식: {eq_name}")
        print("-" * 50)
        
        # 위수조건 결과
        order = result['order_condition']
        order_symbol = "✓" if order['satisfied'] else "✗"
        print(f"위수조건 (Order Condition): {order_symbol}")
        print(f"  - 제외된 외생변수: {order['excluded_exog']} (필요: {order['required']})")
        print(f"  - 상태: {order['status']}")
        
        # 계수조건 결과
        rank = result['rank_condition']
        if rank['satisfied'] is not None:
            rank_symbol = "✓" if rank['satisfied'] else "✗"
            print(f"계수조건 (Rank Condition): {rank_symbol}")
        else:
            print(f"계수조건 (Rank Condition): ?")
        print(f"  - 세부사항: {rank['details']}")
        
        # 변수 정보
        vars_info = result['variables']
        print(f"  - 내생변수: {vars_info['included_endogenous']}")
        print(f"  - 외생변수: {vars_info['included_exogenous']}")
        print(f"  - 도구변수: {vars_info['instruments']}")
    
    # 전체 시스템 결론
    print(f"\n{'='*50}")
    all_order_ok = all(r['order_condition']['satisfied'] for r in results.values())
    all_rank_ok = all(r['rank_condition']['satisfied'] for r in results.values() 
                     if r['rank_condition']['satisfied'] is not None)
    
    print(f"위수조건: {'모두 만족' if all_order_ok else '일부 불만족'}")
    
    rank_results = [r['rank_condition']['satisfied'] for r in results.values()]
    if None not in rank_results:
        print(f"계수조건: {'모두 만족' if all_rank_ok else '일부 불만족'}")
        final_status = "완전 식별 가능" if (all_order_ok and all_rank_ok) else "식별 문제 있음"
    else:
        print(f"계수조건: 데이터 필요")
        final_status = "위수조건만 확인됨" if all_order_ok else "위수조건 불만족"
    
    print(f"최종 결론: {final_status}")

"""
# 사용 예제
if __name__ == "__main__":
    
    # 예제 1: 데이터 없이 위수조건만 확인
    print("예제 1: 위수조건만 확인 (데이터 없음)")
    equations = {
        "hours": "hours ~ educ + age + kidslt6 + nwifeinc + [lwage ~ exper + expersq]",
        "lwage": "lwage ~ educ + exper + expersq + [hours ~ age + kidslt6 + nwifeinc]"
    }
    
    results = check_identification_complete(equations)
    print_complete_identification_results(results)
    
    print("\n" + "="*70)
    
    # 예제 2: 가상 데이터로 계수조건까지 확인
    print("예제 2: 가상 데이터로 계수조건까지 확인")
    
    # 가상 데이터 생성
    np.random.seed(42)
    n = 100
    fake_data = pd.DataFrame({
        'hours': np.random.normal(40, 10, n),
        'lwage': np.random.normal(2.5, 0.5, n),
        'educ': np.random.randint(8, 20, n),
        'age': np.random.randint(20, 65, n),
        'kidslt6': np.random.randint(0, 4, n),
        'nwifeinc': np.random.normal(30, 15, n),
        'exper': np.random.randint(0, 40, n),
        'expersq': np.random.randint(0, 40, n)**2  # 사실 exper^2이어야 하지만 예제용
    })
    
    results_with_data = check_identification_complete(equations, fake_data)
    print_complete_identification_results(results_with_data)
"""

import linearmodels as lm
import statsmodels.api as sm
from linearmodels import IV2SLS, IV3SLS, SUR, IVSystemGMM
from linearmodels.datasets import mroz

# 데이터 읽어오기
data = mroz.load()
data = data[["hours", "educ", "age", "kidslt6", "nwifeinc", "lwage", "exper", "expersq"]]
data = data.dropna()    
    
    
    
equations = {
    "hours": "hours ~ educ + age + kidslt6 + nwifeinc + [lwage ~ exper + expersq]",
    "lwage": "lwage ~ educ + exper + expersq + [hours ~ age + kidslt6 + nwifeinc]"
}

results = check_identification(equations)
print_identification_summary(results)

