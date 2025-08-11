import numpy as np
import pandas as pd
import pulp
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDEA:
    """
    통합 DEA 분석 클래스 - 일반화된 버전
    데이터프레임과 변수명을 사용하여 분석
    """
    
    def __init__(self, data: pd.DataFrame, 
                 input_vars: List[str], 
                 output_vars: List[str],
                 dmu_var: str = None):
        """
        Parameters:
        data: 전체 데이터프레임
        input_vars: 입력 변수 컬럼명 리스트
        output_vars: 출력 변수 컬럼명 리스트  
        dmu_var: DMU 식별 컬럼명 (없으면 인덱스 사용)
        """
        self.data = data.copy()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.dmu_var = dmu_var
        
        # 입력/출력 데이터 추출
        self.inputs = self.data[input_vars].values.astype(float)
        self.outputs = self.data[output_vars].values.astype(float)
        
        self.n_dmu = self.inputs.shape[0]
        self.n_inputs = self.inputs.shape[1] 
        self.n_outputs = self.outputs.shape[1]
        
        # DMU 이름 설정
        if dmu_var and dmu_var in data.columns:
            self.dmu_names = data[dmu_var].astype(str).tolist()
        else:
            self.dmu_names = [f"DMU_{i+1}" for i in range(self.n_dmu)]
    
    def solve_dea(self, model_type: str = 'BCC', orientation: str = 'input') -> Tuple[np.ndarray, np.ndarray]:
        """
        DEA 모델 해결
        
        Parameters:
        model_type: 'CCR' 또는 'BCC'
        orientation: 'input' 또는 'output'
        
        Returns:
        efficiency_scores: 효율성 점수
        lambdas: 가중치 행렬
        """
        efficiency_scores = np.zeros(self.n_dmu)
        lambdas = np.zeros((self.n_dmu, self.n_dmu))
        
        for j0 in range(self.n_dmu):
            if orientation == 'input':
                prob = pulp.LpProblem(f"DEA_{model_type}_{j0}", pulp.LpMinimize)
                theta = pulp.LpVariable("theta", lowBound=0)
                objective = theta
            else:  # output orientation
                prob = pulp.LpProblem(f"DEA_{model_type}_{j0}", pulp.LpMaximize)
                phi = pulp.LpVariable("phi", lowBound=0)
                objective = phi
            
            lambda_vars = [pulp.LpVariable(f"lambda_{j}", lowBound=0) 
                          for j in range(self.n_dmu)]
            
            # 목적함수
            prob += objective
            
            # 제약조건
            if orientation == 'input':
                # 입력 제약
                for i in range(self.n_inputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.inputs[j, i] 
                                       for j in range(self.n_dmu)]) 
                            <= theta * self.inputs[j0, i])
                
                # 출력 제약
                for r in range(self.n_outputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.outputs[j, r] 
                                       for j in range(self.n_dmu)]) 
                            >= self.outputs[j0, r])
            else:  # output orientation
                # 입력 제약
                for i in range(self.n_inputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.inputs[j, i] 
                                       for j in range(self.n_dmu)]) 
                            <= self.inputs[j0, i])
                
                # 출력 제약
                for r in range(self.n_outputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.outputs[j, r] 
                                       for j in range(self.n_dmu)]) 
                            >= phi * self.outputs[j0, r])
            
            # BCC 모델의 볼록성 제약
            if model_type == 'BCC':
                prob += pulp.lpSum(lambda_vars) == 1
            
            # 문제 해결
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # 결과 저장
            if orientation == 'input':
                efficiency_scores[j0] = pulp.value(theta)
            else:
                efficiency_scores[j0] = 1.0 / pulp.value(phi) if pulp.value(phi) > 0 else 0
            
            for j in range(self.n_dmu):
                lambdas[j0, j] = pulp.value(lambda_vars[j]) if pulp.value(lambda_vars[j]) else 0
        
        return efficiency_scores, lambdas
    
    def calculate_slacks(self, efficiency_scores: np.ndarray, lambdas: np.ndarray, 
                        orientation: str = 'input') -> Tuple[np.ndarray, np.ndarray]:
        """슬랙 계산"""
        input_slacks = np.zeros((self.n_dmu, self.n_inputs))
        output_slacks = np.zeros((self.n_dmu, self.n_outputs))
        
        for j0 in range(self.n_dmu):
            # 투영점 계산
            projected_inputs = np.sum(lambdas[j0, :].reshape(-1, 1) * self.inputs, axis=0)
            projected_outputs = np.sum(lambdas[j0, :].reshape(-1, 1) * self.outputs, axis=0)
            
            if orientation == 'input':
                input_slacks[j0] = efficiency_scores[j0] * self.inputs[j0] - projected_inputs
                output_slacks[j0] = projected_outputs - self.outputs[j0]
            else:  # output orientation
                input_slacks[j0] = self.inputs[j0] - projected_inputs
                output_slacks[j0] = projected_outputs - efficiency_scores[j0] * self.outputs[j0]
        
        return np.maximum(input_slacks, 0), np.maximum(output_slacks, 0)
    
    def get_reference_set(self, lambdas: np.ndarray, threshold: float = 1e-6) -> Dict[str, List[str]]:
        """참조집합 계산"""
        reference_sets = {}
        for i in range(self.n_dmu):
            references = []
            for j in range(self.n_dmu):
                if lambdas[i, j] > threshold:
                    references.append(self.dmu_names[j])
            reference_sets[self.dmu_names[i]] = references
        return reference_sets
    
    def sensitivity_analysis(self, dmu_index: int, model_type: str = 'BCC', 
                           orientation: str = 'input', 
                           change_range: Tuple[float, float] = (-0.2, 0.2), 
                           n_points: int = 21) -> Dict[str, np.ndarray]:
        """민감도 분석"""
        changes = np.linspace(change_range[0], change_range[1], n_points)
        original_inputs = self.inputs.copy()
        original_outputs = self.outputs.copy()
        
        input_sensitivity = np.zeros((self.n_inputs, n_points))
        output_sensitivity = np.zeros((self.n_outputs, n_points))
        
        print(f"   ?? 민감도 분석 진행 중...")
        
        # 입력 변수 민감도
        for i in range(self.n_inputs):
            for j, change in enumerate(changes):
                # 임시로 입력값 변경
                temp_inputs = self.inputs.copy()
                temp_inputs[dmu_index, i] = original_inputs[dmu_index, i] * (1 + change)
                
                # 임시 DEA 객체 생성하여 분석
                temp_dea = ComprehensiveDEA(pd.DataFrame(), [], [])
                temp_dea.inputs = temp_inputs
                temp_dea.outputs = self.outputs
                temp_dea.n_dmu = self.n_dmu
                temp_dea.n_inputs = self.n_inputs
                temp_dea.n_outputs = self.n_outputs
                temp_dea.dmu_names = self.dmu_names
                
                scores, _ = temp_dea.solve_dea(model_type, orientation)
                input_sensitivity[i, j] = scores[dmu_index]
        
        # 출력 변수 민감도
        for i in range(self.n_outputs):
            for j, change in enumerate(changes):
                # 임시로 출력값 변경
                temp_outputs = self.outputs.copy()
                temp_outputs[dmu_index, i] = original_outputs[dmu_index, i] * (1 + change)
                
                # 임시 DEA 객체 생성하여 분석
                temp_dea = ComprehensiveDEA(pd.DataFrame(), [], [])
                temp_dea.inputs = self.inputs
                temp_dea.outputs = temp_outputs
                temp_dea.n_dmu = self.n_dmu
                temp_dea.n_inputs = self.n_inputs
                temp_dea.n_outputs = self.n_outputs
                temp_dea.dmu_names = self.dmu_names
                
                scores, _ = temp_dea.solve_dea(model_type, orientation)
                output_sensitivity[i, j] = scores[dmu_index]
        
        print(f"   ? 민감도 분석 완료")
        
        return {
            'changes': changes,
            'input_sensitivity': input_sensitivity,
            'output_sensitivity': output_sensitivity
        }

def Run_Comprehensive_DEA_analysis(data: Union[pd.DataFrame, np.ndarray], 
                              input_vars: Union[List[str], List[int]] = None,
                              output_vars: Union[List[str], List[int]] = None,
                              dmu_var: str = None,
                              input_data: np.ndarray = None,  # 역호환성
                              output_data: np.ndarray = None,  # 역호환성
                              dmu_names: List[str] = None,   # 역호환성
                              orientation: str = 'input',
                              sensitivity_dmu: Union[int, str] = None,
                              plot_results: bool = True,
                              models: List[str] = ['CCR', 'BCC'],
                              change_range: Tuple[float, float] = (-0.2, 0.2)) -> Dict:
    """
    통합 DEA 분석 함수 - 일반화된 버전
    
    Parameters:
    data: 데이터프레임 또는 numpy 배열
    input_vars: 입력 변수 컬럼명 리스트 (DataFrame 사용시) 또는 인덱스 (배열 사용시)
    output_vars: 출력 변수 컬럼명 리스트 (DataFrame 사용시) 또는 인덱스 (배열 사용시)
    dmu_var: DMU 식별 컬럼명 (DataFrame 사용시)
    input_data: 입력 데이터 배열 (역호환성을 위해 유지)
    output_data: 출력 데이터 배열 (역호환성을 위해 유지)
    dmu_names: DMU 이름 리스트 (역호환성을 위해 유지)
    orientation: 'input' 또는 'output'
    sensitivity_dmu: 민감도 분석할 DMU (인덱스 또는 이름)
    plot_results: 결과 시각화 여부
    models: 분석할 모델 리스트 ['CCR', 'BCC']
    change_range: 민감도 분석 변화 범위
    
    Returns:
    분석 결과 딕셔너리
    
    Usage Examples:
    
    # 방법 1: 데이터프레임 사용 (권장)
    results = comprehensive_dea_analysis(
        data=df,
        input_vars=['employees', 'operating_cost'],
        output_vars=['deposits', 'loans'],
        dmu_var='bank_name',
        orientation='input'
    )
    
    # 방법 2: 배열 사용 (역호환성)
    results = comprehensive_dea_analysis(
        input_data=input_array,
        output_data=output_array,
        dmu_names=name_list,
        orientation='input'
    )
    """
    
    print("="*80)
    print("                     통합 DEA 분석 시스템 (일반화 버전)")
    print("="*80)
    
    # 입력 데이터 처리 및 검증
    if input_data is not None and output_data is not None:
        # 역호환성: 기존 방식
        print("?? 데이터 입력 방식: 배열 (역호환 모드)")
        inputs = np.array(input_data, dtype=float)
        outputs = np.array(output_data, dtype=float)
        
        if dmu_names is None:
            dmu_names = [f"DMU_{i+1}" for i in range(inputs.shape[0])]
        
        # 임시 데이터프레임 생성
        df = pd.DataFrame(inputs, columns=[f"input_{i+1}" for i in range(inputs.shape[1])])
        df = pd.concat([df, pd.DataFrame(outputs, columns=[f"output_{i+1}" for i in range(outputs.shape[1])])], axis=1)
        df['dmu_name'] = dmu_names
        
        input_vars = [f"input_{i+1}" for i in range(inputs.shape[1])]
        output_vars = [f"output_{i+1}" for i in range(outputs.shape[1])]
        dmu_var = 'dmu_name'
        
        dea = ComprehensiveDEA(df, input_vars, output_vars, dmu_var)
        
    elif isinstance(data, pd.DataFrame):
        # 신규 방식: 데이터프레임 사용
        print("?? 데이터 입력 방식: 데이터프레임")
        
        # 변수 검증
        if input_vars is None or output_vars is None:
            raise ValueError("데이터프레임 사용시 input_vars, output_vars는 필수입니다.")
        
        missing_cols = []
        for var in input_vars + output_vars:
            if var not in data.columns:
                missing_cols.append(var)
        
        if missing_cols:
            raise ValueError(f"다음 컬럼들이 데이터에 없습니다: {missing_cols}")
        
        # DEA 객체 생성
        dea = ComprehensiveDEA(data, input_vars, output_vars, dmu_var)
        
    elif isinstance(data, np.ndarray):
        # 배열 입력 방식
        print("?? 데이터 입력 방식: 배열")
        
        if input_vars is None or output_vars is None:
            raise ValueError("배열 사용시 input_vars, output_vars 인덱스가 필요합니다.")
        
        # 배열에서 입력/출력 분리
        inputs = data[:, input_vars] if isinstance(input_vars[0], int) else data
        outputs = data[:, output_vars] if isinstance(output_vars[0], int) else data
        
        if dmu_names is None:
            dmu_names = [f"DMU_{i+1}" for i in range(data.shape[0])]
        
        # 임시 데이터프레임 생성
        df = pd.DataFrame(inputs, columns=[f"input_{i+1}" for i in range(inputs.shape[1])])
        df = pd.concat([df, pd.DataFrame(outputs, columns=[f"output_{i+1}" for i in range(outputs.shape[1])])], axis=1)
        df['dmu_name'] = dmu_names
        
        input_vars = [f"input_{i+1}" for i in range(inputs.shape[1])]
        output_vars = [f"output_{i+1}" for i in range(outputs.shape[1])]
        dmu_var = 'dmu_name'
        
        dea = ComprehensiveDEA(df, input_vars, output_vars, dmu_var)
    else:
        raise ValueError("지원되지 않는 데이터 형식입니다.")
    
    print(f"\n?? 분석 개요")
    print(f"   ? DMU 수: {dea.n_dmu}")
    print(f"   ? 입력 변수 수: {dea.n_inputs}")
    print(f"   ? 출력 변수 수: {dea.n_outputs}")
    print(f"   ? 분석 방향: {orientation.upper()} 지향")
    print(f"   ? 입력 변수: {', '.join(input_vars)}")
    print(f"   ? 출력 변수: {', '.join(output_vars)}")
    print(f"   ? 분석 모델: {', '.join(models)}")
    
    # 기본 통계 출력
    print(f"\n?? 데이터 기본 통계:")
    print(f"   입력 변수 통계:")
    for i, var in enumerate(input_vars):
        values = dea.inputs[:, i]
        print(f"     ? {var}: 평균 {np.mean(values):.2f}, 표준편차 {np.std(values):.2f}, 범위 [{np.min(values):.2f}, {np.max(values):.2f}]")
    
    print(f"   출력 변수 통계:")
    for i, var in enumerate(output_vars):
        values = dea.outputs[:, i]
        print(f"     ? {var}: 평균 {np.mean(values):.2f}, 표준편차 {np.std(values):.2f}, 범위 [{np.min(values):.2f}, {np.max(values):.2f}]")
    
    # 1. DEA 모델 분석
    print(f"\n" + "="*80)
    print("1. DEA 모델 효율성 분석")
    print("="*80)
    
    model_results = {}
    
    for model in models:
        print(f"\n?? {model} 모델 분석 중...")
        scores, lambdas = dea.solve_dea(model, orientation)
        model_results[model] = {
            'scores': scores,
            'lambdas': lambdas
        }
        
        print(f"   ? 평균 효율성: {np.mean(scores):.4f}")
        print(f"   ? 표준편차: {np.std(scores):.4f}")
        print(f"   ? 효율적 DMU 수: {np.sum(scores >= 0.99)}")
        print(f"   ? 최고 효율성: {np.max(scores):.4f}")
        print(f"   ? 최저 효율성: {np.min(scores):.4f}")
    
    # 주요 모델 선택 (BCC 우선, 없으면 첫 번째)
    primary_model = 'BCC' if 'BCC' in models else models[0]
    primary_scores = model_results[primary_model]['scores']
    primary_lambdas = model_results[primary_model]['lambdas']
    
    # 규모효율성 계산 (CCR과 BCC 모두 있을 때)
    scale_efficiency = None
    returns_to_scale = None
    if 'CCR' in models and 'BCC' in models:
        ccr_scores = model_results['CCR']['scores']
        bcc_scores = model_results['BCC']['scores']
        scale_efficiency = ccr_scores / bcc_scores
        
        returns_to_scale = []
        for i in range(dea.n_dmu):
            if abs(scale_efficiency[i] - 1.0) < 1e-6:
                returns_to_scale.append("CRS")
            elif scale_efficiency[i] < 1.0:
                returns_to_scale.append("IRS")
            else:
                returns_to_scale.append("DRS")
    
    # 2. 결과표 생성
    print(f"\n" + "="*80)
    print("2. 종합 효율성 분석 결과표")
    print("="*80)
    
    results_df = pd.DataFrame({
        'DMU': dea.dmu_names
    })
    
    # 모델별 효율성 점수 추가
    for model in models:
        results_df[f'{model}_효율성'] = model_results[model]['scores']
        results_df[f'{model}_순위'] = pd.Series(model_results[model]['scores']).rank(ascending=False, method='min').astype(int)
    
    # 규모효율성 추가 (가능한 경우)
    if scale_efficiency is not None:
        results_df['규모효율성'] = scale_efficiency
        results_df['규모수익'] = returns_to_scale
    
    results_df = results_df.round(4)
    print(results_df.to_string(index=False))
    
    # 효율적/비효율적 DMU 분류
    efficient_dmus = results_df[results_df[f'{primary_model}_효율성'] >= 0.99]['DMU'].tolist()
    inefficient_dmus = results_df[results_df[f'{primary_model}_효율성'] < 0.99]['DMU'].tolist()
    
    print(f"\n?? 효율성 분류 ({primary_model} 기준):")
    print(f"   효율적 DMU: {', '.join(efficient_dmus) if efficient_dmus else '없음'}")
    print(f"   비효율적 DMU: {', '.join(inefficient_dmus) if inefficient_dmus else '없음'}")
    
    # 3. 참조집합 분석
    print(f"\n" + "="*80)
    print("3. 참조집합(벤치마크) 분석")
    print("="*80)
    
    reference_sets = dea.get_reference_set(primary_lambdas)
    
    print(f"\n?? DMU별 벤치마크 ({primary_model} 기준):")
    for dmu, refs in reference_sets.items():
        if len(refs) > 1 or (len(refs) == 1 and refs[0] != dmu):
            print(f"   {dmu}: {', '.join(refs)}")
        elif len(refs) == 1 and refs[0] == dmu:
            print(f"   {dmu}: 자기 자신 (효율적)")
    
    # 벤치마크 빈도 분석
    benchmark_count = {}
    for refs in reference_sets.values():
        for ref in refs:
            benchmark_count[ref] = benchmark_count.get(ref, 0) + 1
    
    if benchmark_count:
        top_benchmarks = sorted(benchmark_count.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n?? 주요 벤치마크 DMU:")
        for dmu, count in top_benchmarks:
            print(f"   {dmu}: {count}회 참조됨")
    
    # 4. 슬랙 분석
    print(f"\n" + "="*80)
    print("4. 슬랙(여유분) 분석")
    print("="*80)
    
    input_slacks, output_slacks = dea.calculate_slacks(primary_scores, primary_lambdas, orientation)
    
    print(f"\n?? 슬랙 통계 ({primary_model} 모델 기준):")
    print(f"   입력 슬랙 평균:")
    for i, var in enumerate(input_vars):
        avg_slack = np.mean(input_slacks[:, i])
        print(f"     ? {var}: {avg_slack:.4f}")
    
    print(f"   출력 슬랙 평균:")
    for i, var in enumerate(output_vars):
        avg_slack = np.mean(output_slacks[:, i])
        print(f"     ? {var}: {avg_slack:.4f}")
    
    # 슬랙이 큰 DMU들 식별
    total_input_slack = np.sum(input_slacks, axis=1)
    total_output_slack = np.sum(output_slacks, axis=1)
    
    print(f"\n?? 주요 슬랙 보유 DMU:")
    for i, dmu in enumerate(dea.dmu_names):
        if total_input_slack[i] > 0 or total_output_slack[i] > 0:
            print(f"   {dmu}:")
            for j, var in enumerate(input_vars):
                if input_slacks[i, j] > 1e-6:
                    print(f"     - {var} 추가 절감 가능: {input_slacks[i, j]:.4f}")
            for j, var in enumerate(output_vars):
                if output_slacks[i, j] > 1e-6:
                    print(f"     - {var} 추가 증대 가능: {output_slacks[i, j]:.4f}")
    
    # 5. 개선방안 제시
    print(f"\n" + "="*80)
    print("5. 구체적 개선방안")
    print("="*80)
    
    print(f"\n?? 비효율적 DMU 개선방안 ({orientation.upper()} 지향, {primary_model} 기준):")
    
    inefficient_indices = np.where(primary_scores < 0.99)[0]
    for idx in inefficient_indices:
        dmu_name = dea.dmu_names[idx]
        efficiency = primary_scores[idx]
        
        print(f"\n   ?? {dmu_name} (효율성: {efficiency:.4f})")
        
        if orientation == 'input':
            # 입력 감소 목표
            target_inputs = dea.inputs[idx] * efficiency
            input_reductions = dea.inputs[idx] - target_inputs
            
            print(f"      입력 감소 목표:")
            for i, var in enumerate(input_vars):
                if input_reductions[i] > 1e-6:
                    current_val = dea.inputs[idx, i]
                    reduction_pct = (input_reductions[i] / current_val) * 100
                    print(f"        ? {var}: {current_val:.4f} → {target_inputs[i]:.4f} ({reduction_pct:.1f}% 감소)")
        else:
            # 출력 증가 목표  
            target_outputs = dea.outputs[idx] / efficiency
            output_increases = target_outputs - dea.outputs[idx]
            
            print(f"      출력 증가 목표:")
            for i, var in enumerate(output_vars):
                if output_increases[i] > 1e-6:
                    current_val = dea.outputs[idx, i]
                    increase_pct = (output_increases[i] / current_val) * 100
                    print(f"        ? {var}: {current_val:.4f} → {target_outputs[i]:.4f} ({increase_pct:.1f}% 증가)")
        
        # 벤치마크 정보
        refs = reference_sets[dmu_name]
        if refs and refs != [dmu_name]:
            print(f"      벤치마크: {', '.join(refs)}")
    
    # 6. 민감도 분석
    print(f"\n" + "="*80)
    print("6. 민감도 분석")
    print("="*80)
    
    # 민감도 분석 대상 DMU 결정
    if sensitivity_dmu is None:
        if len(inefficient_indices) > 0:
            sensitivity_dmu = inefficient_indices[np.argmin(primary_scores[inefficient_indices])]
        else:
            sensitivity_dmu = np.argmin(primary_scores)
    elif isinstance(sensitivity_dmu, str):
        # 이름으로 인덱스 찾기
        try:
            sensitivity_dmu = dea.dmu_names.index(sensitivity_dmu)
        except ValueError:
            print(f"   ? '{sensitivity_dmu}' DMU를 찾을 수 없습니다. 가장 비효율적인 DMU로 대체합니다.")
            sensitivity_dmu = np.argmin(primary_scores)
    
    target_dmu = dea.dmu_names[sensitivity_dmu]
    print(f"?? 민감도 분석 대상 DMU: {target_dmu}")
    print(f"   현재 효율성: {primary_scores[sensitivity_dmu]:.4f}")
    
    try:
        sensitivity_results = dea.sensitivity_analysis(sensitivity_dmu, primary_model, orientation, change_range)
        
        print(f"\n?? {target_dmu}의 변수별 민감도 분석 ({change_range[0]*100:+.0f}% ~ {change_range[1]*100:+.0f}% 변화):")
        
        changes = sensitivity_results['changes']
        
        print(f"\n   ?? 입력 변수 민감도:")
        for i, var in enumerate(input_vars):
            sensitivity = sensitivity_results['input_sensitivity'][i]
            min_eff = np.min(sensitivity)
            max_eff = np.max(sensitivity)
            range_eff = max_eff - min_eff
            
            best_change_idx = np.argmax(sensitivity)
            best_change_pct = changes[best_change_idx] * 100
            best_eff = sensitivity[best_change_idx]
            
            print(f"     ?? {var}:")
            print(f"       ? 효율성 변화폭: {range_eff:.4f} (최소: {min_eff:.4f}, 최대: {max_eff:.4f})")
            print(f"       ? 최적 변화율: {best_change_pct:+.1f}% → 효율성 {best_eff:.4f}")
        
        print(f"\n   ?? 출력 변수 민감도:")
        for i, var in enumerate(output_vars):
            sensitivity = sensitivity_results['output_sensitivity'][i]
            min_eff = np.min(sensitivity)
            max_eff = np.max(sensitivity)
            range_eff = max_eff - min_eff
            
            best_change_idx = np.argmax(sensitivity)
            best_change_pct = changes[best_change_idx] * 100
            best_eff = sensitivity[best_change_idx]
            
            print(f"     ?? {var}:")
            print(f"       ? 효율성 변화폭: {range_eff:.4f} (최소: {min_eff:.4f}, 최대: {max_eff:.4f})")
            print(f"       ? 최적 변화율: {best_change_pct:+.1f}% → 효율성 {best_eff:.4f}")
        
        # 가장 영향력 있는 변수 식별
        input_impacts = [np.max(sensitivity_results['input_sensitivity'][i]) - np.min(sensitivity_results['input_sensitivity'][i]) 
                         for i in range(dea.n_inputs)]
        output_impacts = [np.max(sensitivity_results['output_sensitivity'][i]) - np.min(sensitivity_results['output_sensitivity'][i]) 
                          for i in range(dea.n_outputs)]
        
        if dea.n_inputs > 0:
            most_sensitive_input_idx = np.argmax(input_impacts)
            print(f"\n   ?? 가장 민감한 입력 변수: {input_vars[most_sensitive_input_idx]} (변화폭: {input_impacts[most_sensitive_input_idx]:.4f})")
        
        if dea.n_outputs > 0:
            most_sensitive_output_idx = np.argmax(output_impacts)
            print(f"   ?? 가장 민감한 출력 변수: {output_vars[most_sensitive_output_idx]} (변화폭: {output_impacts[most_sensitive_output_idx]:.4f})")
            
        # 실용적인 개선 제안
        print(f"\n   ?? 민감도 기반 개선 제안:")
        if orientation == 'input':
            if dea.n_inputs > 0:
                input_reduction_effects = []
                for i in range(dea.n_inputs):
                    negative_changes = sensitivity_results['input_sensitivity'][i][:len(changes)//2]
                    max_improvement = np.max(negative_changes) - primary_scores[sensitivity_dmu]
                    input_reduction_effects.append(max_improvement)
                
                best_input_to_reduce = np.argmax(input_reduction_effects)
                print(f"       ? {input_vars[best_input_to_reduce]} 감소에 집중 (예상 효율성 개선: {input_reduction_effects[best_input_to_reduce]:.4f})")
        else:
            if dea.n_outputs > 0:
                output_increase_effects = []
                for i in range(dea.n_outputs):
                    positive_changes = sensitivity_results['output_sensitivity'][i][len(changes)//2:]
                    max_improvement = np.max(positive_changes) - primary_scores[sensitivity_dmu]
                    output_increase_effects.append(max_improvement)
                
                best_output_to_increase = np.argmax(output_increase_effects)
                print(f"       ? {output_vars[best_output_to_increase]} 증대에 집중 (예상 효율성 개선: {output_increase_effects[best_output_to_increase]:.4f})")
        
    except Exception as e:
        print(f"   ? 민감도 분석 중 오류 발생: {str(e)}")
        sensitivity_results = None
    
    # 7. 시각화
    if plot_results:
        print(f"\n" + "="*80)
        print("7. 결과 시각화")
        print("="*80)
        
        # 플롯 크기 조정 (모델 수에 따라)
        n_models = len(models)
        fig_width = max(18, 6 * n_models)
        fig, axes = plt.subplots(2, 3, figsize=(fig_width, 12))
        
        # 1) 효율성 점수 비교
        x = np.arange(len(dea.dmu_names))
        width = 0.8 / n_models
        colors = ['skyblue', 'orange', 'green', 'red', 'purple']
        
        for i, model in enumerate(models):
            scores = model_results[model]['scores']
            axes[0, 0].bar(x + (i - n_models/2 + 0.5) * width, scores, width, 
                          label=model, alpha=0.8, color=colors[i % len(colors)])
        
        axes[0, 0].set_ylabel('효율성 점수')
        axes[0, 0].set_title('모델별 효율성 점수 비교')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(dea.dmu_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # 2) 효율성 분포
        all_scores = [model_results[model]['scores'] for model in models]
        axes[0, 1].hist(all_scores, bins=10, alpha=0.7, label=models, 
                       color=colors[:len(models)])
        axes[0, 1].set_xlabel('효율성 점수')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].set_title('효율성 점수 분포')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3) 규모효율성 (가능한 경우)
        if scale_efficiency is not None:
            colors_rts = ['green' if x >= 0.99 else 'red' if x < 0.95 else 'orange' for x in scale_efficiency]
            axes[0, 2].bar(range(len(dea.dmu_names)), scale_efficiency, alpha=0.8, color=colors_rts)
            axes[0, 2].set_ylabel('규모효율성')
            axes[0, 2].set_title('규모효율성 점수')
            axes[0, 2].set_xticks(range(len(dea.dmu_names)))
            axes[0, 2].set_xticklabels(dea.dmu_names, rotation=45, ha='right')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        else:
            axes[0, 2].text(0.5, 0.5, f'규모효율성\n계산 불가\n(CCR+BCC 필요)', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('규모효율성')
        
        # 4) 입력-출력 관계 (2차원인 경우)
        if dea.n_inputs == 1 and dea.n_outputs == 1:
            colors_scatter = ['red' if score < 0.99 else 'blue' for score in primary_scores]
            axes[1, 0].scatter(dea.inputs[:, 0], dea.outputs[:, 0], c=colors_scatter, alpha=0.7, s=100)
            axes[1, 0].set_xlabel(input_vars[0])
            axes[1, 0].set_ylabel(output_vars[0])
            axes[1, 0].set_title('입력-출력 관계 (빨강: 비효율)')
            
            # 효율적 경계선
            efficient_dmu_indices = np.where(primary_scores >= 0.99)[0]
            if len(efficient_dmu_indices) > 1:
                eff_inputs = dea.inputs[efficient_dmu_indices, 0]
                eff_outputs = dea.outputs[efficient_dmu_indices, 0]
                sorted_idx = np.argsort(eff_inputs)
                axes[1, 0].plot(eff_inputs[sorted_idx], eff_outputs[sorted_idx], 
                               'k--', alpha=0.5, label='효율적 경계')
                axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, f'다차원 입력-출력\n시각화 불가\n({dea.n_inputs}입력, {dea.n_outputs}출력)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('입력-출력 관계')
        
        # 5) 슬랙 분석
        total_slacks = np.sum(input_slacks, axis=1) + np.sum(output_slacks, axis=1)
        axes[1, 1].bar(range(len(dea.dmu_names)), total_slacks, alpha=0.8, color='purple')
        axes[1, 1].set_ylabel('총 슬랙')
        axes[1, 1].set_title('DMU별 총 슬랙')
        axes[1, 1].set_xticks(range(len(dea.dmu_names)))
        axes[1, 1].set_xticklabels(dea.dmu_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6) 민감도 분석
        if sensitivity_results is not None:
            changes_pct = sensitivity_results['changes'] * 100
            
            input_ranges = [np.max(sensitivity_results['input_sensitivity'][i]) - 
                           np.min(sensitivity_results['input_sensitivity'][i]) 
                           for i in range(dea.n_inputs)]
            output_ranges = [np.max(sensitivity_results['output_sensitivity'][i]) - 
                            np.min(sensitivity_results['output_sensitivity'][i]) 
                            for i in range(dea.n_outputs)]
            
            if dea.n_inputs > 0:
                most_sensitive_input = np.argmax(input_ranges)
                axes[1, 2].plot(changes_pct, sensitivity_results['input_sensitivity'][most_sensitive_input], 
                               'b-', linewidth=2, label=f'{input_vars[most_sensitive_input]} (입력)')
            
            if dea.n_outputs > 0:
                most_sensitive_output = np.argmax(output_ranges)
                axes[1, 2].plot(changes_pct, sensitivity_results['output_sensitivity'][most_sensitive_output], 
                               'r-', linewidth=2, label=f'{output_vars[most_sensitive_output]} (출력)')
            
            axes[1, 2].set_xlabel('변화율 (%)')
            axes[1, 2].set_ylabel('효율성 점수')
            axes[1, 2].set_title(f'{target_dmu} 민감도 분석')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].axhline(y=primary_scores[sensitivity_dmu], color='green', linestyle='--', alpha=0.5)
        else:
            axes[1, 2].text(0.5, 0.5, '민감도 분석\n오류 발생', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12, color='red')
            axes[1, 2].set_title('민감도 분석')
        
        plt.tight_layout()
        plt.show()
        
        print("   ? 시각화 완료")
    
    # 8. 요약 및 권고사항
    print(f"\n" + "="*80)
    print("8. 분석 요약 및 권고사항")
    print("="*80)
    
    print(f"\n?? 분석 요약:")
    print(f"   ? 전체 DMU 중 {len(efficient_dmus)}개가 효율적 ({primary_model} 기준)")
    print(f"   ? 평균 효율성: {np.mean(primary_scores):.4f}")
    
    if scale_efficiency is not None:
        print(f"   ? 평균 규모효율성: {np.mean(scale_efficiency):.4f}")
        rts_counts = pd.Series(returns_to_scale).value_counts()
        print(f"   ? 규모수익 분포:")
        for rts, count in rts_counts.items():
            rts_full = {'CRS': '규모수익불변', 'IRS': '규모수익증가', 'DRS': '규모수익감소'}
            print(f"     - {rts_full.get(rts, rts)}: {count}개")
    
    print(f"   ? 개선이 필요한 DMU: {len(inefficient_dmus)}개")
    
    print(f"\n?? 주요 권고사항:")
    if len(inefficient_dmus) > 0:
        worst_dmu_idx = np.argmin(primary_scores)
        worst_dmu = dea.dmu_names[worst_dmu_idx]
        print(f"   1. 우선 개선 대상: {worst_dmu} (효율성: {primary_scores[worst_dmu_idx]:.4f})")
        
        if benchmark_count:
            best_benchmark = max(benchmark_count.items(), key=lambda x: x[1])[0]
            print(f"   2. 주요 벤치마크 모델: {best_benchmark}")
        
        print(f"   3. {orientation.upper()} 지향 개선에 집중")
        
        # 가장 개선 효과가 큰 변수 식별
        if orientation == 'input':
            avg_input_slack = np.mean(input_slacks[inefficient_indices], axis=0)
            if np.any(avg_input_slack > 0):
                max_slack_input = np.argmax(avg_input_slack)
                print(f"   4. 우선 개선 변수: {input_vars[max_slack_input]} (평균 슬랙: {avg_input_slack[max_slack_input]:.4f})")
        else:
            avg_output_slack = np.mean(output_slacks[inefficient_indices], axis=0)
            if np.any(avg_output_slack > 0):
                max_slack_output = np.argmax(avg_output_slack)
                print(f"   4. 우선 개선 변수: {output_vars[max_slack_output]} (평균 슬랙: {avg_output_slack[max_slack_output]:.4f})")
    else:
        print(f"   ? 모든 DMU가 효율적으로 운영되고 있습니다!")
    
    print(f"\n" + "="*80)
    print("                     분석 완료")
    print("="*80)
    
    # 결과 반환
    result_dict = {
        'results_df': results_df,
        'model_results': model_results,
        'reference_sets': reference_sets,
        'input_slacks': input_slacks,
        'output_slacks': output_slacks,
        'sensitivity_results': sensitivity_results,
        'benchmark_count': benchmark_count,
        'data_info': {
            'input_vars': input_vars,
            'output_vars': output_vars,
            'dmu_var': dmu_var,
            'orientation': orientation,
            'models': models
        }
    }
    
    if scale_efficiency is not None:
        result_dict['scale_efficiency'] = scale_efficiency
        result_dict['returns_to_scale'] = returns_to_scale
    
    return result_dict