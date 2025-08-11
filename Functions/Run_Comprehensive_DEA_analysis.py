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
    ���� DEA �м� Ŭ���� - �Ϲ�ȭ�� ����
    �����������Ӱ� �������� ����Ͽ� �м�
    """
    
    def __init__(self, data: pd.DataFrame, 
                 input_vars: List[str], 
                 output_vars: List[str],
                 dmu_var: str = None):
        """
        Parameters:
        data: ��ü ������������
        input_vars: �Է� ���� �÷��� ����Ʈ
        output_vars: ��� ���� �÷��� ����Ʈ  
        dmu_var: DMU �ĺ� �÷��� (������ �ε��� ���)
        """
        self.data = data.copy()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.dmu_var = dmu_var
        
        # �Է�/��� ������ ����
        self.inputs = self.data[input_vars].values.astype(float)
        self.outputs = self.data[output_vars].values.astype(float)
        
        self.n_dmu = self.inputs.shape[0]
        self.n_inputs = self.inputs.shape[1] 
        self.n_outputs = self.outputs.shape[1]
        
        # DMU �̸� ����
        if dmu_var and dmu_var in data.columns:
            self.dmu_names = data[dmu_var].astype(str).tolist()
        else:
            self.dmu_names = [f"DMU_{i+1}" for i in range(self.n_dmu)]
    
    def solve_dea(self, model_type: str = 'BCC', orientation: str = 'input') -> Tuple[np.ndarray, np.ndarray]:
        """
        DEA �� �ذ�
        
        Parameters:
        model_type: 'CCR' �Ǵ� 'BCC'
        orientation: 'input' �Ǵ� 'output'
        
        Returns:
        efficiency_scores: ȿ���� ����
        lambdas: ����ġ ���
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
            
            # �����Լ�
            prob += objective
            
            # ��������
            if orientation == 'input':
                # �Է� ����
                for i in range(self.n_inputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.inputs[j, i] 
                                       for j in range(self.n_dmu)]) 
                            <= theta * self.inputs[j0, i])
                
                # ��� ����
                for r in range(self.n_outputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.outputs[j, r] 
                                       for j in range(self.n_dmu)]) 
                            >= self.outputs[j0, r])
            else:  # output orientation
                # �Է� ����
                for i in range(self.n_inputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.inputs[j, i] 
                                       for j in range(self.n_dmu)]) 
                            <= self.inputs[j0, i])
                
                # ��� ����
                for r in range(self.n_outputs):
                    prob += (pulp.lpSum([lambda_vars[j] * self.outputs[j, r] 
                                       for j in range(self.n_dmu)]) 
                            >= phi * self.outputs[j0, r])
            
            # BCC ���� ���ϼ� ����
            if model_type == 'BCC':
                prob += pulp.lpSum(lambda_vars) == 1
            
            # ���� �ذ�
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # ��� ����
            if orientation == 'input':
                efficiency_scores[j0] = pulp.value(theta)
            else:
                efficiency_scores[j0] = 1.0 / pulp.value(phi) if pulp.value(phi) > 0 else 0
            
            for j in range(self.n_dmu):
                lambdas[j0, j] = pulp.value(lambda_vars[j]) if pulp.value(lambda_vars[j]) else 0
        
        return efficiency_scores, lambdas
    
    def calculate_slacks(self, efficiency_scores: np.ndarray, lambdas: np.ndarray, 
                        orientation: str = 'input') -> Tuple[np.ndarray, np.ndarray]:
        """���� ���"""
        input_slacks = np.zeros((self.n_dmu, self.n_inputs))
        output_slacks = np.zeros((self.n_dmu, self.n_outputs))
        
        for j0 in range(self.n_dmu):
            # ������ ���
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
        """�������� ���"""
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
        """�ΰ��� �м�"""
        changes = np.linspace(change_range[0], change_range[1], n_points)
        original_inputs = self.inputs.copy()
        original_outputs = self.outputs.copy()
        
        input_sensitivity = np.zeros((self.n_inputs, n_points))
        output_sensitivity = np.zeros((self.n_outputs, n_points))
        
        print(f"   ?? �ΰ��� �м� ���� ��...")
        
        # �Է� ���� �ΰ���
        for i in range(self.n_inputs):
            for j, change in enumerate(changes):
                # �ӽ÷� �Է°� ����
                temp_inputs = self.inputs.copy()
                temp_inputs[dmu_index, i] = original_inputs[dmu_index, i] * (1 + change)
                
                # �ӽ� DEA ��ü �����Ͽ� �м�
                temp_dea = ComprehensiveDEA(pd.DataFrame(), [], [])
                temp_dea.inputs = temp_inputs
                temp_dea.outputs = self.outputs
                temp_dea.n_dmu = self.n_dmu
                temp_dea.n_inputs = self.n_inputs
                temp_dea.n_outputs = self.n_outputs
                temp_dea.dmu_names = self.dmu_names
                
                scores, _ = temp_dea.solve_dea(model_type, orientation)
                input_sensitivity[i, j] = scores[dmu_index]
        
        # ��� ���� �ΰ���
        for i in range(self.n_outputs):
            for j, change in enumerate(changes):
                # �ӽ÷� ��°� ����
                temp_outputs = self.outputs.copy()
                temp_outputs[dmu_index, i] = original_outputs[dmu_index, i] * (1 + change)
                
                # �ӽ� DEA ��ü �����Ͽ� �м�
                temp_dea = ComprehensiveDEA(pd.DataFrame(), [], [])
                temp_dea.inputs = self.inputs
                temp_dea.outputs = temp_outputs
                temp_dea.n_dmu = self.n_dmu
                temp_dea.n_inputs = self.n_inputs
                temp_dea.n_outputs = self.n_outputs
                temp_dea.dmu_names = self.dmu_names
                
                scores, _ = temp_dea.solve_dea(model_type, orientation)
                output_sensitivity[i, j] = scores[dmu_index]
        
        print(f"   ? �ΰ��� �м� �Ϸ�")
        
        return {
            'changes': changes,
            'input_sensitivity': input_sensitivity,
            'output_sensitivity': output_sensitivity
        }

def Run_Comprehensive_DEA_analysis(data: Union[pd.DataFrame, np.ndarray], 
                              input_vars: Union[List[str], List[int]] = None,
                              output_vars: Union[List[str], List[int]] = None,
                              dmu_var: str = None,
                              input_data: np.ndarray = None,  # ��ȣȯ��
                              output_data: np.ndarray = None,  # ��ȣȯ��
                              dmu_names: List[str] = None,   # ��ȣȯ��
                              orientation: str = 'input',
                              sensitivity_dmu: Union[int, str] = None,
                              plot_results: bool = True,
                              models: List[str] = ['CCR', 'BCC'],
                              change_range: Tuple[float, float] = (-0.2, 0.2)) -> Dict:
    """
    ���� DEA �м� �Լ� - �Ϲ�ȭ�� ����
    
    Parameters:
    data: ������������ �Ǵ� numpy �迭
    input_vars: �Է� ���� �÷��� ����Ʈ (DataFrame ����) �Ǵ� �ε��� (�迭 ����)
    output_vars: ��� ���� �÷��� ����Ʈ (DataFrame ����) �Ǵ� �ε��� (�迭 ����)
    dmu_var: DMU �ĺ� �÷��� (DataFrame ����)
    input_data: �Է� ������ �迭 (��ȣȯ���� ���� ����)
    output_data: ��� ������ �迭 (��ȣȯ���� ���� ����)
    dmu_names: DMU �̸� ����Ʈ (��ȣȯ���� ���� ����)
    orientation: 'input' �Ǵ� 'output'
    sensitivity_dmu: �ΰ��� �м��� DMU (�ε��� �Ǵ� �̸�)
    plot_results: ��� �ð�ȭ ����
    models: �м��� �� ����Ʈ ['CCR', 'BCC']
    change_range: �ΰ��� �м� ��ȭ ����
    
    Returns:
    �м� ��� ��ųʸ�
    
    Usage Examples:
    
    # ��� 1: ������������ ��� (����)
    results = comprehensive_dea_analysis(
        data=df,
        input_vars=['employees', 'operating_cost'],
        output_vars=['deposits', 'loans'],
        dmu_var='bank_name',
        orientation='input'
    )
    
    # ��� 2: �迭 ��� (��ȣȯ��)
    results = comprehensive_dea_analysis(
        input_data=input_array,
        output_data=output_array,
        dmu_names=name_list,
        orientation='input'
    )
    """
    
    print("="*80)
    print("                     ���� DEA �м� �ý��� (�Ϲ�ȭ ����)")
    print("="*80)
    
    # �Է� ������ ó�� �� ����
    if input_data is not None and output_data is not None:
        # ��ȣȯ��: ���� ���
        print("?? ������ �Է� ���: �迭 (��ȣȯ ���)")
        inputs = np.array(input_data, dtype=float)
        outputs = np.array(output_data, dtype=float)
        
        if dmu_names is None:
            dmu_names = [f"DMU_{i+1}" for i in range(inputs.shape[0])]
        
        # �ӽ� ������������ ����
        df = pd.DataFrame(inputs, columns=[f"input_{i+1}" for i in range(inputs.shape[1])])
        df = pd.concat([df, pd.DataFrame(outputs, columns=[f"output_{i+1}" for i in range(outputs.shape[1])])], axis=1)
        df['dmu_name'] = dmu_names
        
        input_vars = [f"input_{i+1}" for i in range(inputs.shape[1])]
        output_vars = [f"output_{i+1}" for i in range(outputs.shape[1])]
        dmu_var = 'dmu_name'
        
        dea = ComprehensiveDEA(df, input_vars, output_vars, dmu_var)
        
    elif isinstance(data, pd.DataFrame):
        # �ű� ���: ������������ ���
        print("?? ������ �Է� ���: ������������")
        
        # ���� ����
        if input_vars is None or output_vars is None:
            raise ValueError("������������ ���� input_vars, output_vars�� �ʼ��Դϴ�.")
        
        missing_cols = []
        for var in input_vars + output_vars:
            if var not in data.columns:
                missing_cols.append(var)
        
        if missing_cols:
            raise ValueError(f"���� �÷����� �����Ϳ� �����ϴ�: {missing_cols}")
        
        # DEA ��ü ����
        dea = ComprehensiveDEA(data, input_vars, output_vars, dmu_var)
        
    elif isinstance(data, np.ndarray):
        # �迭 �Է� ���
        print("?? ������ �Է� ���: �迭")
        
        if input_vars is None or output_vars is None:
            raise ValueError("�迭 ���� input_vars, output_vars �ε����� �ʿ��մϴ�.")
        
        # �迭���� �Է�/��� �и�
        inputs = data[:, input_vars] if isinstance(input_vars[0], int) else data
        outputs = data[:, output_vars] if isinstance(output_vars[0], int) else data
        
        if dmu_names is None:
            dmu_names = [f"DMU_{i+1}" for i in range(data.shape[0])]
        
        # �ӽ� ������������ ����
        df = pd.DataFrame(inputs, columns=[f"input_{i+1}" for i in range(inputs.shape[1])])
        df = pd.concat([df, pd.DataFrame(outputs, columns=[f"output_{i+1}" for i in range(outputs.shape[1])])], axis=1)
        df['dmu_name'] = dmu_names
        
        input_vars = [f"input_{i+1}" for i in range(inputs.shape[1])]
        output_vars = [f"output_{i+1}" for i in range(outputs.shape[1])]
        dmu_var = 'dmu_name'
        
        dea = ComprehensiveDEA(df, input_vars, output_vars, dmu_var)
    else:
        raise ValueError("�������� �ʴ� ������ �����Դϴ�.")
    
    print(f"\n?? �м� ����")
    print(f"   ? DMU ��: {dea.n_dmu}")
    print(f"   ? �Է� ���� ��: {dea.n_inputs}")
    print(f"   ? ��� ���� ��: {dea.n_outputs}")
    print(f"   ? �м� ����: {orientation.upper()} ����")
    print(f"   ? �Է� ����: {', '.join(input_vars)}")
    print(f"   ? ��� ����: {', '.join(output_vars)}")
    print(f"   ? �м� ��: {', '.join(models)}")
    
    # �⺻ ��� ���
    print(f"\n?? ������ �⺻ ���:")
    print(f"   �Է� ���� ���:")
    for i, var in enumerate(input_vars):
        values = dea.inputs[:, i]
        print(f"     ? {var}: ��� {np.mean(values):.2f}, ǥ������ {np.std(values):.2f}, ���� [{np.min(values):.2f}, {np.max(values):.2f}]")
    
    print(f"   ��� ���� ���:")
    for i, var in enumerate(output_vars):
        values = dea.outputs[:, i]
        print(f"     ? {var}: ��� {np.mean(values):.2f}, ǥ������ {np.std(values):.2f}, ���� [{np.min(values):.2f}, {np.max(values):.2f}]")
    
    # 1. DEA �� �м�
    print(f"\n" + "="*80)
    print("1. DEA �� ȿ���� �м�")
    print("="*80)
    
    model_results = {}
    
    for model in models:
        print(f"\n?? {model} �� �м� ��...")
        scores, lambdas = dea.solve_dea(model, orientation)
        model_results[model] = {
            'scores': scores,
            'lambdas': lambdas
        }
        
        print(f"   ? ��� ȿ����: {np.mean(scores):.4f}")
        print(f"   ? ǥ������: {np.std(scores):.4f}")
        print(f"   ? ȿ���� DMU ��: {np.sum(scores >= 0.99)}")
        print(f"   ? �ְ� ȿ����: {np.max(scores):.4f}")
        print(f"   ? ���� ȿ����: {np.min(scores):.4f}")
    
    # �ֿ� �� ���� (BCC �켱, ������ ù ��°)
    primary_model = 'BCC' if 'BCC' in models else models[0]
    primary_scores = model_results[primary_model]['scores']
    primary_lambdas = model_results[primary_model]['lambdas']
    
    # �Ը�ȿ���� ��� (CCR�� BCC ��� ���� ��)
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
    
    # 2. ���ǥ ����
    print(f"\n" + "="*80)
    print("2. ���� ȿ���� �м� ���ǥ")
    print("="*80)
    
    results_df = pd.DataFrame({
        'DMU': dea.dmu_names
    })
    
    # �𵨺� ȿ���� ���� �߰�
    for model in models:
        results_df[f'{model}_ȿ����'] = model_results[model]['scores']
        results_df[f'{model}_����'] = pd.Series(model_results[model]['scores']).rank(ascending=False, method='min').astype(int)
    
    # �Ը�ȿ���� �߰� (������ ���)
    if scale_efficiency is not None:
        results_df['�Ը�ȿ����'] = scale_efficiency
        results_df['�Ը����'] = returns_to_scale
    
    results_df = results_df.round(4)
    print(results_df.to_string(index=False))
    
    # ȿ����/��ȿ���� DMU �з�
    efficient_dmus = results_df[results_df[f'{primary_model}_ȿ����'] >= 0.99]['DMU'].tolist()
    inefficient_dmus = results_df[results_df[f'{primary_model}_ȿ����'] < 0.99]['DMU'].tolist()
    
    print(f"\n?? ȿ���� �з� ({primary_model} ����):")
    print(f"   ȿ���� DMU: {', '.join(efficient_dmus) if efficient_dmus else '����'}")
    print(f"   ��ȿ���� DMU: {', '.join(inefficient_dmus) if inefficient_dmus else '����'}")
    
    # 3. �������� �м�
    print(f"\n" + "="*80)
    print("3. ��������(��ġ��ũ) �м�")
    print("="*80)
    
    reference_sets = dea.get_reference_set(primary_lambdas)
    
    print(f"\n?? DMU�� ��ġ��ũ ({primary_model} ����):")
    for dmu, refs in reference_sets.items():
        if len(refs) > 1 or (len(refs) == 1 and refs[0] != dmu):
            print(f"   {dmu}: {', '.join(refs)}")
        elif len(refs) == 1 and refs[0] == dmu:
            print(f"   {dmu}: �ڱ� �ڽ� (ȿ����)")
    
    # ��ġ��ũ �� �м�
    benchmark_count = {}
    for refs in reference_sets.values():
        for ref in refs:
            benchmark_count[ref] = benchmark_count.get(ref, 0) + 1
    
    if benchmark_count:
        top_benchmarks = sorted(benchmark_count.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n?? �ֿ� ��ġ��ũ DMU:")
        for dmu, count in top_benchmarks:
            print(f"   {dmu}: {count}ȸ ������")
    
    # 4. ���� �м�
    print(f"\n" + "="*80)
    print("4. ����(������) �м�")
    print("="*80)
    
    input_slacks, output_slacks = dea.calculate_slacks(primary_scores, primary_lambdas, orientation)
    
    print(f"\n?? ���� ��� ({primary_model} �� ����):")
    print(f"   �Է� ���� ���:")
    for i, var in enumerate(input_vars):
        avg_slack = np.mean(input_slacks[:, i])
        print(f"     ? {var}: {avg_slack:.4f}")
    
    print(f"   ��� ���� ���:")
    for i, var in enumerate(output_vars):
        avg_slack = np.mean(output_slacks[:, i])
        print(f"     ? {var}: {avg_slack:.4f}")
    
    # ������ ū DMU�� �ĺ�
    total_input_slack = np.sum(input_slacks, axis=1)
    total_output_slack = np.sum(output_slacks, axis=1)
    
    print(f"\n?? �ֿ� ���� ���� DMU:")
    for i, dmu in enumerate(dea.dmu_names):
        if total_input_slack[i] > 0 or total_output_slack[i] > 0:
            print(f"   {dmu}:")
            for j, var in enumerate(input_vars):
                if input_slacks[i, j] > 1e-6:
                    print(f"     - {var} �߰� ���� ����: {input_slacks[i, j]:.4f}")
            for j, var in enumerate(output_vars):
                if output_slacks[i, j] > 1e-6:
                    print(f"     - {var} �߰� ���� ����: {output_slacks[i, j]:.4f}")
    
    # 5. ������� ����
    print(f"\n" + "="*80)
    print("5. ��ü�� �������")
    print("="*80)
    
    print(f"\n?? ��ȿ���� DMU ������� ({orientation.upper()} ����, {primary_model} ����):")
    
    inefficient_indices = np.where(primary_scores < 0.99)[0]
    for idx in inefficient_indices:
        dmu_name = dea.dmu_names[idx]
        efficiency = primary_scores[idx]
        
        print(f"\n   ?? {dmu_name} (ȿ����: {efficiency:.4f})")
        
        if orientation == 'input':
            # �Է� ���� ��ǥ
            target_inputs = dea.inputs[idx] * efficiency
            input_reductions = dea.inputs[idx] - target_inputs
            
            print(f"      �Է� ���� ��ǥ:")
            for i, var in enumerate(input_vars):
                if input_reductions[i] > 1e-6:
                    current_val = dea.inputs[idx, i]
                    reduction_pct = (input_reductions[i] / current_val) * 100
                    print(f"        ? {var}: {current_val:.4f} �� {target_inputs[i]:.4f} ({reduction_pct:.1f}% ����)")
        else:
            # ��� ���� ��ǥ  
            target_outputs = dea.outputs[idx] / efficiency
            output_increases = target_outputs - dea.outputs[idx]
            
            print(f"      ��� ���� ��ǥ:")
            for i, var in enumerate(output_vars):
                if output_increases[i] > 1e-6:
                    current_val = dea.outputs[idx, i]
                    increase_pct = (output_increases[i] / current_val) * 100
                    print(f"        ? {var}: {current_val:.4f} �� {target_outputs[i]:.4f} ({increase_pct:.1f}% ����)")
        
        # ��ġ��ũ ����
        refs = reference_sets[dmu_name]
        if refs and refs != [dmu_name]:
            print(f"      ��ġ��ũ: {', '.join(refs)}")
    
    # 6. �ΰ��� �м�
    print(f"\n" + "="*80)
    print("6. �ΰ��� �м�")
    print("="*80)
    
    # �ΰ��� �м� ��� DMU ����
    if sensitivity_dmu is None:
        if len(inefficient_indices) > 0:
            sensitivity_dmu = inefficient_indices[np.argmin(primary_scores[inefficient_indices])]
        else:
            sensitivity_dmu = np.argmin(primary_scores)
    elif isinstance(sensitivity_dmu, str):
        # �̸����� �ε��� ã��
        try:
            sensitivity_dmu = dea.dmu_names.index(sensitivity_dmu)
        except ValueError:
            print(f"   ? '{sensitivity_dmu}' DMU�� ã�� �� �����ϴ�. ���� ��ȿ������ DMU�� ��ü�մϴ�.")
            sensitivity_dmu = np.argmin(primary_scores)
    
    target_dmu = dea.dmu_names[sensitivity_dmu]
    print(f"?? �ΰ��� �м� ��� DMU: {target_dmu}")
    print(f"   ���� ȿ����: {primary_scores[sensitivity_dmu]:.4f}")
    
    try:
        sensitivity_results = dea.sensitivity_analysis(sensitivity_dmu, primary_model, orientation, change_range)
        
        print(f"\n?? {target_dmu}�� ������ �ΰ��� �м� ({change_range[0]*100:+.0f}% ~ {change_range[1]*100:+.0f}% ��ȭ):")
        
        changes = sensitivity_results['changes']
        
        print(f"\n   ?? �Է� ���� �ΰ���:")
        for i, var in enumerate(input_vars):
            sensitivity = sensitivity_results['input_sensitivity'][i]
            min_eff = np.min(sensitivity)
            max_eff = np.max(sensitivity)
            range_eff = max_eff - min_eff
            
            best_change_idx = np.argmax(sensitivity)
            best_change_pct = changes[best_change_idx] * 100
            best_eff = sensitivity[best_change_idx]
            
            print(f"     ?? {var}:")
            print(f"       ? ȿ���� ��ȭ��: {range_eff:.4f} (�ּ�: {min_eff:.4f}, �ִ�: {max_eff:.4f})")
            print(f"       ? ���� ��ȭ��: {best_change_pct:+.1f}% �� ȿ���� {best_eff:.4f}")
        
        print(f"\n   ?? ��� ���� �ΰ���:")
        for i, var in enumerate(output_vars):
            sensitivity = sensitivity_results['output_sensitivity'][i]
            min_eff = np.min(sensitivity)
            max_eff = np.max(sensitivity)
            range_eff = max_eff - min_eff
            
            best_change_idx = np.argmax(sensitivity)
            best_change_pct = changes[best_change_idx] * 100
            best_eff = sensitivity[best_change_idx]
            
            print(f"     ?? {var}:")
            print(f"       ? ȿ���� ��ȭ��: {range_eff:.4f} (�ּ�: {min_eff:.4f}, �ִ�: {max_eff:.4f})")
            print(f"       ? ���� ��ȭ��: {best_change_pct:+.1f}% �� ȿ���� {best_eff:.4f}")
        
        # ���� ����� �ִ� ���� �ĺ�
        input_impacts = [np.max(sensitivity_results['input_sensitivity'][i]) - np.min(sensitivity_results['input_sensitivity'][i]) 
                         for i in range(dea.n_inputs)]
        output_impacts = [np.max(sensitivity_results['output_sensitivity'][i]) - np.min(sensitivity_results['output_sensitivity'][i]) 
                          for i in range(dea.n_outputs)]
        
        if dea.n_inputs > 0:
            most_sensitive_input_idx = np.argmax(input_impacts)
            print(f"\n   ?? ���� �ΰ��� �Է� ����: {input_vars[most_sensitive_input_idx]} (��ȭ��: {input_impacts[most_sensitive_input_idx]:.4f})")
        
        if dea.n_outputs > 0:
            most_sensitive_output_idx = np.argmax(output_impacts)
            print(f"   ?? ���� �ΰ��� ��� ����: {output_vars[most_sensitive_output_idx]} (��ȭ��: {output_impacts[most_sensitive_output_idx]:.4f})")
            
        # �ǿ����� ���� ����
        print(f"\n   ?? �ΰ��� ��� ���� ����:")
        if orientation == 'input':
            if dea.n_inputs > 0:
                input_reduction_effects = []
                for i in range(dea.n_inputs):
                    negative_changes = sensitivity_results['input_sensitivity'][i][:len(changes)//2]
                    max_improvement = np.max(negative_changes) - primary_scores[sensitivity_dmu]
                    input_reduction_effects.append(max_improvement)
                
                best_input_to_reduce = np.argmax(input_reduction_effects)
                print(f"       ? {input_vars[best_input_to_reduce]} ���ҿ� ���� (���� ȿ���� ����: {input_reduction_effects[best_input_to_reduce]:.4f})")
        else:
            if dea.n_outputs > 0:
                output_increase_effects = []
                for i in range(dea.n_outputs):
                    positive_changes = sensitivity_results['output_sensitivity'][i][len(changes)//2:]
                    max_improvement = np.max(positive_changes) - primary_scores[sensitivity_dmu]
                    output_increase_effects.append(max_improvement)
                
                best_output_to_increase = np.argmax(output_increase_effects)
                print(f"       ? {output_vars[best_output_to_increase]} ���뿡 ���� (���� ȿ���� ����: {output_increase_effects[best_output_to_increase]:.4f})")
        
    except Exception as e:
        print(f"   ? �ΰ��� �м� �� ���� �߻�: {str(e)}")
        sensitivity_results = None
    
    # 7. �ð�ȭ
    if plot_results:
        print(f"\n" + "="*80)
        print("7. ��� �ð�ȭ")
        print("="*80)
        
        # �÷� ũ�� ���� (�� ���� ����)
        n_models = len(models)
        fig_width = max(18, 6 * n_models)
        fig, axes = plt.subplots(2, 3, figsize=(fig_width, 12))
        
        # 1) ȿ���� ���� ��
        x = np.arange(len(dea.dmu_names))
        width = 0.8 / n_models
        colors = ['skyblue', 'orange', 'green', 'red', 'purple']
        
        for i, model in enumerate(models):
            scores = model_results[model]['scores']
            axes[0, 0].bar(x + (i - n_models/2 + 0.5) * width, scores, width, 
                          label=model, alpha=0.8, color=colors[i % len(colors)])
        
        axes[0, 0].set_ylabel('ȿ���� ����')
        axes[0, 0].set_title('�𵨺� ȿ���� ���� ��')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(dea.dmu_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        # 2) ȿ���� ����
        all_scores = [model_results[model]['scores'] for model in models]
        axes[0, 1].hist(all_scores, bins=10, alpha=0.7, label=models, 
                       color=colors[:len(models)])
        axes[0, 1].set_xlabel('ȿ���� ����')
        axes[0, 1].set_ylabel('��')
        axes[0, 1].set_title('ȿ���� ���� ����')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3) �Ը�ȿ���� (������ ���)
        if scale_efficiency is not None:
            colors_rts = ['green' if x >= 0.99 else 'red' if x < 0.95 else 'orange' for x in scale_efficiency]
            axes[0, 2].bar(range(len(dea.dmu_names)), scale_efficiency, alpha=0.8, color=colors_rts)
            axes[0, 2].set_ylabel('�Ը�ȿ����')
            axes[0, 2].set_title('�Ը�ȿ���� ����')
            axes[0, 2].set_xticks(range(len(dea.dmu_names)))
            axes[0, 2].set_xticklabels(dea.dmu_names, rotation=45, ha='right')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].axhline(y=1, color='red', linestyle='--', alpha=0.7)
        else:
            axes[0, 2].text(0.5, 0.5, f'�Ը�ȿ����\n��� �Ұ�\n(CCR+BCC �ʿ�)', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('�Ը�ȿ����')
        
        # 4) �Է�-��� ���� (2������ ���)
        if dea.n_inputs == 1 and dea.n_outputs == 1:
            colors_scatter = ['red' if score < 0.99 else 'blue' for score in primary_scores]
            axes[1, 0].scatter(dea.inputs[:, 0], dea.outputs[:, 0], c=colors_scatter, alpha=0.7, s=100)
            axes[1, 0].set_xlabel(input_vars[0])
            axes[1, 0].set_ylabel(output_vars[0])
            axes[1, 0].set_title('�Է�-��� ���� (����: ��ȿ��)')
            
            # ȿ���� ��輱
            efficient_dmu_indices = np.where(primary_scores >= 0.99)[0]
            if len(efficient_dmu_indices) > 1:
                eff_inputs = dea.inputs[efficient_dmu_indices, 0]
                eff_outputs = dea.outputs[efficient_dmu_indices, 0]
                sorted_idx = np.argsort(eff_inputs)
                axes[1, 0].plot(eff_inputs[sorted_idx], eff_outputs[sorted_idx], 
                               'k--', alpha=0.5, label='ȿ���� ���')
                axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, f'������ �Է�-���\n�ð�ȭ �Ұ�\n({dea.n_inputs}�Է�, {dea.n_outputs}���)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('�Է�-��� ����')
        
        # 5) ���� �м�
        total_slacks = np.sum(input_slacks, axis=1) + np.sum(output_slacks, axis=1)
        axes[1, 1].bar(range(len(dea.dmu_names)), total_slacks, alpha=0.8, color='purple')
        axes[1, 1].set_ylabel('�� ����')
        axes[1, 1].set_title('DMU�� �� ����')
        axes[1, 1].set_xticks(range(len(dea.dmu_names)))
        axes[1, 1].set_xticklabels(dea.dmu_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6) �ΰ��� �м�
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
                               'b-', linewidth=2, label=f'{input_vars[most_sensitive_input]} (�Է�)')
            
            if dea.n_outputs > 0:
                most_sensitive_output = np.argmax(output_ranges)
                axes[1, 2].plot(changes_pct, sensitivity_results['output_sensitivity'][most_sensitive_output], 
                               'r-', linewidth=2, label=f'{output_vars[most_sensitive_output]} (���)')
            
            axes[1, 2].set_xlabel('��ȭ�� (%)')
            axes[1, 2].set_ylabel('ȿ���� ����')
            axes[1, 2].set_title(f'{target_dmu} �ΰ��� �м�')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].axhline(y=primary_scores[sensitivity_dmu], color='green', linestyle='--', alpha=0.5)
        else:
            axes[1, 2].text(0.5, 0.5, '�ΰ��� �м�\n���� �߻�', 
                           ha='center', va='center', transform=axes[1, 2].transAxes,
                           fontsize=12, color='red')
            axes[1, 2].set_title('�ΰ��� �м�')
        
        plt.tight_layout()
        plt.show()
        
        print("   ? �ð�ȭ �Ϸ�")
    
    # 8. ��� �� �ǰ����
    print(f"\n" + "="*80)
    print("8. �м� ��� �� �ǰ����")
    print("="*80)
    
    print(f"\n?? �м� ���:")
    print(f"   ? ��ü DMU �� {len(efficient_dmus)}���� ȿ���� ({primary_model} ����)")
    print(f"   ? ��� ȿ����: {np.mean(primary_scores):.4f}")
    
    if scale_efficiency is not None:
        print(f"   ? ��� �Ը�ȿ����: {np.mean(scale_efficiency):.4f}")
        rts_counts = pd.Series(returns_to_scale).value_counts()
        print(f"   ? �Ը���� ����:")
        for rts, count in rts_counts.items():
            rts_full = {'CRS': '�Ը���ͺҺ�', 'IRS': '�Ը��������', 'DRS': '�Ը���Ͱ���'}
            print(f"     - {rts_full.get(rts, rts)}: {count}��")
    
    print(f"   ? ������ �ʿ��� DMU: {len(inefficient_dmus)}��")
    
    print(f"\n?? �ֿ� �ǰ����:")
    if len(inefficient_dmus) > 0:
        worst_dmu_idx = np.argmin(primary_scores)
        worst_dmu = dea.dmu_names[worst_dmu_idx]
        print(f"   1. �켱 ���� ���: {worst_dmu} (ȿ����: {primary_scores[worst_dmu_idx]:.4f})")
        
        if benchmark_count:
            best_benchmark = max(benchmark_count.items(), key=lambda x: x[1])[0]
            print(f"   2. �ֿ� ��ġ��ũ ��: {best_benchmark}")
        
        print(f"   3. {orientation.upper()} ���� ������ ����")
        
        # ���� ���� ȿ���� ū ���� �ĺ�
        if orientation == 'input':
            avg_input_slack = np.mean(input_slacks[inefficient_indices], axis=0)
            if np.any(avg_input_slack > 0):
                max_slack_input = np.argmax(avg_input_slack)
                print(f"   4. �켱 ���� ����: {input_vars[max_slack_input]} (��� ����: {avg_input_slack[max_slack_input]:.4f})")
        else:
            avg_output_slack = np.mean(output_slacks[inefficient_indices], axis=0)
            if np.any(avg_output_slack > 0):
                max_slack_output = np.argmax(avg_output_slack)
                print(f"   4. �켱 ���� ����: {output_vars[max_slack_output]} (��� ����: {avg_output_slack[max_slack_output]:.4f})")
    else:
        print(f"   ? ��� DMU�� ȿ�������� ��ǰ� �ֽ��ϴ�!")
    
    print(f"\n" + "="*80)
    print("                     �м� �Ϸ�")
    print("="*80)
    
    # ��� ��ȯ
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