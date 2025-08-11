import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm, halfnorm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
import warnings
warnings.filterwarnings('ignore')

class _StochasticFrontierAnalyzer:
    """
    Ȯ����������Լ� ������ ���꼺 ������ ���� ���� Ŭ���� (����ڰ� ���� ȣ������ ����)
    """
    
    def __init__(self, data, output_var, input_vars, time_var='t', id_var='id', include_time=True):
        self.data = data.copy()
        self.output_var = output_var
        self.input_vars = input_vars
        self.time_var = time_var
        self.id_var = id_var
        self.include_time = include_time
        
        # ��� ����� ��ųʸ�
        self.results = {}
        self.normalized_data = None
        self.translog_vars = None
        
        # ������ ����
        self._validate_data()
        
    def _validate_data(self):
        """������ ��ȿ�� �˻�"""
        required_vars = [self.output_var] + self.input_vars + [self.id_var]
        if self.include_time:
            required_vars.append(self.time_var)
            
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"���� �������� �����Ϳ� �����ϴ�: {missing_vars}")
            
        # �α� ��ȯ�� ���� ��� üũ
        for var in [self.output_var] + self.input_vars:
            if (self.data[var] <= 0).any():
                raise ValueError(f"���� {var}�� 0 ������ ���� �ֽ��ϴ�. �α� ��ȯ�� �Ұ����մϴ�.")
    
    def exploratory_data_analysis(self):
        """Ž���� ������ �м� ����"""
        print("=" * 60)
        print("Ž���� ������ �м� (EDA)")
        print("=" * 60)
        
        # �⺻ ������
        analysis_vars = [self.output_var] + self.input_vars
        if self.include_time:
            analysis_vars.append(self.time_var)
        
        # 1. �����跮
        print("\n1. �����跮")
        print("-" * 40)
        desc_stats = self.data[analysis_vars].describe()
        print(desc_stats.round(4))
        
        # 2. �������
        print("\n2. ������� ���")
        print("-" * 40)
        corr_matrix = self.data[analysis_vars].corr()
        print(corr_matrix.round(4))
        
        return desc_stats, corr_matrix
    
    def normalize_data(self):
        """��ü�� ������� ������ ǥ��ȭ"""
        print("\n������ ǥ��ȭ ���� ��...")
        
        self.normalized_data = self.data.copy()
        
        # ��ü�� ��� ��� (����/���� ������)
        vars_to_normalize = [self.output_var] + self.input_vars
        
        for var in vars_to_normalize:
            # ��ü�� ���
            mean_by_id = self.data.groupby(self.id_var)[var].transform('mean')
            # ǥ��ȭ
            self.normalized_data[f'nm_{var}'] = self.data[var] / mean_by_id
            # �α� ��ȯ
            self.normalized_data[f'ln_{var}'] = np.log(self.normalized_data[f'nm_{var}'])
        
        if self.include_time:
            # �ð� ������ �α� ��ȯ ���� �״�� ��� (1, 2, 3, ...)
            self.normalized_data[self.time_var] = self.data[self.time_var]
        
        print("������ ǥ��ȭ �Ϸ�")
    
    def create_translog_variables(self):
        """�ʿ���� �Լ��� ���� ������ ���� ����"""
        print("�ʿ���� �Լ� ���� ���� ��...")
        
        if self.normalized_data is None:
            self.normalize_data()
        
        # �α� ��ȯ�� ���� ������ (�ð� ����)
        ln_input_vars = [f'ln_{var}' for var in self.input_vars]
        
        # ��� ȸ�ͺ��� (���Ժ��� + �ð�����)
        all_vars = ln_input_vars.copy()
        if self.include_time:
            all_vars.append(self.time_var)  # �ð��� �׳� t
        
        # 2���� ����
        for var in all_vars:
            self.normalized_data[f'{var}2'] = 0.5 * self.normalized_data[var] ** 2
        
        # ������ ���� (���Ժ����� ��)
        n_inputs = len(ln_input_vars)
        for i in range(n_inputs):
            for j in range(i+1, n_inputs):
                var1 = ln_input_vars[i]
                var2 = ln_input_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.normalized_data[var_name] = self.normalized_data[var1] * self.normalized_data[var2]
        
        # �ð��� ���Ժ����� ������ (�����ȭ)
        if self.include_time:
            for var in ln_input_vars:
                var_name = f'{var}_{self.time_var}'
                self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[self.time_var]
        
        # ȸ�ͺм��� ���� ����Ʈ ����
        self.translog_vars = []
        
        # 1����
        self.translog_vars.extend(all_vars)
        
        # 2����
        for var in all_vars:
            self.translog_vars.append(f'{var}2')
        
        # ���Ժ����� �� ������
        for i in range(n_inputs):
            for j in range(i+1, n_inputs):
                var1 = ln_input_vars[i]
                var2 = ln_input_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.translog_vars.append(var_name)
        
        # �ð�-���� ������
        if self.include_time:
            for var in self.input_vars:
                self.translog_vars.append(f'ln_{var}_{self.time_var}')
        
        print(f"������ ���� ��: {len(self.translog_vars)}")
        print("�ʿ���� �Լ� ���� ���� �Ϸ�")
    
    def estimate_ols(self):
        """OLS ���� (�ʱⰪ��)"""
        print("\nOLS ���� ���� ��...")
        
        if self.translog_vars is None:
            self.create_translog_variables()
        
        # ���Ӻ���
        y = self.normalized_data[f'ln_{self.output_var}']
        
        # ��������
        X = self.normalized_data[self.translog_vars]
        X = sm.add_constant(X)
        
        # OLS ����
        ols_model = sm.OLS(y, X).fit()
        
        self.results['ols'] = ols_model
        
        print("OLS ���� �Ϸ�")
        print(f"R-squared: {ols_model.rsquared:.4f}")
        
        return ols_model
    
    def estimate_stochastic_frontier(self, distribution='half_normal'):
        """Ȯ�������Լ� ����"""
        print(f"\nȮ�������Լ� ���� ���� �� (����: {distribution})...")
        
        if 'ols' not in self.results:
            self.estimate_ols()
        
        # �ʱⰪ ����
        ols_params = self.results['ols'].params.values
        
        # ���Ӻ����� ��������
        y = self.normalized_data[f'ln_{self.output_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])  # ����� �߰�
        
        # �ִ�쵵����
        if distribution == 'half_normal':
            result = self._ml_estimation_half_normal(y, X, ols_params)
        else:
            raise ValueError("����� half-normal ������ �����մϴ�.")
        
        self.results['frontier'] = result
        
        print("Ȯ�������Լ� ���� �Ϸ�")
        
        return result
    
    def _ml_estimation_half_normal(self, y, X, initial_params):
        """Half-normal ������ ������ �ִ�쵵����"""
        
        def log_likelihood(params):
            n_beta = X.shape[1]
            beta = params[:n_beta]
            sigma_u = np.exp(params[n_beta])  # ��� ����
            sigma_v = np.exp(params[n_beta + 1])  # ��� ����
            
            # ����
            residuals = y - X @ beta
            
            # ���տ����� �л�
            sigma_sq = sigma_u**2 + sigma_v**2
            sigma = np.sqrt(sigma_sq)
            
            # ����
            lambd = sigma_u / sigma_v
            
            # �α׿쵵�Լ�
            epsilon_star = residuals * lambd / sigma
            
            log_phi = norm.logpdf(residuals / sigma)
            log_Phi = norm.logcdf(-epsilon_star)
            
            log_likelihood = (np.log(2) - np.log(sigma) + log_phi + log_Phi).sum()
            
            return -log_likelihood  # �ּ�ȭ�� ���� ����
        
        # �ʱⰪ ����
        initial_sigma_u = 0.1
        initial_sigma_v = 0.1
        initial_vals = np.concatenate([initial_params, [np.log(initial_sigma_u), np.log(initial_sigma_v)]])
        
        # ����ȭ
        result = minimize(log_likelihood, initial_vals, method='BFGS')
        
        # ǥ�ؿ��� ��� (������ �ٻ�)
        try:
            eps = 1e-6
            n_params = len(result.x)
            hessian_diag = np.zeros(n_params)
            
            f0 = log_likelihood(result.x)
            
            for i in range(n_params):
                x_plus = result.x.copy()
                x_minus = result.x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                
                f_plus = log_likelihood(x_plus)
                f_minus = log_likelihood(x_minus)
                
                hessian_diag[i] = (f_plus - 2*f0 + f_minus) / (eps**2)
            
            std_errors = np.sqrt(1 / np.abs(hessian_diag))
            std_errors[std_errors > 1000] = np.nan
                
        except:
            std_errors = np.full(len(result.x), np.nan)
        
        # ��� ����
        n_beta = X.shape[1]
        estimated_params = {
            'beta': result.x[:n_beta],
            'sigma_u': np.exp(result.x[n_beta]),
            'sigma_v': np.exp(result.x[n_beta + 1]),
            'log_likelihood': -result.fun,
            'success': result.success,
            'std_errors': std_errors,
            'n_obs': len(y),
            'n_params': len(result.x)
        }
        
        return estimated_params
    
    def calculate_efficiency(self):
        """����� ȿ���� ���"""
        print("\n����� ȿ���� ��� ��...")
        
        if 'frontier' not in self.results:
            self.estimate_stochastic_frontier()
        
        # �Ķ���� ����
        beta = self.results['frontier']['beta']
        sigma_u = self.results['frontier']['sigma_u']
        sigma_v = self.results['frontier']['sigma_v']
        
        # ���Ӻ����� ��������
        y = self.normalized_data[f'ln_{self.output_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        
        # ����
        residuals = y - X @ beta
        
        # ȿ���� ��� (Jondrow et al., 1982)
        sigma_sq = sigma_u**2 + sigma_v**2
        sigma = np.sqrt(sigma_sq)
        lambd = sigma_u / sigma_v
        
        mu_star = -residuals * sigma_u**2 / sigma_sq
        sigma_star = sigma_u * sigma_v / sigma
        
        # ���Ǻ� ���
        ratio = mu_star / sigma_star
        efficiency = np.exp(-mu_star + 0.5 * sigma_star**2) * (1 - norm.cdf(-ratio + sigma_star)) / (1 - norm.cdf(-ratio))
        
        self.normalized_data['efficiency'] = efficiency
        
        print("����� ȿ���� ��� �Ϸ�")
        
        return efficiency
    
    def calculate_productivity_components(self):
        """�ѿ�һ��꼺 �������� ���"""
        print("\n�ѿ�һ��꼺 �������� ��� ��...")
        
        if 'efficiency' not in self.normalized_data.columns:
            self.calculate_efficiency()
        
        # �Ķ����
        beta = self.results['frontier']['beta']
        
        # ����ź�¼� ���
        self._calculate_output_elasticities(beta)
        
        # �����ȭ ���
        if self.include_time:
            self._calculate_technical_change(beta)
        
        # �Ը��� ���� ���
        self._calculate_scale_effects()
        
        # ����� ȿ���� ��ȭ ���
        self._calculate_efficiency_change()
        
        # �ѿ�һ��꼺 ���
        self._calculate_tfp()
        
        print("�ѿ�һ��꼺 �������� ��� �Ϸ�")
    
    def _calculate_output_elasticities(self, beta):
        """����ź�¼� ���"""
        data = self.normalized_data
        
        for i, var in enumerate(self.input_vars):
            ln_var = f'ln_{var}'
            
            # 1���� ���
            beta_idx = 1 + i  # ����� ����
            elasticity = beta[beta_idx]
            
            # 2���� �߰�
            var2_idx = self.translog_vars.index(f'{ln_var}2')
            elasticity += beta[1 + var2_idx] * data[ln_var]
            
            # ������ �߰� (�ٸ� ���Ժ������)
            for j, other_var in enumerate(self.input_vars):
                if i != j:
                    other_ln_var = f'ln_{other_var}'
                    cross_term = f'{ln_var}_{other_var}' if i < j else f'{other_ln_var}_{var}'
                    if cross_term in self.translog_vars:
                        cross_idx = self.translog_vars.index(cross_term)
                        elasticity += beta[1 + cross_idx] * data[other_ln_var]
            
            # �ð����� ������
            if self.include_time:
                time_cross = f'{ln_var}_{self.time_var}'
                if time_cross in self.translog_vars:
                    time_cross_idx = self.translog_vars.index(time_cross)
                    elasticity += beta[1 + time_cross_idx] * data[self.time_var]
            
            data[f'eta_{var}'] = elasticity
    
    def _calculate_technical_change(self, beta):
        """�����ȭ ���"""
        if not self.include_time:
            return
        
        data = self.normalized_data
        
        # �ð��� 1���� ��� ã��
        time_idx = self.translog_vars.index(self.time_var)
        tech_change = beta[1 + time_idx]  # ����� ����
        
        # �ð��� 2����
        time2_idx = self.translog_vars.index(f'{self.time_var}2')
        tech_change += beta[1 + time2_idx] * data[self.time_var]
        
        # �ð��� ���Ժ����� ������
        for var in self.input_vars:
            time_cross = f'ln_{var}_{self.time_var}'
            if time_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_cross)
                tech_change += beta[1 + cross_idx] * data[f'ln_{var}']
        
        data['tech_change'] = tech_change
    
    def _calculate_scale_effects(self):
        """�Ը��� ���� ȿ�� ���"""
        data = self.normalized_data
        
        # �� ����ź�¼� (�Ը��� ����)
        rts = sum(data[f'eta_{var}'] for var in self.input_vars)
        data['rts'] = rts
        
        # �� ����� ���� ���
        for var in self.input_vars:
            data[f'lambda_{var}'] = data[f'eta_{var}'] / rts
        
        # ���������� ���
        data_sorted = data.sort_values([self.id_var, self.time_var])
        for var in self.input_vars:
            ln_var = f'ln_{var}'
            data_sorted[f'gr_{var}'] = data_sorted.groupby(self.id_var)[ln_var].diff()
        
        # �Ը�ȿ�� ���
        scale_effect = (rts - 1) * sum(
            data_sorted[f'lambda_{var}'] * data_sorted[f'gr_{var}'] 
            for var in self.input_vars
        )
        
        data_sorted['scale_effect'] = scale_effect
        self.normalized_data = data_sorted
    
    def _calculate_efficiency_change(self):
        """����� ȿ���� ��ȭ ���"""
        data = self.normalized_data.sort_values([self.id_var, self.time_var])
        
        # ȿ���� ��ȭ��
        data['efficiency_change'] = data.groupby(self.id_var)['efficiency'].pct_change()
        
        self.normalized_data = data
    
    def _calculate_tfp(self):
        """�ѿ�һ��꼺 ���"""
        data = self.normalized_data
        
        # TFP = �Ը�ȿ�� + �����ȭ + ȿ������ȭ
        tfp_components = ['scale_effect']
        
        if self.include_time:
            tfp_components.append('tech_change')
        
        if 'efficiency_change' in data.columns:
            tfp_components.append('efficiency_change')
        
        data['tfp'] = sum(data[comp].fillna(0) for comp in tfp_components)
        
        self.normalized_data = data
    
    def print_results(self, save_path='tfp_results.csv'):
        """��� ���"""
        print("\n" + "=" * 100)
        print("Ȯ����������Լ� ���� ���")
        print("=" * 100)
        
        # 1. �Ķ���� ����ġ ���
        if 'frontier' in self.results:
            frontier = self.results['frontier']
            beta = frontier['beta']
            std_errors = frontier.get('std_errors', np.full(len(beta), np.nan))
            
            print("\n1. �Ķ���� ����ġ")
            print("-" * 100)
            print(f"{'������':<20} {'���':<12} {'ǥ�ؿ���':<12} {'t-��':<10} {'p-��':<10} {'���Ǽ�':<8}")
            print("-" * 100)
            
            # �����
            t_stat = beta[0] / std_errors[0] if not np.isnan(std_errors[0]) and std_errors[0] != 0 else np.nan
            p_value = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
            
            print(f"{'�����':<20} {beta[0]:>8.6f} {std_errors[0]:>8.4f} {t_stat:>8.3f} {p_value:>8.4f} {significance:<8}")
            
            # ������ ���
            for i, var_name in enumerate(self.translog_vars):
                coef = beta[i + 1]  # ����� ����
                se = std_errors[i + 1] if i + 1 < len(std_errors) else np.nan
                t_stat = coef / se if not np.isnan(se) and se != 0 else np.nan
                p_value = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
                significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                
                print(f"{var_name:<20} {coef:>8.6f} {se:>8.4f} {t_stat:>8.3f} {p_value:>8.4f} {significance:<8}")
            
            print("-" * 100)
            print("���Ǽ�: *** p<0.01, ** p<0.05, * p<0.1")
            
            # 2. �� ��跮
            print(f"\n2. �� ��跮")
            print("-" * 40)
            print(f"������: {frontier.get('n_obs', 'N/A'):>20}")
            print(f"�Ķ���� ��: {frontier.get('n_params', 'N/A'):>15}")
            print(f"�α׿쵵: {frontier['log_likelihood']:>15.4f}")
            print(f"�ñ׸�_u: {frontier['sigma_u']:>15.4f}")
            print(f"�ñ׸�_v: {frontier['sigma_v']:>15.4f}")
            print(f"�ñ׸���: {frontier['sigma_u']**2 + frontier['sigma_v']**2:>15.4f}")
            print(f"���� (��u/��v): {frontier['sigma_u']/frontier['sigma_v']:>10.4f}")
            print(f"�� = ��u��/���: {frontier['sigma_u']**2/(frontier['sigma_u']**2 + frontier['sigma_v']**2):>13.4f}")
        
        # 3. ȿ���� ���
        if 'efficiency' in self.normalized_data.columns:
            eff = self.normalized_data['efficiency']
            print(f"\n3. ����� ȿ���� ���")
            print("-" * 40)
            print(f"���: {eff.mean():>20.4f}")
            print(f"ǥ������: {eff.std():>15.4f}")
            print(f"�ּڰ�: {eff.min():>16.4f}")
            print(f"�ִ�: {eff.max():>16.4f}")
            print(f"������: {eff.median():>16.4f}")
        
        # 4. �ѿ�һ��꼺 ���� ��� - ��ü �ø��� ���
        if 'tfp' in self.normalized_data.columns:
            print(f"\n4. �ѿ�һ��꼺 ���� ��� (��ü �ø���)")
            print("=" * 80)
            
            # �ð����� ����
            tfp_data = self.normalized_data.sort_values([self.time_var, self.id_var]).copy()
            
            # ��ü�� ID�� �ð� ����
            display_cols = [self.id_var, self.time_var]
            
            # TFP ������ҵ�
            tfp_components = ['tfp']
            if 'scale_effect' in tfp_data.columns:
                tfp_components.append('scale_effect')
            if 'tech_change' in tfp_data.columns:
                tfp_components.append('tech_change')
            if 'efficiency_change' in tfp_data.columns:
                tfp_components.append('efficiency_change')
            
            display_cols.extend(tfp_components)
            
            # ����ġ ����
            tfp_display = tfp_data[display_cols].dropna()
            
            if len(tfp_display) > 0:
                # �÷��� ���� (������)
                col_rename = {
                    'tfp': 'TFP',
                    'scale_effect': '�Ը�ȿ��',
                    'tech_change': '�����ȭ', 
                    'efficiency_change': 'ȿ������ȭ'
                }
                
                tfp_display_renamed = tfp_display.rename(columns=col_rename)
                
                # **��ü ������ ���**
                print("��ü TFP ���� ���:")
                print(tfp_display_renamed.round(6).to_string(index=False))
                print(f"\n�� {len(tfp_display_renamed)}�� ����ġ")
                
                print("\n" + "-" * 60)
                print("TFP ������� ������")
                print("-" * 60)
                print(f"{'�������':<15} {'���':<12} {'ǥ������':<12} {'�ּڰ�':<12} {'�ִ�':<12}")
                print("-" * 60)
                
                for comp in tfp_components:
                    if comp in tfp_data.columns:
                        comp_data = tfp_data[comp].dropna()
                        if len(comp_data) > 0:
                            comp_name = col_rename.get(comp, comp)
                            print(f"{comp_name:<15} {comp_data.mean():>8.6f} {comp_data.std():>12.6f} {comp_data.min():>12.6f} {comp_data.max():>12.6f}")
                
                print("-" * 60)
                
                # �ð��� ���
                if self.include_time and len(tfp_display) > 0:
                    print(f"\n�ð��� ���:")
                    time_avg = tfp_data.groupby(self.time_var)[tfp_components].mean()
                    time_avg_renamed = time_avg.rename(columns=col_rename)
                    print(time_avg_renamed.round(6).to_string())
                
                # ID�� ��� (��ü�� ��)
                print(f"\nID�� ��� (��ü�� ��):")
                print("-" * 60)
                id_avg = tfp_data.groupby(self.id_var)[tfp_components].mean()
                id_avg_renamed = id_avg.rename(columns=col_rename)
                print(id_avg_renamed.round(6).to_string())
                
                # ȿ������ ID���� �����ֱ�
                if 'efficiency' in tfp_data.columns:
                    print(f"\nID�� ��� ȿ����:")
                    id_eff = tfp_data.groupby(self.id_var)['efficiency'].mean()
                    print(f"{'ID':<5} {'���ȿ����':<12}")
                    print("-" * 20)
                    for id_val, eff_val in id_eff.items():
                        print(f"{id_val:<5} {eff_val:>8.6f}")
                
                # TFP ������ ���� (������ ��ο�)
                print(f"\n?? TFP ������ ����: {save_path}")
                tfp_display_renamed.to_csv(save_path, index=False, encoding='utf-8-sig')
                print("���� �Ϸ�!")
            
            else:
                print("TFP �����Ͱ� �����ϴ�")
        
        # 5. ����ź�¼� ���
        elasticity_vars = [f'eta_{var}' for var in self.input_vars if f'eta_{var}' in self.normalized_data.columns]
        if elasticity_vars:
            print(f"\n5. ����ź�¼� ���")
            print("-" * 50)
            print(f"{'���Կ��':<10} {'���ź�¼�':<15} {'ǥ������':<15}")
            print("-" * 50)
            
            for var in self.input_vars:
                eta_var = f'eta_{var}'
                if eta_var in self.normalized_data.columns:
                    eta_data = self.normalized_data[eta_var]
                    print(f"{var.upper():<10} {eta_data.mean():>10.6f} {eta_data.std():>15.6f}")
            
            # �Ը��� ����
            if 'rts' in self.normalized_data.columns:
                rts_data = self.normalized_data['rts']
                rts_mean = rts_data.mean()
                print("-" * 50)
                print(f"{'�Ը��հ�':<10} {rts_mean:>10.6f} {rts_data.std():>15.6f}")
                
                if rts_mean > 1.01:
                    print("  �� �Ը���� ���� (Increasing Returns to Scale)")
                elif rts_mean < 0.99:
                    print("  �� �Ը���� ���� (Decreasing Returns to Scale)")
                else:
                    print("  �� �Ը���� �Һ� (Constant Returns to Scale)")
        
        print("\n" + "=" * 100)
    
    def plot_results(self):
        """��� �ð�ȭ"""
        if 'efficiency' not in self.normalized_data.columns:
            print("ȿ������ ������ �ʾҽ��ϴ�.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ȿ���� ����
        axes[0, 0].hist(self.normalized_data['efficiency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('����� ȿ���� ����')
        axes[0, 0].set_xlabel('ȿ����')
        axes[0, 0].set_ylabel('��')
        
        # ȿ���� �ð迭
        if self.include_time:
            eff_by_time = self.normalized_data.groupby(self.time_var)['efficiency'].mean()
            axes[0, 1].plot(eff_by_time.index, eff_by_time.values, marker='o')
            axes[0, 1].set_title('�ð��� ��� ȿ����')
            axes[0, 1].set_xlabel('�ð�')
            axes[0, 1].set_ylabel('��� ȿ����')
        
        # TFP ����
        if 'tfp' in self.normalized_data.columns:
            tfp_data = self.normalized_data['tfp'].dropna()
            if len(tfp_data) > 0:
                axes[1, 0].hist(tfp_data, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('�ѿ�һ��꼺 ����')
                axes[1, 0].set_xlabel('TFP')
                axes[1, 0].set_ylabel('��')
        
        # TFP �������
        if all(col in self.normalized_data.columns for col in ['scale_effect', 'tech_change']):
            components = ['scale_effect']
            comp_labels = ['�Ը�ȿ��']
            
            if self.include_time:
                components.append('tech_change')
                comp_labels.append('�����ȭ')
            
            comp_means = [self.normalized_data[comp].dropna().mean() for comp in components]
            axes[1, 1].bar(comp_labels, comp_means)
            axes[1, 1].set_title('TFP ������� ���')
            axes[1, 1].set_ylabel('�⿩��')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self, save_path='tfp_results.csv'):
        """��ü �м� ����"""
        print("Ȯ����������Լ� �м��� �����մϴ�...")
        
        # 1. EDA
        self.exploratory_data_analysis()
        
        # 2. ������ �غ�
        self.normalize_data()
        self.create_translog_variables()
        
        # 3. ����
        self.estimate_ols()
        self.estimate_stochastic_frontier()
        
        # 4. ���꼺 �м�
        self.calculate_efficiency()
        self.calculate_productivity_components()
        
        # 5. ��� ���
        self.print_results(save_path)
        self.plot_results()
        
        print("\n�м��� �Ϸ�Ǿ����ϴ�!")
        
        return self.results, self.normalized_data


# Ȯ����������Լ� �м� �Լ�
def Run_StochasticFrontierProduction(data, output_var, input_vars, time_var='t', id_var='id', include_time=True, save_path='tfp_results.csv'):
    """
    Ȯ����������Լ� �м� ���� �Լ� - �� ���� ��� ��� ���
    
    ����:
    -------
    # �⺻ ���
    StochasticFrontierProduction(data, 'y', ['l', 'k', 'm'])
    
    # ���� ��ġ ����
    StochasticFrontierProduction(data, 'y', ['l', 'k', 'm'], save_path='my_analysis.csv')
    
    # �ð� Ʈ���� ����
    StochasticFrontierProduction(data, 'y', ['l', 'k'], include_time=False)
    
    ���:
    -----
    - ��� �м� ����� ȭ�鿡 ��µ�
    - TFP ���� ����� ������ ��η� �����
    - ID�� ����� �߰��� ��µ�
    """
    
    analyzer = _StochasticFrontierAnalyzer(
        data=data,
        output_var=output_var,
        input_vars=input_vars,
        time_var=time_var,
        id_var=id_var,
        include_time=include_time
    )
    
    analyzer.run_complete_analysis(save_path)
    
    return analyzer

