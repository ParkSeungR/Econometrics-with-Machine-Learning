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

class _StochasticFrontierCostAnalyzer:
    """
    Ȯ���������Լ� �м�, ��� ȿ����, TFP �������� ���ظ� ���� ���� Ŭ����
    (����ڰ� ���� ȣ������ ����)
    """
    
    def __init__(self, data, cost_var, price_vars, input_vars, output_var, time_var='t', id_var='id', include_time=True):
        self.data = data.copy()
        self.cost_var = cost_var  # �Ѻ��
        self.price_vars = price_vars  # ��Ұ��ݵ� 
        self.input_vars = input_vars  # ���Է���
        self.output_var = output_var  # ���ⷮ
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
        required_vars = [self.cost_var, self.output_var] + self.price_vars + self.input_vars + [self.id_var]
        if self.include_time:
            required_vars.append(self.time_var)
            
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"���� �������� �����Ϳ� �����ϴ�: {missing_vars}")
        
        # ���ݺ����� ���Է� ������ ������ ������ Ȯ��
        if len(self.price_vars) != len(self.input_vars):
            raise ValueError(f"���ݺ��� ����({len(self.price_vars)})�� ���Է����� ����({len(self.input_vars)})�� �ٸ��ϴ�.")
            
        # �α� ��ȯ�� ���� ��� üũ
        for var in [self.cost_var, self.output_var] + self.price_vars + self.input_vars:
            if (self.data[var] <= 0).any():
                raise ValueError(f"���� {var}�� 0 ������ ���� �ֽ��ϴ�. �α� ��ȯ�� �Ұ����մϴ�.")
        
        print(f"? ������ ���� �Ϸ�:")
        print(f"   ���ݺ���: {self.price_vars}")
        print(f"   ���Է�����: {self.input_vars}")
        print(f"   �Ѻ�뺯��: {self.cost_var}")
        print(f"   ���ⷮ����: {self.output_var}")
    
    def exploratory_data_analysis(self):
        """Ž���� ������ �м� ����"""
        print("=" * 60)
        print("Ž���� ������ �м� (EDA) - Ȯ���������Լ�")
        print("=" * 60)
        
        # �⺻ ������
        analysis_vars = [self.cost_var, self.output_var] + self.price_vars
        if self.include_time:
            analysis_vars.append(self.time_var)
        
        # 1. ������跮
        print("\n1. ������跮")
        print("-" * 40)
        desc_stats = self.data[analysis_vars].describe()
        print(desc_stats.round(4))
        
        # 2. �������
        print("\n2. ������� ��ȸ")
        print("-" * 40)
        corr_matrix = self.data[analysis_vars].corr()
        print(corr_matrix.round(4))
        
        return desc_stats, corr_matrix
    
    def normalize_data(self):
        """��ü�� ������� ������ ǥ��ȭ"""
        print("\n������ ǥ��ȭ ���� ��...")
        
        self.normalized_data = self.data.copy()
        
        # ��ü�� ��� ��� (���, ����, ���ⷮ)
        vars_to_normalize = [self.cost_var, self.output_var] + self.price_vars
        
        for var in vars_to_normalize:
            # ��ü�� ���
            mean_by_id = self.data.groupby(self.id_var)[var].transform('mean')
            # ǥ��ȭ
            self.normalized_data[f'nm_{var}'] = self.data[var] / mean_by_id
            # �α� ��ȯ
            self.normalized_data[f'ln_{var}'] = np.log(self.normalized_data[f'nm_{var}'])
        
        # ���Է��� ���� (���� ����)
        for input_var in self.input_vars:
            self.normalized_data[input_var] = self.data[input_var]
        
        if self.include_time:
            # �ð� ������ �α� ��ȯ ���� �״�� ��� (1, 2, 3, ...)
            self.normalized_data[self.time_var] = self.data[self.time_var]
        
        print("������ ǥ��ȭ �Ϸ�")
    
    def create_translog_variables(self):
        """�ʿ���� ����Լ��� ���� ������ ���� ����"""
        print("�ʿ���� ����Լ� ���� ���� ��...")
        
        if self.normalized_data is None:
            self.normalize_data()
        
        # �α� ��ȯ�� ���� ������� ���ⷮ
        ln_price_vars = [f'ln_{var}' for var in self.price_vars]
        ln_output_var = f'ln_{self.output_var}'
        
        # ��� �ٽɺ��� (���� + ���ⷮ + �ð�����)
        all_vars = ln_price_vars + [ln_output_var]
        if self.include_time:
            all_vars.append(self.time_var)  # �ð��� �׳� t
        
        # 2���� ����
        for var in all_vars:
            self.normalized_data[f'{var}2'] = 0.5 * self.normalized_data[var] ** 2
        
        # ���ݺ����� �� ������
        n_prices = len(ln_price_vars)
        for i in range(n_prices):
            for j in range(i+1, n_prices):
                var1 = ln_price_vars[i]
                var2 = ln_price_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.normalized_data[var_name] = self.normalized_data[var1] * self.normalized_data[var2]
        
        # ���ݰ� ���ⷮ�� ������
        for var in ln_price_vars:
            var_name = f'{var}_{self.output_var}'
            self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[ln_output_var]
        
        # �ð��� �ٸ� �������� ������
        if self.include_time:
            for var in ln_price_vars + [ln_output_var]:
                var_name = f'{var}_{self.time_var}'
                self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[self.time_var]
        
        # �ٽɺм��� ���� ����Ʈ ����
        self.translog_vars = []
        
        # 1����
        self.translog_vars.extend(all_vars)
        
        # 2����
        for var in all_vars:
            self.translog_vars.append(f'{var}2')
        
        # ���ݺ����� �� ������
        for i in range(n_prices):
            for j in range(i+1, n_prices):
                var1 = ln_price_vars[i]
                var2 = ln_price_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.translog_vars.append(var_name)
        
        # ����-���ⷮ ������
        for var in self.price_vars:
            self.translog_vars.append(f'ln_{var}_{self.output_var}')
        
        # �ð� ������
        if self.include_time:
            for var in self.price_vars + [self.output_var]:
                self.translog_vars.append(f'ln_{var}_{self.time_var}')
        
        print(f"������ ���� ��: {len(self.translog_vars)}")
        print("�ʿ���� ����Լ� ���� ���� �Ϸ�")
    
    def estimate_ols(self):
        """OLS ���� (�ʱⰪ��)"""
        print("\nOLS ���� ���� ��...")
        
        if self.translog_vars is None:
            self.create_translog_variables()
        
        # ���Ӻ��� (�Ѻ��)
        y = self.normalized_data[f'ln_{self.cost_var}']
        
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
        """Ȯ���������Լ� ���� - ������ ����"""
        print(f"\nȮ���������Լ� ���� ���� �� (����: {distribution})...")
        
        if 'ols' not in self.results:
            print("OLS ������ ���� �����մϴ�...")
            self.estimate_ols()
        
        # OLS ��� ����
        if not hasattr(self.results['ols'], 'params'):
            print("? OLS ���� ����� �����ϴ�.")
            return self._create_fallback_result()
        
        # �ʱⰪ ����
        ols_params = self.results['ols'].params.values
        
        # OLS ��� ����
        if np.any(np.isnan(ols_params)) or np.any(np.isinf(ols_params)):
            print("?? OLS �Ķ���Ϳ� NaN/Inf�� �ֽ��ϴ�. �⺻���� ����մϴ�.")
            ols_params = np.zeros(len(ols_params))
            ols_params[0] = np.log(self.normalized_data[f'ln_{self.cost_var}'].mean())
        
        # ���Ӻ����� ��������
        y = self.normalized_data[f'ln_{self.cost_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])  # ����� �߰�
        
        # ������ ����
        if np.any(np.isnan(y)) or np.any(np.isnan(X)):
            print("? �����Ϳ� NaN�� �ֽ��ϴ�.")
            return self._create_fallback_result()
        
        print(f"   ������ ũ��: y={y.shape}, X={X.shape}")
        print(f"   OLS R��: {self.results['ols'].rsquared:.4f}")
        
        # �ִ�쵵����
        print("?? �ִ�쵵���� ����...")
        try:
            if distribution == 'half_normal':
                result = self._ml_estimation_half_normal_cost(y, X, ols_params)
            else:
                raise ValueError("����� half-normal ������ �����մϴ�.")
        except Exception as e:
            print(f"? �ִ�쵵���� �� ����: {str(e)}")
            return self._create_fallback_result()
        
        # ��� ����
        if result['success']:
            # �Ķ���� �ո��� ����
            if (result['sigma_u'] > 0 and result['sigma_v'] > 0 and 
                result['sigma_u'] < 100 and result['sigma_v'] < 100 and
                np.isfinite(result['log_likelihood'])):
                
                self.results['frontier'] = result
                print("? Ȯ���������Լ� ���� �Ϸ�")
                
                # �� ���յ� ����
                gamma = result['sigma_u']**2 / (result['sigma_u']**2 + result['sigma_v']**2)
                print(f"   �� = ���u/���: {gamma:.4f}")
                if gamma > 0.5:
                    print("   �� ��ȿ������ ������ �ֿ� �����Դϴ�.")
                else:
                    print("   �� Ȯ���� ������ �ֿ� �����Դϴ�.")
                
                return result
            else:
                print("?? ������ �Ķ���Ͱ� ���ո����Դϴ�.")
                return self._create_fallback_result()
        else:
            print(f"?? ���� ����: {result.get('message', 'Unknown error')}")
            return self._create_fallback_result()
    
    def _create_fallback_result(self):
        """���� ���н� ��ü ��� ����"""
        print("   ��ü ��� ���� ��...")
        
        # �⺻ �Ķ���� ����
        n_params = len(self.translog_vars) + 1  # ����� ����
        fallback_beta = np.zeros(n_params)
        
        # ������� ��� ������� ����
        fallback_beta[0] = self.normalized_data[f'ln_{self.cost_var}'].mean()
        
        # �⺻ sigma ��
        cost_std = self.normalized_data[f'ln_{self.cost_var}'].std()
        fallback_sigma_u = max(cost_std * 0.1, 0.01)
        fallback_sigma_v = max(cost_std * 0.1, 0.01)
        
        fallback_result = {
            'beta': fallback_beta,
            'sigma_u': fallback_sigma_u,
            'sigma_v': fallback_sigma_v,
            'log_likelihood': np.nan,
            'success': False,
            'std_errors': np.full(n_params + 2, np.nan),
            'n_obs': len(self.normalized_data),
            'n_params': n_params + 2,
            'message': 'Fallback parameters due to estimation failure'
        }
        
        self.results['frontier'] = fallback_result
        print("   ?? ��ü �Ķ���� ��� - ��� �ؼ��� �����ϼ���.")
        
        return fallback_result
    
    def _ml_estimation_half_normal_cost(self, y, X, initial_params):
        """����Լ��� Half-normal ������ ������ �ִ�쵵���� - ������ ����"""
        
        def log_likelihood(params):
            try:
                n_beta = X.shape[1]
                beta = params[:n_beta]
                log_sigma_u = params[n_beta]      
                log_sigma_v = params[n_beta + 1]  
                
                # sigma ��� (�� ������)
                sigma_u = np.exp(np.clip(log_sigma_u, -10, 5))  # �ذ� ����
                sigma_v = np.exp(np.clip(log_sigma_v, -10, 5))
                
                # �ּҰ� ����
                sigma_u = np.maximum(sigma_u, 1e-4)
                sigma_v = np.maximum(sigma_v, 1e-4)
                
                # ���� ���
                residuals = y - X @ beta
                
                # ������ ������ üũ
                if np.std(residuals) > 100 or np.std(residuals) < 1e-6:
                    return 1e8
                
                # ���տ��� �Ķ����
                sigma_sq = sigma_u**2 + sigma_v**2
                sigma = np.sqrt(sigma_sq)
                
                if sigma < 1e-4 or sigma > 100:
                    return 1e8
                
                # ���� (���� ����)
                lambd = np.clip(sigma_u / sigma_v, 0.01, 100)
                
                # ǥ��ȭ�� ����
                residuals_std = residuals / sigma
                residuals_std = np.clip(residuals_std, -8, 8)
                
                # epsilon* ��� (����Լ�)
                epsilon_star = residuals_std * lambd
                epsilon_star = np.clip(epsilon_star, -8, 8)
                
                # �α� Ȯ���е� ���
                log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * residuals_std**2
                
                # �α� �������� ��� (������ ���)
                log_Phi = np.where(epsilon_star > -5, 
                                  np.log(norm.cdf(epsilon_star) + 1e-15),
                                  epsilon_star - 0.5 * epsilon_star**2 - np.log(np.sqrt(2*np.pi)))
                
                # �α׿쵵 ���
                log_likelihood_val = (np.log(2) - np.log(sigma) + log_phi + log_Phi).sum()
                
                # ���� ����
                if not np.isfinite(log_likelihood_val) or log_likelihood_val < -1e6:
                    return 1e8
                
                return -log_likelihood_val
                
            except Exception as e:
                return 1e8
        
        # ������ �ʱⰪ ����
        print("   �ʱⰪ ���� ��...")
        ols_residuals = y - X @ initial_params
        ols_sigma = np.std(ols_residuals)
        
        # �� ū �ʱⰪ ��� (������ ������ ����)
        initial_sigma_u = np.clip(ols_sigma * 0.8, 0.05, 1.0)  # �� ū ��
        initial_sigma_v = np.clip(ols_sigma * 0.6, 0.05, 1.0)  # �� ū ��
        
        # log scale�� �ʱⰪ ����
        initial_vals = np.concatenate([
            initial_params, 
            [np.log(initial_sigma_u), np.log(initial_sigma_v)]
        ])
        
        print(f"   OLS ��: {ols_sigma:.4f}")
        print(f"   �ʱ� ��_u: {initial_sigma_u:.4f}, ��_v: {initial_sigma_v:.4f}")
        
        # �� ������ bounds ����
        n_beta = len(initial_params)
        bounds = []
        
        # beta �Ķ����: �� ���� ����
        for i in range(n_beta):
            bounds.append((-20, 20))  
        
        # log_sigma: �� ���� ����
        bounds.append((-8, 3))   # exp(-8) ? 0.0003, exp(3) ? 20
        bounds.append((-8, 3))
        
        # ����ȭ �õ� (�� ������ ����)
        print("   ����ȭ ����...")
        result = None
        
        # ��� 1: L-BFGS-B (�� ������ ����)
        try:
            print("   �õ� 1: L-BFGS-B")
            result = minimize(log_likelihood, initial_vals, method='L-BFGS-B', 
                            bounds=bounds, 
                            options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if result.success and result.fun < 1e7:
                print(f"   ? L-BFGS-B ����: f={result.fun:.4f}")
            else:
                print(f"   L-BFGS-B ����: f={result.fun:.4f}")
                result = None
        except Exception as e:
            print(f"   L-BFGS-B ����: {str(e)}")
            result = None
        
        # ��� 2: �ٸ� �ʱⰪ���� ��õ�
        if result is None:
            try:
                print("   �õ� 2: �ٸ� �ʱⰪ���� L-BFGS-B")
                # �ʱⰪ�� �ణ ����
                alt_initial_vals = initial_vals.copy()
                alt_initial_vals[-2:] += np.random.normal(0, 0.5, 2)  # sigma �ʱⰪ ����
                
                result = minimize(log_likelihood, alt_initial_vals, method='L-BFGS-B', 
                                bounds=bounds, 
                                options={'maxiter': 2000, 'ftol': 1e-6})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? ��� L-BFGS-B ����: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   ��� L-BFGS-B ����: {str(e)}")
                result = None
        
        # ��� 3: BFGS (bounds ����)
        if result is None:
            try:
                print("   �õ� 3: BFGS")
                result = minimize(log_likelihood, initial_vals, method='BFGS', 
                                options={'maxiter': 1500, 'gtol': 1e-4})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? BFGS ����: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   BFGS ����: {str(e)}")
                result = None
        
        # ��� 4: Powell (derivative-free)
        if result is None:
            try:
                print("   �õ� 4: Powell")
                result = minimize(log_likelihood, initial_vals, method='Powell', 
                                options={'maxiter': 1000, 'ftol': 1e-6})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? Powell ����: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   Powell ����: {str(e)}")
                result = None
        
        # ��� ��� ���н� �⺻�� ��ȯ
        if result is None:
            print("   ?? ��� ����ȭ ��� ���� - �⺻�� ���")
            result = type('obj', (object,), {
                'x': initial_vals, 
                'fun': 1e10, 
                'success': False,
                'message': 'All optimization methods failed'
            })()
        
        # ������ ǥ�ؿ��� ���
        print("   ǥ�ؿ��� ��� ��...")
        try:
            if result.success and result.fun < 1e9:
                # Hessian ��� (�� �������� ���)
                eps = 1e-5
                n_params = len(result.x)
                hessian = np.zeros((n_params, n_params))
                
                f0 = log_likelihood(result.x)
                
                # diagonal elements�� ��� (�� ������)
                for i in range(n_params):
                    x_plus = result.x.copy()
                    x_minus = result.x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    
                    f_plus = log_likelihood(x_plus)
                    f_minus = log_likelihood(x_minus)
                    
                    second_deriv = (f_plus - 2*f0 + f_minus) / (eps**2)
                    hessian[i, i] = abs(second_deriv)  # ���� ���
                
                # ǥ�ؿ��� ���
                std_errors = np.zeros(n_params)
                for i in range(n_params):
                    if hessian[i, i] > 1e-10:
                        std_errors[i] = 1.0 / np.sqrt(hessian[i, i])
                    else:
                        std_errors[i] = np.nan
                
                # ���ո����� ǥ�ؿ��� ����
                std_errors = np.where(std_errors > 100, np.nan, std_errors)
                
            else:
                std_errors = np.full(len(result.x), np.nan)
                
        except Exception as e:
            print(f"   ǥ�ؿ��� ��� ����: {str(e)}")
            std_errors = np.full(len(result.x), np.nan)
        
        # ��� ����
        n_beta = X.shape[1]
        estimated_params = {
            'beta': result.x[:n_beta],
            'sigma_u': np.exp(result.x[n_beta]),
            'sigma_v': np.exp(result.x[n_beta + 1]),
            'log_likelihood': -result.fun if result.fun < 1e9 else np.nan,
            'success': result.success and result.fun < 1e9,
            'std_errors': std_errors,
            'n_obs': len(y),
            'n_params': len(result.x),
            'message': getattr(result, 'message', 'No message'),
            'convergence_info': {
                'function_value': result.fun,
                'iterations': getattr(result, 'nit', 0),
                'method_used': getattr(result, 'method', 'Unknown')
            }
        }
        
        # ��� ���
        if estimated_params['success']:
            print(f"   ? ����ȭ ����!")
            print(f"   �α׿쵵: {estimated_params['log_likelihood']:.4f}")
            print(f"   ��_u: {estimated_params['sigma_u']:.4f}")
            print(f"   ��_v: {estimated_params['sigma_v']:.4f}")
            print(f"   �� (��_u/��_v): {estimated_params['sigma_u']/estimated_params['sigma_v']:.4f}")
        else:
            print(f"   ?? ����ȭ ����: {estimated_params['message']}")
            print(f"   �Լ���: {result.fun:.4f}")
        
        return estimated_params
    
    def calculate_efficiency(self):
        """����� ȿ����(Technical Efficiency) ��� - ������ ����"""
        print("\n����� ȿ����(TE) ��� ��...")
        
        if 'frontier' not in self.results:
            self.estimate_stochastic_frontier()
        
        # ������ ������ ��� ó��
        if not self.results['frontier']['success']:
            print("?? Ȯ�������Լ� ������ �����Ͽ� ȿ������ ����� �� �����ϴ�.")
            # �⺻������ 1.0 (���� ȿ����) �Ҵ�
            self.normalized_data['technical_efficiency'] = 1.0
            return np.ones(len(self.normalized_data))
        
        # �Ķ���� ����
        beta = self.results['frontier']['beta']
        sigma_u = self.results['frontier']['sigma_u']
        sigma_v = self.results['frontier']['sigma_v']
        
        # numerical stability üũ
        if sigma_u < 1e-6 or sigma_v < 1e-6:
            print("?? �� ���� �ʹ� �۾� ȿ���� ����� �ǳʶݴϴ�.")
            self.normalized_data['technical_efficiency'] = 1.0
            return np.ones(len(self.normalized_data))
        
        # ���Ӻ����� ��������
        y = self.normalized_data[f'ln_{self.cost_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        
        # ����
        residuals = y - X @ beta
        
        # ����� ȿ���� ��� (����Լ��� - Jondrow et al., 1982)
        sigma_sq = sigma_u**2 + sigma_v**2
        sigma = np.sqrt(sigma_sq)
        lambd = sigma_u / sigma_v
        
        # numerical stability�� ���� ó��
        try:
            mu_star = residuals * sigma_u**2 / sigma_sq  # ����Լ��� ��ȣ �ݴ�
            sigma_star = sigma_u * sigma_v / sigma
            
            # numerical stability: �ذ� ����
            mu_star = np.clip(mu_star, -10, 10)
            sigma_star = np.maximum(sigma_star, 1e-6)
            
            # ���Ǻ� ��� (����� ȿ����)
            ratio = mu_star / sigma_star
            ratio = np.clip(ratio, -10, 10)  # �ذ� ����
            
            # �������� ����� ���� �ܰ躰�� ���
            exp_term = np.exp(-mu_star + 0.5 * sigma_star**2)
            
            # CDF ��� �� numerical stability
            cdf_term1 = norm.cdf(ratio + sigma_star)
            cdf_term2 = norm.cdf(ratio)
            
            # 0���� ������ ����
            denominator = 1 - cdf_term2
            denominator = np.maximum(denominator, 1e-10)
            
            numerator = 1 - cdf_term1
            
            technical_efficiency = exp_term * numerator / denominator
            
            # ��� ���� �� ��ó��
            technical_efficiency = np.clip(technical_efficiency, 1e-6, 1.0)
            
            # NaN�̳� inf üũ
            invalid_mask = ~np.isfinite(technical_efficiency)
            if invalid_mask.any():
                print(f"?? {invalid_mask.sum()}�� ���������� ȿ���� ��� ���� - �⺻�� 0.5 �Ҵ�")
                technical_efficiency[invalid_mask] = 0.5
            
            self.normalized_data['technical_efficiency'] = technical_efficiency
            
            print("? ����� ȿ����(TE) ��� �Ϸ�")
            print(f"   ��� TE: {technical_efficiency.mean():.4f}")
            print(f"   �ּ� TE: {technical_efficiency.min():.4f}")
            print(f"   �ִ� TE: {technical_efficiency.max():.4f}")
            
            return technical_efficiency
            
        except Exception as e:
            print(f"? ȿ���� ��� �� ����: {str(e)}")
            # ���� �� �⺻�� �Ҵ�
            technical_efficiency = np.full(len(residuals), 0.8)
            self.normalized_data['technical_efficiency'] = technical_efficiency
            return technical_efficiency
    
    def calculate_cost_economics(self):
        """����Լ� �������� ��ǥ ���"""
        print("\n����Լ� �������� ��ǥ ��� ��...")
        
        if 'technical_efficiency' not in self.normalized_data.columns:
            self.calculate_efficiency()
        
        # �Ķ����
        beta = self.results['frontier']['beta']
        
        # ���� ���� ��� (�ùٸ� ���)
        self._calculate_actual_cost_shares()
        
        # ����ź�¼� ���
        self._calculate_price_elasticities(beta)
        
        # �����ȭ ��� (������ - ����Լ� ����)
        if self.include_time:
            self._calculate_technical_change_cost_corrected(beta)
        
        # �Ը��� ���� ��� (������)
        self._calculate_scale_economies(beta)
        
        # TFP ���� ���
        self._calculate_tfp_decomposition()
        
        print("����Լ� �������� ��ǥ ��� �Ϸ�")
        
        print("\n?? TFP ���� ����:")
        print("TFP ������ = �����ȭ + �����ȿ������ȭ + �Ը��ǰ���ȿ��")
        print("���⼭:")
        print("  ? �����ȭ: -��lnC/��t (��� ���� ȿ��)")  
        print("  ? �����ȿ������ȭ: ��ln(TE)/��t")
        print("  ? �Ը��ǰ���ȿ��: (1-1/�Ը��ǰ���) �� ���ⷮ������")
    
    def _calculate_actual_cost_shares(self):
        """���� ���� ��� (�ùٸ� ���)"""
        data = self.normalized_data
        
        print("\n���� ���� ��� ��...")
        
        # �Ѻ�� ��� �� ����
        total_cost_calculated = sum(data[price_var] * data[input_var] 
                                  for price_var, input_var in zip(self.price_vars, self.input_vars))
        data['calculated_total_cost'] = total_cost_calculated
        
        # ���� �Ѻ��� ���� �Ѻ�� ��
        actual_total_cost = data[self.cost_var]
        cost_diff = abs(actual_total_cost - total_cost_calculated).mean()
        print(f"   �Ѻ�� ����: ���� vs ��� ���� ��� = {cost_diff:.6f}")
        
        # ���� ���� ���: si = pi * xi / TC
        for price_var, input_var in zip(self.price_vars, self.input_vars):
            input_cost = data[price_var] * data[input_var]
            cost_share = input_cost / actual_total_cost  # ���� �Ѻ�� ���
            data[f'share_{input_var}'] = cost_share
        
        # ���� �հ� ����
        total_shares = sum(data[f'share_{input_var}'] for input_var in self.input_vars)
        data['total_shares'] = total_shares
        
        print(f"\n? ���� ����:")
        print(f"   ���� �հ� ���: {total_shares.mean():.6f}")
        print(f"   ���� �հ� ǥ������: {total_shares.std():.6f}")
        print(f"   ���� �հ� ����: [{total_shares.min():.6f}, {total_shares.max():.6f}]")
        
        # ���� ���� ���
        for input_var in self.input_vars:
            share_mean = data[f'share_{input_var}'].mean()
            print(f"   ��� {input_var.upper()} ����: {share_mean:.6f}")
        
        if abs(total_shares.mean() - 1.0) < 0.01:
            print("   ? ���� �հ谡 �ùٸ��ϴ� (? 1)")
        else:
            print(f"   ??  ���� �հ谡 1���� ������ϴ�: {total_shares.mean():.6f}")
    
    def _calculate_price_elasticities(self, beta):
        """����ź�¼� ���"""
        data = self.normalized_data
        
        # �ڱⰡ��ź�¼��� ��������ź�¼�
        for i, (price_var, input_var) in enumerate(zip(self.price_vars, self.input_vars)):
            share_i = data[f'share_{input_var}']
            
            for j, (price_var_j, input_var_j) in enumerate(zip(self.price_vars, self.input_vars)):
                if i == j:
                    # �ڱⰡ��ź�¼�
                    ln_var = f'ln_{price_var}'
                    beta_idx = 1 + i
                    elasticity = beta[beta_idx]
                    
                    # 2����
                    var2_idx = self.translog_vars.index(f'{ln_var}2')
                    elasticity += beta[1 + var2_idx] * data[ln_var]
                    
                    # ���� ź�¼�
                    elasticity = elasticity / share_i - 1
                    
                else:
                    # ��������ź�¼�
                    ln_var_i = f'ln_{price_var}'
                    ln_var_j = f'ln_{price_var_j}'
                    
                    cross_term = f'{ln_var_i}_{price_var_j.replace("p", "")}' if i < j else f'{ln_var_j}_{price_var.replace("p", "")}'
                    if cross_term in self.translog_vars:
                        cross_idx = self.translog_vars.index(cross_term)
                        elasticity = beta[1 + cross_idx] / share_i
                    else:
                        elasticity = 0
                
                data[f'elasticity_{input_var}_{input_var_j}'] = elasticity
    
    def _calculate_technical_change_cost_corrected(self, beta):
        """�����ȭ ��� (����Լ�) - ��� Ȯ�� �� ������ ���"""
        if not self.include_time:
            return
        
        data = self.normalized_data
        
        print(f"\n?? �����ȭ ���� ��� Ȯ��:")
        print("-" * 50)
        
        # �ð��� 1���� ��� Ȯ��
        time_idx = self.translog_vars.index(self.time_var)
        beta_t = beta[1 + time_idx]  # ����� ����
        print(f"   ��_t (�ð� 1���� ���): {beta_t:.6f}")
        
        # �ð��� 2���� ��� Ȯ��
        time2_idx = self.translog_vars.index(f'{self.time_var}2')
        beta_t2 = beta[1 + time2_idx]
        print(f"   ��_tt (�ð� 2���� ���): {beta_t2:.6f}")
        
        # �ð��� ���ݺ����� ������ ��� Ȯ��
        print("   �ð�-���� ������ ���:")
        time_price_coeffs = {}
        for var in self.price_vars:
            time_cross = f'ln_{var}_{self.time_var}'
            if time_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_cross)
                beta_cross = beta[1 + cross_idx]
                time_price_coeffs[var] = beta_cross
                print(f"     ��_{var}_t: {beta_cross:.6f}")
        
        # �ð��� ���ⷮ�� ������ ��� Ȯ��
        time_output_cross = f'ln_{self.output_var}_{self.time_var}'
        beta_yt = 0
        if time_output_cross in self.translog_vars:
            cross_idx = self.translog_vars.index(time_output_cross)
            beta_yt = beta[1 + cross_idx]
            print(f"   ��_y_t (�ð�-���ⷮ ������): {beta_yt:.6f}")
        
        print("-" * 50)
        
        # �����ȭ ���: TECH = -��lnC/��t (PDF �����)
        # ��lnC/��t = ��_t + ��_tt*t + �ҥ�_it*ln(pi) + ��_yt*ln(y)
        # ���� TECH = -(��_t + ��_tt*t + �ҥ�_it*ln(pi) + ��_yt*ln(y))
        
        # 1����: -��_t
        tech_change = -beta_t
        
        # 2����: -��_tt * t
        tech_change += -beta_t2 * data[self.time_var]
        
        # �ð�-���� ������: -�ҥ�_it * ln(pi)
        for var in self.price_vars:
            if var in time_price_coeffs:
                tech_change += -time_price_coeffs[var] * data[f'ln_{var}']
        
        # �ð�-���ⷮ ������: -��_yt * ln(y)
        if beta_yt != 0:
            tech_change += -beta_yt * data[f'ln_{self.output_var}']
        
        data['tech_change_cost'] = tech_change
        
        print(f"\n?? �����ȭ ��� ���:")
        print(f"   ����: TECH = -��lnC/��t")
        print(f"   ��� �����ȭ: {tech_change.mean():.6f}")
        print(f"   �����ȭ ����: [{tech_change.min():.6f}, {tech_change.max():.6f}]")
        
        # �ؼ� ����
        if tech_change.mean() > 0:
            print("   ? ��� �� ������� (��� ����)")
        elif tech_change.mean() < 0:
            print("   ??  ���� �� ����� �Ǵ� ��� ����")
            print("   ?? ������ ����:")
            print("      - �ð���� ��_t > 0 (�ð��� ���� �������)")
            print("      - �����Ϳ� ����� �ݿ�")
            print("      - ���� ���� �Ѱ�")
        else:
            print("   ??  �� �� �����ȭ ����")
        
        # ������Һ� �⿩�� �м�
        print(f"\n?? �����ȭ ������� �м�:")
        component1 = -beta_t
        component2 = (-beta_t2 * data[self.time_var]).mean()
        print(f"   1���� �⿩�� (-��_t): {component1:.6f}")
        print(f"   2���� �⿩�� ��� (-��_tt*t): {component2:.6f}")
        
        total_cross_effect = 0
        for var in self.price_vars:
            if var in time_price_coeffs:
                cross_effect = (-time_price_coeffs[var] * data[f'ln_{var}']).mean()
                total_cross_effect += cross_effect
                print(f"   {var} ������ ���: {cross_effect:.6f}")
        
        if beta_yt != 0:
            output_cross_effect = (-beta_yt * data[f'ln_{self.output_var}']).mean()
            total_cross_effect += output_cross_effect
            print(f"   ���ⷮ ������ ���: {output_cross_effect:.6f}")
        
        print(f"   �� ������ ȿ��: {total_cross_effect:.6f}")
        print(f"   ��ü ���: {component1 + component2 + total_cross_effect:.6f}")
        print("-" * 50)
    
    def _calculate_scale_economies(self, beta):
        """�Ը��� ���� ��� (����Լ�)"""
        data = self.normalized_data
        
        # ���ⷮ�� ���� ���ź�¼� (IRTS = ��lnC/��lnY)
        ln_output_var = f'ln_{self.output_var}'
        output_idx = self.translog_vars.index(ln_output_var)
        
        # 1����: ��_y
        output_elasticity = beta[1 + output_idx]  # ����� ����
        
        # 2����: ��_yy * ln(y)
        output2_idx = self.translog_vars.index(f'{ln_output_var}2')
        output_elasticity += beta[1 + output2_idx] * data[ln_output_var]
        
        # ����-���ⷮ ������: �ҥ�_iy * ln(wi)
        for var in self.price_vars:
            output_cross = f'ln_{var}_{self.output_var}'
            if output_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(output_cross)
                output_elasticity += beta[1 + cross_idx] * data[f'ln_{var}']
        
        # �ð�-���ⷮ ������: ��_yt * t
        if self.include_time:
            time_output_cross = f'ln_{self.output_var}_{self.time_var}'
            if time_output_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_output_cross)
                output_elasticity += beta[1 + cross_idx] * data[self.time_var]
        
        # �Ը��� ���� = 1 / IRTS (���⼭ IRTS = ������ ���ź�¼�)
        scale_economies = 1 / output_elasticity
        
        # �����Ϳ� ���� (TFP ���ؿ��� ���)
        data['irts'] = output_elasticity  # ������ ���ź�¼� (1/RTS)
        data['output_elasticity'] = output_elasticity  # ���� ȣȯ��
        data['scale_economies'] = scale_economies
    
    def _calculate_tfp_decomposition(self):
        """TFP �������� �������� ���� ���"""
        data = self.normalized_data.sort_values([self.id_var, self.time_var])
        
        # 1. ����� ȿ���� ��ȭ ��� (TE�� �α� ����)
        data['ln_te'] = np.log(data['technical_efficiency'])
        data['tech_efficiency_change'] = data.groupby(self.id_var)['ln_te'].diff()
        
        # 2. ���ⷮ ������ ��� (�Ը�ȿ�� ����)
        data['ln_output'] = data[f'ln_{self.output_var}']
        data['output_growth'] = data.groupby(self.id_var)['ln_output'].diff()
        
        # 3. �����ȭ ȿ�� (������ - �̹� �ùٸ� ��ȣ�� ����)
        if 'tech_change_cost' in data.columns:
            # �̹� Stata ������� ���Ǿ� �ùٸ� ��ȣ�� ����
            data['tech_change_effect'] = data['tech_change_cost']
        else:
            data['tech_change_effect'] = 0
        
        # 4. �Ը�ȿ�� ��� = (1-IRTS) �� ���������� (PDF �����)
        # Stata: gen SCALE = (1-IRTS)*gr_y
        if 'irts' in data.columns:
            data['scale_effect'] = (1 - data['irts']) * data['output_growth']
        else:
            data['scale_effect'] = 0
        
        # 5. TFP ������ ��� = SCALE + TECH + TEFF (PDF �����)
        # Stata: gen TFP = SCALE + TECH + TEFF
        data['tfp_growth'] = (data['scale_effect'] + 
                             data['tech_change_effect'] + 
                             data['tech_efficiency_change'])
        
        # 6. ����� ��ȯ�� ���� ������
        for var in ['tech_efficiency_change', 'tech_change_effect', 'scale_effect', 
                   'tfp_growth', 'output_growth']:
            if var in data.columns:
                data[f'{var}_pct'] = data[var] * 100
        
        self.normalized_data = data
        
        print(f"\n?? TFP ���� ��� �Ϸ�:")
        valid_data = data.dropna(subset=['tfp_growth', 'tech_change_effect', 'tech_efficiency_change', 'scale_effect'])
        if len(valid_data) > 0:
            print(f"   ��� TFP ������: {valid_data['tfp_growth'].mean()*100:.4f}%")
            print(f"   ��� �����ȭ: {valid_data['tech_change_effect'].mean()*100:.4f}%")
            print(f"   ��� �����ȿ������ȭ: {valid_data['tech_efficiency_change'].mean()*100:.4f}%")
            print(f"   ��� �Ը��ǰ���ȿ��: {valid_data['scale_effect'].mean()*100:.4f}%")
    
    def print_results(self, save_path='cost_results.csv'):
        """��� ��� - TFP ���� 4���� ��Ҹ� ���"""
        print("\n" + "=" * 100)
        print("Ȯ���������Լ� ���� ���")
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
            
            # ?? �����ȭ ���� ��� �ؼ� �߰�
            print(f"\n?? �����ȭ ���� ��� �ؼ�:")
            print("-" * 50)
            
            # �ð� ���� ��� ã�� �� �ؼ�
            if self.include_time and self.time_var in self.translog_vars:
                time_idx = self.translog_vars.index(self.time_var)
                beta_t = beta[time_idx + 1]  # ����� ����
                
                print(f"   �ð�(t) ��� ��_t = {beta_t:.6f}")
                if beta_t > 0:
                    print("   �� ��_t > 0: �ð��� ���� ������� (����� �Ǵ� ����� ����)")
                    print("   �� �����ȭ TECH = -��_t = {:.6f} (����)".format(-beta_t))
                elif beta_t < 0:
                    print("   �� ��_t < 0: �ð��� ���� ��밨�� (�������)")
                    print("   �� �����ȭ TECH = -��_t = {:.6f} (���)".format(-beta_t))
                else:
                    print("   �� ��_t = 0: �ð��� ���� ��뺯ȭ ����")
                
                print(f"   ?? �����ȭ ����: TECH = -��lnC/��t = -��_t - ��_tt��t - �ҥ�_it��ln(pi) - ��_yt��ln(y)")
            
            print("-" * 50)
            
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
        
        # 3. ����� ȿ����(TE) ���
        if 'technical_efficiency' in self.normalized_data.columns:
            te = self.normalized_data['technical_efficiency']
            print(f"\n3. ����� ȿ����(TE) ���")
            print("-" * 40)
            print(f"���: {te.mean():>20.4f}")
            print(f"ǥ������: {te.std():>15.4f}")
            print(f"�ּҰ�: {te.min():>16.4f}")
            print(f"�ִ밪: {te.max():>16.4f}")
            print(f"������: {te.median():>16.4f}")
        
        # 4. TFP ���� ��� (�ٽ� 4���� ��������)
        display_cols = [self.id_var, self.time_var]
        
        # TFP �ٽ� �������ε鸸 �߰�
        tfp_core_components = ['tfp_growth', 'tech_change_effect', 'tech_efficiency_change', 'scale_effect']
        
        for comp in tfp_core_components:
            if comp in self.normalized_data.columns:
                display_cols.append(comp)
        
        # ����ġ ����
        cost_display = self.normalized_data[display_cols].dropna()
        
        if len(cost_display) > 0:
            print(f"\n4. TFP ���� ��� (�ٽ� 4���� ��������)")
            print("=" * 80)
            
            # �÷��� ���� (������)
            col_rename = {
                'tfp_growth': 'TFP������',
                'tech_change_effect': '�����ȭ',
                'tech_efficiency_change': '�����ȿ������ȭ',
                'scale_effect': '�Ը��ǰ���ȿ��'
            }
            
            cost_display_renamed = cost_display.rename(columns=col_rename)
            
            # ��ü ������ ���
            print("TFP ���� ���:")
            print(cost_display_renamed.round(6).to_string(index=False))
            print(f"\n�� {len(cost_display_renamed)}�� ����ġ")
            
            # TFP ���� �������� ���
            tfp_stats_cols = ['TFP������', '�����ȭ', '�����ȿ������ȭ', '�Ը��ǰ���ȿ��']
            available_tfp_cols = [col for col in tfp_stats_cols if col in cost_display_renamed.columns]
            
            if available_tfp_cols:
                print("\n" + "-" * 70)
                print("TFP �������� ��� (���� ��ȭ��, %)")
                print("-" * 70)
                print(f"{'��������':<20} {'���':<12} {'ǥ������':<12} {'�ּҰ�':<12} {'�ִ밪':<12}")
                print("-" * 70)
                
                for col in available_tfp_cols:
                    if col in cost_display_renamed.columns:
                        col_data = cost_display_renamed[col].dropna() * 100  # �����
                        if len(col_data) > 0:
                            print(f"{col:<20} {col_data.mean():>8.4f} {col_data.std():>12.4f} {col_data.min():>12.4f} {col_data.max():>12.4f}")
                
                print("-" * 70)
                
                # TFP ���� ����
                if all(col in cost_display_renamed.columns for col in ['TFP������', '�����ȭ', '�����ȿ������ȭ', '�Ը��ǰ���ȿ��']):
                    calculated_tfp = (cost_display_renamed['�����ȭ'] + 
                                    cost_display_renamed['�����ȿ������ȭ'] + 
                                    cost_display_renamed['�Ը��ǰ���ȿ��'])
                    decomposition_error = cost_display_renamed['TFP������'] - calculated_tfp
                    
                    print(f"\n?? TFP ���� ��Ȯ��:")
                    print(f"   ��� ���ؿ���: {decomposition_error.mean()*100:>8.6f}%")
                    print(f"   �ִ� �������: {decomposition_error.abs().max()*100:>8.6f}%")
                    
                    if decomposition_error.abs().max() < 0.05:  # 5% ����
                        print("   ? ���ذ� ��Ȯ�մϴ�")
                    elif decomposition_error.abs().max() < 0.10:  # 10% ����
                        print("   ??  ���� ������ �ټ� �ֽ��ϴ�")
                    else:
                        print("   ? ���� ������ Ů�ϴ�")
            
            # �ð��� ���
            if self.include_time and len(cost_display) > 0:
                print(f"\n�ð��� ���:")
                time_avg = cost_display.groupby(self.time_var)[tfp_core_components].mean()
                time_avg_renamed = time_avg.rename(columns=col_rename)
                print(time_avg_renamed.round(6).to_string())
            
            # ID�� ��� (��ü�� ��)
            print(f"\nID�� ��� (��ü�� ��):")
            print("-" * 60)
            id_avg = cost_display.groupby(self.id_var)[tfp_core_components].mean()
            id_avg_renamed = id_avg.rename(columns=col_rename)
            print(id_avg_renamed.round(6).to_string())
            
            # ���� ����
            print(f"\n?? TFP ���� �м� ������ ����: {save_path}")
            
            # ���丮�� ������ ����
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            cost_display_renamed.to_csv(save_path, index=False, encoding='utf-8-sig')
            print("���� �Ϸ�!")
        
        else:
            print("TFP ���� �м� �����Ͱ� �����ϴ�")
        
        print("\n" + "=" * 100)
    
    def plot_results(self):
        """��� �ð�ȭ"""
        if 'technical_efficiency' not in self.normalized_data.columns:
            print("����� ȿ����(TE)�� ������ �ʾҽ��ϴ�.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ����� ȿ���� ����
        axes[0, 0].hist(self.normalized_data['technical_efficiency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('����� ȿ����(TE) ����')
        axes[0, 0].set_xlabel('����� ȿ����')
        axes[0, 0].set_ylabel('��')
        
        # TFP ������ �ð迭
        if self.include_time and 'tfp_growth' in self.normalized_data.columns:
            tfp_by_time = self.normalized_data.groupby(self.time_var)['tfp_growth'].mean()
            axes[0, 1].plot(tfp_by_time.index, tfp_by_time.values * 100, marker='o')
            axes[0, 1].set_title('�ð��� ��� TFP ������')
            axes[0, 1].set_xlabel('�ð�')
            axes[0, 1].set_ylabel('TFP ������ (%)')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # TFP �������κ� �ð迭
        if self.include_time and all(col in self.normalized_data.columns for col in ['tech_change_effect', 'tech_efficiency_change', 'scale_effect']):
            components = ['tech_change_effect', 'tech_efficiency_change', 'scale_effect']
            component_names = ['�����ȭ', '�����ȿ������ȭ', '�Ը��ǰ���ȿ��']
            
            for comp, name in zip(components, component_names):
                comp_by_time = self.normalized_data.groupby(self.time_var)[comp].mean()
                axes[1, 0].plot(comp_by_time.index, comp_by_time.values * 100, marker='o', label=name)
            
            axes[1, 0].set_title('TFP �������κ� �ð迭')
            axes[1, 0].set_xlabel('�ð�')
            axes[1, 0].set_ylabel('��ȭ�� (%)')
            axes[1, 0].legend()
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # ���� ����
        share_vars = [f'share_{input_var}' for input_var in self.input_vars if f'share_{input_var}' in self.normalized_data.columns]
        if share_vars:
            share_data = self.normalized_data[share_vars].mean()
            axes[1, 1].bar(range(len(share_data)), share_data.values)
            axes[1, 1].set_title('��� ����')
            axes[1, 1].set_xlabel('������')
            axes[1, 1].set_ylabel('����')
            axes[1, 1].set_xticks(range(len(share_data)))
            axes[1, 1].set_xticklabels([var.replace('share_', '').upper() for var in share_vars])
        
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self, save_path='cost_tfp_results.csv'):
        """��ü TFP ���� �м� ����"""
        print("Ȯ���������Լ� �� TFP ���� �м��� �����մϴ�...")
        
        # 1. EDA
        self.exploratory_data_analysis()
        
        # 2. ������ �غ�
        self.normalize_data()
        self.create_translog_variables()
        
        # 3. ����
        self.estimate_ols()
        self.estimate_stochastic_frontier()
        
        # 4. TFP �� �������� �м�
        self.calculate_efficiency()
        self.calculate_cost_economics()
        
        # 5. ��� ���
        self.print_results(save_path)
        self.plot_results()
        
        print("\nTFP ���� �м��� �Ϸ�Ǿ����ϴ�!")
        
        return self.results, self.normalized_data


def Run_StochasticFrontierCost(data, cost_var, output_var, price_vars, input_vars, 
                          time_var='year', id_var='id', include_time=True, save_path='cost_results.csv'):
    """
    Ȯ���������Լ��� �̿��� TFP ���� �м� - ����� ģȭ�� �������̽�
    """
    
    print("?? Ȯ���������Լ��� �̿��� TFP ���� �м��� �����մϴ�...")
    print("=" * 80)
    
    try:
        # �м��� ����
        analyzer = _StochasticFrontierCostAnalyzer(
            data=data,
            cost_var=cost_var,
            price_vars=price_vars,
            input_vars=input_vars,
            output_var=output_var,
            time_var=time_var,
            id_var=id_var,
            include_time=include_time
        )
        
        # ��ü �м� ����
        results, processed_data = analyzer.run_complete_analysis(save_path)
        
        print("? �м��� �Ϸ�Ǿ����ϴ�!")
        
        return results, processed_data
        
    except Exception as e:
        print(f"? �м� �� ���� �߻�: {str(e)}")
        print("�� ����:")
        import traceback
        traceback.print_exc()
        return None, None
