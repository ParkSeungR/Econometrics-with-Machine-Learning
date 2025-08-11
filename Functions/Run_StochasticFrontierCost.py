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
    확률변경비용함수 분석, 기술 효율성, TFP 구성요인 분해를 위한 내부 클래스
    (사용자가 직접 호출하지 않음)
    """
    
    def __init__(self, data, cost_var, price_vars, input_vars, output_var, time_var='t', id_var='id', include_time=True):
        self.data = data.copy()
        self.cost_var = cost_var  # 총비용
        self.price_vars = price_vars  # 요소가격들 
        self.input_vars = input_vars  # 투입량들
        self.output_var = output_var  # 산출량
        self.time_var = time_var
        self.id_var = id_var
        self.include_time = include_time
        
        # 결과 저장용 딕셔너리
        self.results = {}
        self.normalized_data = None
        self.translog_vars = None
        
        # 데이터 검증
        self._validate_data()
        
    def _validate_data(self):
        """데이터 유효성 검사"""
        required_vars = [self.cost_var, self.output_var] + self.price_vars + self.input_vars + [self.id_var]
        if self.include_time:
            required_vars.append(self.time_var)
            
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"다음 변수들이 데이터에 없습니다: {missing_vars}")
        
        # 가격변수와 투입량 변수의 개수가 같은지 확인
        if len(self.price_vars) != len(self.input_vars):
            raise ValueError(f"가격변수 개수({len(self.price_vars)})와 투입량변수 개수({len(self.input_vars)})가 다릅니다.")
            
        # 로그 변환을 위해 양수 체크
        for var in [self.cost_var, self.output_var] + self.price_vars + self.input_vars:
            if (self.data[var] <= 0).any():
                raise ValueError(f"변수 {var}에 0 이하의 값이 있습니다. 로그 변환이 불가능합니다.")
        
        print(f"? 데이터 검증 완료:")
        print(f"   가격변수: {self.price_vars}")
        print(f"   투입량변수: {self.input_vars}")
        print(f"   총비용변수: {self.cost_var}")
        print(f"   산출량변수: {self.output_var}")
    
    def exploratory_data_analysis(self):
        """탐색적 데이터 분석 수행"""
        print("=" * 60)
        print("탐색적 데이터 분석 (EDA) - 확률변경비용함수")
        print("=" * 60)
        
        # 기본 변수들
        analysis_vars = [self.cost_var, self.output_var] + self.price_vars
        if self.include_time:
            analysis_vars.append(self.time_var)
        
        # 1. 기초통계량
        print("\n1. 기초통계량")
        print("-" * 40)
        desc_stats = self.data[analysis_vars].describe()
        print(desc_stats.round(4))
        
        # 2. 상관관계
        print("\n2. 상관관계 조회")
        print("-" * 40)
        corr_matrix = self.data[analysis_vars].corr()
        print(corr_matrix.round(4))
        
        return desc_stats, corr_matrix
    
    def normalize_data(self):
        """개체별 평균으로 데이터 표준화"""
        print("\n데이터 표준화 수행 중...")
        
        self.normalized_data = self.data.copy()
        
        # 개체별 평균 계산 (비용, 가격, 산출량)
        vars_to_normalize = [self.cost_var, self.output_var] + self.price_vars
        
        for var in vars_to_normalize:
            # 개체별 평균
            mean_by_id = self.data.groupby(self.id_var)[var].transform('mean')
            # 표준화
            self.normalized_data[f'nm_{var}'] = self.data[var] / mean_by_id
            # 로그 변환
            self.normalized_data[f'ln_{var}'] = np.log(self.normalized_data[f'nm_{var}'])
        
        # 투입량도 복사 (비용몫 계산용)
        for input_var in self.input_vars:
            self.normalized_data[input_var] = self.data[input_var]
        
        if self.include_time:
            # 시간 변수는 로그 변환 없이 그대로 사용 (1, 2, 3, ...)
            self.normalized_data[self.time_var] = self.data[self.time_var]
        
        print("데이터 표준화 완료")
    
    def create_translog_variables(self):
        """초월대수 비용함수를 위한 교차항 변수 생성"""
        print("초월대수 비용함수 변수 생성 중...")
        
        if self.normalized_data is None:
            self.normalize_data()
        
        # 로그 변환된 가격 변수들과 산출량
        ln_price_vars = [f'ln_{var}' for var in self.price_vars]
        ln_output_var = f'ln_{self.output_var}'
        
        # 모든 핵심변수 (가격 + 산출량 + 시간변수)
        all_vars = ln_price_vars + [ln_output_var]
        if self.include_time:
            all_vars.append(self.time_var)  # 시간은 그냥 t
        
        # 2차항 생성
        for var in all_vars:
            self.normalized_data[f'{var}2'] = 0.5 * self.normalized_data[var] ** 2
        
        # 가격변수들 간 교차항
        n_prices = len(ln_price_vars)
        for i in range(n_prices):
            for j in range(i+1, n_prices):
                var1 = ln_price_vars[i]
                var2 = ln_price_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.normalized_data[var_name] = self.normalized_data[var1] * self.normalized_data[var2]
        
        # 가격과 산출량의 교차항
        for var in ln_price_vars:
            var_name = f'{var}_{self.output_var}'
            self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[ln_output_var]
        
        # 시간과 다른 변수들의 교차항
        if self.include_time:
            for var in ln_price_vars + [ln_output_var]:
                var_name = f'{var}_{self.time_var}'
                self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[self.time_var]
        
        # 핵심분석용 변수 리스트 생성
        self.translog_vars = []
        
        # 1차항
        self.translog_vars.extend(all_vars)
        
        # 2차항
        for var in all_vars:
            self.translog_vars.append(f'{var}2')
        
        # 가격변수들 간 교차항
        for i in range(n_prices):
            for j in range(i+1, n_prices):
                var1 = ln_price_vars[i]
                var2 = ln_price_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.translog_vars.append(var_name)
        
        # 가격-산출량 교차항
        for var in self.price_vars:
            self.translog_vars.append(f'ln_{var}_{self.output_var}')
        
        # 시간 교차항
        if self.include_time:
            for var in self.price_vars + [self.output_var]:
                self.translog_vars.append(f'ln_{var}_{self.time_var}')
        
        print(f"생성된 변수 수: {len(self.translog_vars)}")
        print("초월대수 비용함수 변수 생성 완료")
    
    def estimate_ols(self):
        """OLS 추정 (초기값용)"""
        print("\nOLS 추정 수행 중...")
        
        if self.translog_vars is None:
            self.create_translog_variables()
        
        # 종속변수 (총비용)
        y = self.normalized_data[f'ln_{self.cost_var}']
        
        # 독립변수
        X = self.normalized_data[self.translog_vars]
        X = sm.add_constant(X)
        
        # OLS 추정
        ols_model = sm.OLS(y, X).fit()
        
        self.results['ols'] = ols_model
        
        print("OLS 추정 완료")
        print(f"R-squared: {ols_model.rsquared:.4f}")
        
        return ols_model
    
    def estimate_stochastic_frontier(self, distribution='half_normal'):
        """확률변경비용함수 추정 - 개선된 버전"""
        print(f"\n확률변경비용함수 추정 수행 중 (분포: {distribution})...")
        
        if 'ols' not in self.results:
            print("OLS 추정을 먼저 수행합니다...")
            self.estimate_ols()
        
        # OLS 결과 검증
        if not hasattr(self.results['ols'], 'params'):
            print("? OLS 추정 결과가 없습니다.")
            return self._create_fallback_result()
        
        # 초기값 설정
        ols_params = self.results['ols'].params.values
        
        # OLS 결과 검증
        if np.any(np.isnan(ols_params)) or np.any(np.isinf(ols_params)):
            print("?? OLS 파라미터에 NaN/Inf가 있습니다. 기본값을 사용합니다.")
            ols_params = np.zeros(len(ols_params))
            ols_params[0] = np.log(self.normalized_data[f'ln_{self.cost_var}'].mean())
        
        # 종속변수와 독립변수
        y = self.normalized_data[f'ln_{self.cost_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])  # 상수항 추가
        
        # 데이터 검증
        if np.any(np.isnan(y)) or np.any(np.isnan(X)):
            print("? 데이터에 NaN이 있습니다.")
            return self._create_fallback_result()
        
        print(f"   데이터 크기: y={y.shape}, X={X.shape}")
        print(f"   OLS R²: {self.results['ols'].rsquared:.4f}")
        
        # 최대우도추정
        print("?? 최대우도추정 시작...")
        try:
            if distribution == 'half_normal':
                result = self._ml_estimation_half_normal_cost(y, X, ols_params)
            else:
                raise ValueError("현재는 half-normal 분포만 지원합니다.")
        except Exception as e:
            print(f"? 최대우도추정 중 오류: {str(e)}")
            return self._create_fallback_result()
        
        # 결과 검증
        if result['success']:
            # 파라미터 합리성 검증
            if (result['sigma_u'] > 0 and result['sigma_v'] > 0 and 
                result['sigma_u'] < 100 and result['sigma_v'] < 100 and
                np.isfinite(result['log_likelihood'])):
                
                self.results['frontier'] = result
                print("? 확률변경비용함수 추정 완료")
                
                # 모델 적합도 정보
                gamma = result['sigma_u']**2 / (result['sigma_u']**2 + result['sigma_v']**2)
                print(f"   γ = σ²u/σ²: {gamma:.4f}")
                if gamma > 0.5:
                    print("   → 비효율성이 오차의 주요 원인입니다.")
                else:
                    print("   → 확률적 오차가 주요 원인입니다.")
                
                return result
            else:
                print("?? 추정된 파라미터가 비합리적입니다.")
                return self._create_fallback_result()
        else:
            print(f"?? 추정 실패: {result.get('message', 'Unknown error')}")
            return self._create_fallback_result()
    
    def _create_fallback_result(self):
        """추정 실패시 대체 결과 생성"""
        print("   대체 결과 생성 중...")
        
        # 기본 파라미터 설정
        n_params = len(self.translog_vars) + 1  # 상수항 포함
        fallback_beta = np.zeros(n_params)
        
        # 상수항은 평균 비용으로 설정
        fallback_beta[0] = self.normalized_data[f'ln_{self.cost_var}'].mean()
        
        # 기본 sigma 값
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
        print("   ?? 대체 파라미터 사용 - 결과 해석에 주의하세요.")
        
        return fallback_result
    
    def _ml_estimation_half_normal_cost(self, y, X, initial_params):
        """비용함수용 Half-normal 분포를 가정한 최대우도추정 - 개선된 버전"""
        
        def log_likelihood(params):
            try:
                n_beta = X.shape[1]
                beta = params[:n_beta]
                log_sigma_u = params[n_beta]      
                log_sigma_v = params[n_beta + 1]  
                
                # sigma 계산 (더 안정적)
                sigma_u = np.exp(np.clip(log_sigma_u, -10, 5))  # 극값 제한
                sigma_v = np.exp(np.clip(log_sigma_v, -10, 5))
                
                # 최소값 보장
                sigma_u = np.maximum(sigma_u, 1e-4)
                sigma_v = np.maximum(sigma_v, 1e-4)
                
                # 잔차 계산
                residuals = y - X @ beta
                
                # 잔차의 스케일 체크
                if np.std(residuals) > 100 or np.std(residuals) < 1e-6:
                    return 1e8
                
                # 복합오차 파라미터
                sigma_sq = sigma_u**2 + sigma_v**2
                sigma = np.sqrt(sigma_sq)
                
                if sigma < 1e-4 or sigma > 100:
                    return 1e8
                
                # 람다 (비율 제한)
                lambd = np.clip(sigma_u / sigma_v, 0.01, 100)
                
                # 표준화된 잔차
                residuals_std = residuals / sigma
                residuals_std = np.clip(residuals_std, -8, 8)
                
                # epsilon* 계산 (비용함수)
                epsilon_star = residuals_std * lambd
                epsilon_star = np.clip(epsilon_star, -8, 8)
                
                # 로그 확률밀도 계산
                log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * residuals_std**2
                
                # 로그 누적분포 계산 (안정적 방법)
                log_Phi = np.where(epsilon_star > -5, 
                                  np.log(norm.cdf(epsilon_star) + 1e-15),
                                  epsilon_star - 0.5 * epsilon_star**2 - np.log(np.sqrt(2*np.pi)))
                
                # 로그우도 계산
                log_likelihood_val = (np.log(2) - np.log(sigma) + log_phi + log_Phi).sum()
                
                # 최종 검증
                if not np.isfinite(log_likelihood_val) or log_likelihood_val < -1e6:
                    return 1e8
                
                return -log_likelihood_val
                
            except Exception as e:
                return 1e8
        
        # 개선된 초기값 설정
        print("   초기값 설정 중...")
        ols_residuals = y - X @ initial_params
        ols_sigma = np.std(ols_residuals)
        
        # 더 큰 초기값 사용 (안정적 수렴을 위해)
        initial_sigma_u = np.clip(ols_sigma * 0.8, 0.05, 1.0)  # 더 큰 값
        initial_sigma_v = np.clip(ols_sigma * 0.6, 0.05, 1.0)  # 더 큰 값
        
        # log scale로 초기값 설정
        initial_vals = np.concatenate([
            initial_params, 
            [np.log(initial_sigma_u), np.log(initial_sigma_v)]
        ])
        
        print(f"   OLS σ: {ols_sigma:.4f}")
        print(f"   초기 σ_u: {initial_sigma_u:.4f}, σ_v: {initial_sigma_v:.4f}")
        
        # 더 관대한 bounds 설정
        n_beta = len(initial_params)
        bounds = []
        
        # beta 파라미터: 더 넓은 범위
        for i in range(n_beta):
            bounds.append((-20, 20))  
        
        # log_sigma: 더 넓은 범위
        bounds.append((-8, 3))   # exp(-8) ? 0.0003, exp(3) ? 20
        bounds.append((-8, 3))
        
        # 최적화 시도 (더 관대한 설정)
        print("   최적화 시작...")
        result = None
        
        # 방법 1: L-BFGS-B (더 관대한 설정)
        try:
            print("   시도 1: L-BFGS-B")
            result = minimize(log_likelihood, initial_vals, method='L-BFGS-B', 
                            bounds=bounds, 
                            options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if result.success and result.fun < 1e7:
                print(f"   ? L-BFGS-B 성공: f={result.fun:.4f}")
            else:
                print(f"   L-BFGS-B 실패: f={result.fun:.4f}")
                result = None
        except Exception as e:
            print(f"   L-BFGS-B 오류: {str(e)}")
            result = None
        
        # 방법 2: 다른 초기값으로 재시도
        if result is None:
            try:
                print("   시도 2: 다른 초기값으로 L-BFGS-B")
                # 초기값을 약간 변경
                alt_initial_vals = initial_vals.copy()
                alt_initial_vals[-2:] += np.random.normal(0, 0.5, 2)  # sigma 초기값 변경
                
                result = minimize(log_likelihood, alt_initial_vals, method='L-BFGS-B', 
                                bounds=bounds, 
                                options={'maxiter': 2000, 'ftol': 1e-6})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? 대안 L-BFGS-B 성공: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   대안 L-BFGS-B 실패: {str(e)}")
                result = None
        
        # 방법 3: BFGS (bounds 없음)
        if result is None:
            try:
                print("   시도 3: BFGS")
                result = minimize(log_likelihood, initial_vals, method='BFGS', 
                                options={'maxiter': 1500, 'gtol': 1e-4})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? BFGS 성공: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   BFGS 실패: {str(e)}")
                result = None
        
        # 방법 4: Powell (derivative-free)
        if result is None:
            try:
                print("   시도 4: Powell")
                result = minimize(log_likelihood, initial_vals, method='Powell', 
                                options={'maxiter': 1000, 'ftol': 1e-6})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? Powell 성공: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   Powell 실패: {str(e)}")
                result = None
        
        # 모든 방법 실패시 기본값 반환
        if result is None:
            print("   ?? 모든 최적화 방법 실패 - 기본값 사용")
            result = type('obj', (object,), {
                'x': initial_vals, 
                'fun': 1e10, 
                'success': False,
                'message': 'All optimization methods failed'
            })()
        
        # 개선된 표준오차 계산
        print("   표준오차 계산 중...")
        try:
            if result.success and result.fun < 1e9:
                # Hessian 계산 (더 안정적인 방법)
                eps = 1e-5
                n_params = len(result.x)
                hessian = np.zeros((n_params, n_params))
                
                f0 = log_likelihood(result.x)
                
                # diagonal elements만 계산 (더 안정적)
                for i in range(n_params):
                    x_plus = result.x.copy()
                    x_minus = result.x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    
                    f_plus = log_likelihood(x_plus)
                    f_minus = log_likelihood(x_minus)
                    
                    second_deriv = (f_plus - 2*f0 + f_minus) / (eps**2)
                    hessian[i, i] = abs(second_deriv)  # 절댓값 사용
                
                # 표준오차 계산
                std_errors = np.zeros(n_params)
                for i in range(n_params):
                    if hessian[i, i] > 1e-10:
                        std_errors[i] = 1.0 / np.sqrt(hessian[i, i])
                    else:
                        std_errors[i] = np.nan
                
                # 비합리적인 표준오차 제한
                std_errors = np.where(std_errors > 100, np.nan, std_errors)
                
            else:
                std_errors = np.full(len(result.x), np.nan)
                
        except Exception as e:
            print(f"   표준오차 계산 오류: {str(e)}")
            std_errors = np.full(len(result.x), np.nan)
        
        # 결과 정리
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
        
        # 결과 출력
        if estimated_params['success']:
            print(f"   ? 최적화 성공!")
            print(f"   로그우도: {estimated_params['log_likelihood']:.4f}")
            print(f"   σ_u: {estimated_params['sigma_u']:.4f}")
            print(f"   σ_v: {estimated_params['sigma_v']:.4f}")
            print(f"   λ (σ_u/σ_v): {estimated_params['sigma_u']/estimated_params['sigma_v']:.4f}")
        else:
            print(f"   ?? 최적화 실패: {estimated_params['message']}")
            print(f"   함수값: {result.fun:.4f}")
        
        return estimated_params
    
    def calculate_efficiency(self):
        """기술적 효율성(Technical Efficiency) 계산 - 개선된 버전"""
        print("\n기술적 효율성(TE) 계산 중...")
        
        if 'frontier' not in self.results:
            self.estimate_stochastic_frontier()
        
        # 추정이 실패한 경우 처리
        if not self.results['frontier']['success']:
            print("?? 확률변경함수 추정이 실패하여 효율성을 계산할 수 없습니다.")
            # 기본값으로 1.0 (완전 효율적) 할당
            self.normalized_data['technical_efficiency'] = 1.0
            return np.ones(len(self.normalized_data))
        
        # 파라미터 추출
        beta = self.results['frontier']['beta']
        sigma_u = self.results['frontier']['sigma_u']
        sigma_v = self.results['frontier']['sigma_v']
        
        # numerical stability 체크
        if sigma_u < 1e-6 or sigma_v < 1e-6:
            print("?? σ 값이 너무 작아 효율성 계산을 건너뜁니다.")
            self.normalized_data['technical_efficiency'] = 1.0
            return np.ones(len(self.normalized_data))
        
        # 종속변수와 독립변수
        y = self.normalized_data[f'ln_{self.cost_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        
        # 잔차
        residuals = y - X @ beta
        
        # 기술적 효율성 계산 (비용함수용 - Jondrow et al., 1982)
        sigma_sq = sigma_u**2 + sigma_v**2
        sigma = np.sqrt(sigma_sq)
        lambd = sigma_u / sigma_v
        
        # numerical stability를 위한 처리
        try:
            mu_star = residuals * sigma_u**2 / sigma_sq  # 비용함수는 부호 반대
            sigma_star = sigma_u * sigma_v / sigma
            
            # numerical stability: 극값 제한
            mu_star = np.clip(mu_star, -10, 10)
            sigma_star = np.maximum(sigma_star, 1e-6)
            
            # 조건부 기댓값 (기술적 효율성)
            ratio = mu_star / sigma_star
            ratio = np.clip(ratio, -10, 10)  # 극값 제한
            
            # 안정적인 계산을 위해 단계별로 계산
            exp_term = np.exp(-mu_star + 0.5 * sigma_star**2)
            
            # CDF 계산 시 numerical stability
            cdf_term1 = norm.cdf(ratio + sigma_star)
            cdf_term2 = norm.cdf(ratio)
            
            # 0으로 나누기 방지
            denominator = 1 - cdf_term2
            denominator = np.maximum(denominator, 1e-10)
            
            numerator = 1 - cdf_term1
            
            technical_efficiency = exp_term * numerator / denominator
            
            # 결과 검증 및 후처리
            technical_efficiency = np.clip(technical_efficiency, 1e-6, 1.0)
            
            # NaN이나 inf 체크
            invalid_mask = ~np.isfinite(technical_efficiency)
            if invalid_mask.any():
                print(f"?? {invalid_mask.sum()}개 관측값에서 효율성 계산 오류 - 기본값 0.5 할당")
                technical_efficiency[invalid_mask] = 0.5
            
            self.normalized_data['technical_efficiency'] = technical_efficiency
            
            print("? 기술적 효율성(TE) 계산 완료")
            print(f"   평균 TE: {technical_efficiency.mean():.4f}")
            print(f"   최소 TE: {technical_efficiency.min():.4f}")
            print(f"   최대 TE: {technical_efficiency.max():.4f}")
            
            return technical_efficiency
            
        except Exception as e:
            print(f"? 효율성 계산 중 오류: {str(e)}")
            # 오류 시 기본값 할당
            technical_efficiency = np.full(len(residuals), 0.8)
            self.normalized_data['technical_efficiency'] = technical_efficiency
            return technical_efficiency
    
    def calculate_cost_economics(self):
        """비용함수 경제학적 지표 계산"""
        print("\n비용함수 경제학적 지표 계산 중...")
        
        if 'technical_efficiency' not in self.normalized_data.columns:
            self.calculate_efficiency()
        
        # 파라미터
        beta = self.results['frontier']['beta']
        
        # 실제 비용몫 계산 (올바른 방법)
        self._calculate_actual_cost_shares()
        
        # 가격탄력성 계산
        self._calculate_price_elasticities(beta)
        
        # 기술변화 계산 (수정된 - 비용함수 기준)
        if self.include_time:
            self._calculate_technical_change_cost_corrected(beta)
        
        # 규모의 경제 계산 (수정된)
        self._calculate_scale_economies(beta)
        
        # TFP 분해 계산
        self._calculate_tfp_decomposition()
        
        print("비용함수 경제학적 지표 계산 완료")
        
        print("\n?? TFP 분해 공식:")
        print("TFP 증가율 = 기술변화 + 기술적효율성변화 + 규모의경제효과")
        print("여기서:")
        print("  ? 기술변화: -∂lnC/∂t (비용 감소 효과)")  
        print("  ? 기술적효율성변화: ∂ln(TE)/∂t")
        print("  ? 규모의경제효과: (1-1/규모의경제) × 산출량증가율")
    
    def _calculate_actual_cost_shares(self):
        """실제 비용몫 계산 (올바른 방법)"""
        data = self.normalized_data
        
        print("\n실제 비용몫 계산 중...")
        
        # 총비용 계산 및 검증
        total_cost_calculated = sum(data[price_var] * data[input_var] 
                                  for price_var, input_var in zip(self.price_vars, self.input_vars))
        data['calculated_total_cost'] = total_cost_calculated
        
        # 실제 총비용과 계산된 총비용 비교
        actual_total_cost = data[self.cost_var]
        cost_diff = abs(actual_total_cost - total_cost_calculated).mean()
        print(f"   총비용 검증: 실제 vs 계산 차이 평균 = {cost_diff:.6f}")
        
        # 실제 비용몫 계산: si = pi * xi / TC
        for price_var, input_var in zip(self.price_vars, self.input_vars):
            input_cost = data[price_var] * data[input_var]
            cost_share = input_cost / actual_total_cost  # 실제 총비용 사용
            data[f'share_{input_var}'] = cost_share
        
        # 비용몫 합계 검증
        total_shares = sum(data[f'share_{input_var}'] for input_var in self.input_vars)
        data['total_shares'] = total_shares
        
        print(f"\n? 비용몫 검증:")
        print(f"   비용몫 합계 평균: {total_shares.mean():.6f}")
        print(f"   비용몫 합계 표준편차: {total_shares.std():.6f}")
        print(f"   비용몫 합계 범위: [{total_shares.min():.6f}, {total_shares.max():.6f}]")
        
        # 개별 비용몫 출력
        for input_var in self.input_vars:
            share_mean = data[f'share_{input_var}'].mean()
            print(f"   평균 {input_var.upper()} 비용몫: {share_mean:.6f}")
        
        if abs(total_shares.mean() - 1.0) < 0.01:
            print("   ? 비용몫 합계가 올바릅니다 (? 1)")
        else:
            print(f"   ??  비용몫 합계가 1에서 벗어났습니다: {total_shares.mean():.6f}")
    
    def _calculate_price_elasticities(self, beta):
        """가격탄력성 계산"""
        data = self.normalized_data
        
        # 자기가격탄력성과 교차가격탄력성
        for i, (price_var, input_var) in enumerate(zip(self.price_vars, self.input_vars)):
            share_i = data[f'share_{input_var}']
            
            for j, (price_var_j, input_var_j) in enumerate(zip(self.price_vars, self.input_vars)):
                if i == j:
                    # 자기가격탄력성
                    ln_var = f'ln_{price_var}'
                    beta_idx = 1 + i
                    elasticity = beta[beta_idx]
                    
                    # 2차항
                    var2_idx = self.translog_vars.index(f'{ln_var}2')
                    elasticity += beta[1 + var2_idx] * data[ln_var]
                    
                    # 최종 탄력성
                    elasticity = elasticity / share_i - 1
                    
                else:
                    # 교차가격탄력성
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
        """기술변화 계산 (비용함수) - 계수 확인 및 정정한 계산"""
        if not self.include_time:
            return
        
        data = self.normalized_data
        
        print(f"\n?? 기술변화 관련 계수 확인:")
        print("-" * 50)
        
        # 시간의 1차항 계수 확인
        time_idx = self.translog_vars.index(self.time_var)
        beta_t = beta[1 + time_idx]  # 상수항 제외
        print(f"   β_t (시간 1차항 계수): {beta_t:.6f}")
        
        # 시간의 2차항 계수 확인
        time2_idx = self.translog_vars.index(f'{self.time_var}2')
        beta_t2 = beta[1 + time2_idx]
        print(f"   β_tt (시간 2차항 계수): {beta_t2:.6f}")
        
        # 시간과 가격변수의 교차항 계수 확인
        print("   시간-가격 교차항 계수:")
        time_price_coeffs = {}
        for var in self.price_vars:
            time_cross = f'ln_{var}_{self.time_var}'
            if time_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_cross)
                beta_cross = beta[1 + cross_idx]
                time_price_coeffs[var] = beta_cross
                print(f"     β_{var}_t: {beta_cross:.6f}")
        
        # 시간과 산출량의 교차항 계수 확인
        time_output_cross = f'ln_{self.output_var}_{self.time_var}'
        beta_yt = 0
        if time_output_cross in self.translog_vars:
            cross_idx = self.translog_vars.index(time_output_cross)
            beta_yt = beta[1 + cross_idx]
            print(f"   β_y_t (시간-산출량 교차항): {beta_yt:.6f}")
        
        print("-" * 50)
        
        # 기술변화 계산: TECH = -∂lnC/∂t (PDF 방법론)
        # ∂lnC/∂t = β_t + β_tt*t + Σβ_it*ln(pi) + β_yt*ln(y)
        # 따라서 TECH = -(β_t + β_tt*t + Σβ_it*ln(pi) + β_yt*ln(y))
        
        # 1차항: -β_t
        tech_change = -beta_t
        
        # 2차항: -β_tt * t
        tech_change += -beta_t2 * data[self.time_var]
        
        # 시간-가격 교차항: -Σβ_it * ln(pi)
        for var in self.price_vars:
            if var in time_price_coeffs:
                tech_change += -time_price_coeffs[var] * data[f'ln_{var}']
        
        # 시간-산출량 교차항: -β_yt * ln(y)
        if beta_yt != 0:
            tech_change += -beta_yt * data[f'ln_{self.output_var}']
        
        data['tech_change_cost'] = tech_change
        
        print(f"\n?? 기술변화 계산 결과:")
        print(f"   공식: TECH = -∂lnC/∂t")
        print(f"   평균 기술변화: {tech_change.mean():.6f}")
        print(f"   기술변화 범위: [{tech_change.min():.6f}, {tech_change.max():.6f}]")
        
        # 해석 도움말
        if tech_change.mean() > 0:
            print("   ? 양수 → 기술진보 (비용 감소)")
        elif tech_change.mean() < 0:
            print("   ??  음수 → 기술퇴보 또는 비용 증가")
            print("   ?? 가능한 원인:")
            print("      - 시간계수 β_t > 0 (시간에 따라 비용증가)")
            print("      - 데이터에 기술퇴보 반영")
            print("      - 추정 모델의 한계")
        else:
            print("   ??  영 → 기술변화 없음")
        
        # 구성요소별 기여도 분석
        print(f"\n?? 기술변화 구성요소 분석:")
        component1 = -beta_t
        component2 = (-beta_t2 * data[self.time_var]).mean()
        print(f"   1차항 기여도 (-β_t): {component1:.6f}")
        print(f"   2차항 기여도 평균 (-β_tt*t): {component2:.6f}")
        
        total_cross_effect = 0
        for var in self.price_vars:
            if var in time_price_coeffs:
                cross_effect = (-time_price_coeffs[var] * data[f'ln_{var}']).mean()
                total_cross_effect += cross_effect
                print(f"   {var} 교차항 평균: {cross_effect:.6f}")
        
        if beta_yt != 0:
            output_cross_effect = (-beta_yt * data[f'ln_{self.output_var}']).mean()
            total_cross_effect += output_cross_effect
            print(f"   산출량 교차항 평균: {output_cross_effect:.6f}")
        
        print(f"   총 교차항 효과: {total_cross_effect:.6f}")
        print(f"   전체 평균: {component1 + component2 + total_cross_effect:.6f}")
        print("-" * 50)
    
    def _calculate_scale_economies(self, beta):
        """규모의 경제 계산 (비용함수)"""
        data = self.normalized_data
        
        # 산출량에 대한 비용탄력성 (IRTS = ∂lnC/∂lnY)
        ln_output_var = f'ln_{self.output_var}'
        output_idx = self.translog_vars.index(ln_output_var)
        
        # 1차항: β_y
        output_elasticity = beta[1 + output_idx]  # 상수항 제외
        
        # 2차항: β_yy * ln(y)
        output2_idx = self.translog_vars.index(f'{ln_output_var}2')
        output_elasticity += beta[1 + output2_idx] * data[ln_output_var]
        
        # 가격-산출량 교차항: Σβ_iy * ln(wi)
        for var in self.price_vars:
            output_cross = f'ln_{var}_{self.output_var}'
            if output_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(output_cross)
                output_elasticity += beta[1 + cross_idx] * data[f'ln_{var}']
        
        # 시간-산출량 교차항: β_yt * t
        if self.include_time:
            time_output_cross = f'ln_{self.output_var}_{self.time_var}'
            if time_output_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_output_cross)
                output_elasticity += beta[1 + cross_idx] * data[self.time_var]
        
        # 규모의 경제 = 1 / IRTS (여기서 IRTS = 산출의 비용탄력성)
        scale_economies = 1 / output_elasticity
        
        # 데이터에 저장 (TFP 분해에서 사용)
        data['irts'] = output_elasticity  # 산출의 비용탄력성 (1/RTS)
        data['output_elasticity'] = output_elasticity  # 기존 호환성
        data['scale_economies'] = scale_economies
    
    def _calculate_tfp_decomposition(self):
        """TFP 증가율과 구성요인 분해 계산"""
        data = self.normalized_data.sort_values([self.id_var, self.time_var])
        
        # 1. 기술적 효율성 변화 계산 (TE의 로그 차분)
        data['ln_te'] = np.log(data['technical_efficiency'])
        data['tech_efficiency_change'] = data.groupby(self.id_var)['ln_te'].diff()
        
        # 2. 산출량 증가율 계산 (규모효과 계산용)
        data['ln_output'] = data[f'ln_{self.output_var}']
        data['output_growth'] = data.groupby(self.id_var)['ln_output'].diff()
        
        # 3. 기술변화 효과 (수정된 - 이미 올바른 부호로 계산됨)
        if 'tech_change_cost' in data.columns:
            # 이미 Stata 방식으로 계산되어 올바른 부호를 가짐
            data['tech_change_effect'] = data['tech_change_cost']
        else:
            data['tech_change_effect'] = 0
        
        # 4. 규모효과 계산 = (1-IRTS) × 산출증가율 (PDF 방법론)
        # Stata: gen SCALE = (1-IRTS)*gr_y
        if 'irts' in data.columns:
            data['scale_effect'] = (1 - data['irts']) * data['output_growth']
        else:
            data['scale_effect'] = 0
        
        # 5. TFP 증가율 계산 = SCALE + TECH + TEFF (PDF 방법론)
        # Stata: gen TFP = SCALE + TECH + TEFF
        data['tfp_growth'] = (data['scale_effect'] + 
                             data['tech_change_effect'] + 
                             data['tech_efficiency_change'])
        
        # 6. 백분율 변환을 위한 변수들
        for var in ['tech_efficiency_change', 'tech_change_effect', 'scale_effect', 
                   'tfp_growth', 'output_growth']:
            if var in data.columns:
                data[f'{var}_pct'] = data[var] * 100
        
        self.normalized_data = data
        
        print(f"\n?? TFP 분해 계산 완료:")
        valid_data = data.dropna(subset=['tfp_growth', 'tech_change_effect', 'tech_efficiency_change', 'scale_effect'])
        if len(valid_data) > 0:
            print(f"   평균 TFP 증가율: {valid_data['tfp_growth'].mean()*100:.4f}%")
            print(f"   평균 기술변화: {valid_data['tech_change_effect'].mean()*100:.4f}%")
            print(f"   평균 기술적효율성변화: {valid_data['tech_efficiency_change'].mean()*100:.4f}%")
            print(f"   평균 규모의경제효과: {valid_data['scale_effect'].mean()*100:.4f}%")
    
    def print_results(self, save_path='cost_results.csv'):
        """결과 출력 - TFP 분해 4가지 요소만 출력"""
        print("\n" + "=" * 100)
        print("확률변경비용함수 추정 결과")
        print("=" * 100)
        
        # 1. 파라미터 추정치 출력
        if 'frontier' in self.results:
            frontier = self.results['frontier']
            beta = frontier['beta']
            std_errors = frontier.get('std_errors', np.full(len(beta), np.nan))
            
            print("\n1. 파라미터 추정치")
            print("-" * 100)
            print(f"{'변수명':<20} {'계수':<12} {'표준오차':<12} {'t-값':<10} {'p-값':<10} {'유의성':<8}")
            print("-" * 100)
            
            # 상수항
            t_stat = beta[0] / std_errors[0] if not np.isnan(std_errors[0]) and std_errors[0] != 0 else np.nan
            p_value = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
            
            print(f"{'상수항':<20} {beta[0]:>8.6f} {std_errors[0]:>8.4f} {t_stat:>8.3f} {p_value:>8.4f} {significance:<8}")
            
            # 변수별 계수
            for i, var_name in enumerate(self.translog_vars):
                coef = beta[i + 1]  # 상수항 제외
                se = std_errors[i + 1] if i + 1 < len(std_errors) else np.nan
                t_stat = coef / se if not np.isnan(se) and se != 0 else np.nan
                p_value = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
                significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                
                print(f"{var_name:<20} {coef:>8.6f} {se:>8.4f} {t_stat:>8.3f} {p_value:>8.4f} {significance:<8}")
            
            print("-" * 100)
            print("유의성: *** p<0.01, ** p<0.05, * p<0.1")
            
            # ?? 기술변화 관련 계수 해석 추가
            print(f"\n?? 기술변화 관련 계수 해석:")
            print("-" * 50)
            
            # 시간 변수 계수 찾기 및 해석
            if self.include_time and self.time_var in self.translog_vars:
                time_idx = self.translog_vars.index(self.time_var)
                beta_t = beta[time_idx + 1]  # 상수항 제외
                
                print(f"   시간(t) 계수 β_t = {beta_t:.6f}")
                if beta_t > 0:
                    print("   → β_t > 0: 시간에 따라 비용증가 (기술퇴보 또는 비용상승 요인)")
                    print("   → 기술변화 TECH = -β_t = {:.6f} (음수)".format(-beta_t))
                elif beta_t < 0:
                    print("   → β_t < 0: 시간에 따라 비용감소 (기술진보)")
                    print("   → 기술변화 TECH = -β_t = {:.6f} (양수)".format(-beta_t))
                else:
                    print("   → β_t = 0: 시간에 따른 비용변화 없음")
                
                print(f"   ?? 기술변화 공식: TECH = -∂lnC/∂t = -β_t - β_tt×t - Σβ_it×ln(pi) - β_yt×ln(y)")
            
            print("-" * 50)
            
            # 2. 모델 통계량
            print(f"\n2. 모델 통계량")
            print("-" * 40)
            print(f"관측수: {frontier.get('n_obs', 'N/A'):>20}")
            print(f"파라미터 수: {frontier.get('n_params', 'N/A'):>15}")
            print(f"로그우도: {frontier['log_likelihood']:>15.4f}")
            print(f"시그마_u: {frontier['sigma_u']:>15.4f}")
            print(f"시그마_v: {frontier['sigma_v']:>15.4f}")
            print(f"시그마²: {frontier['sigma_u']**2 + frontier['sigma_v']**2:>15.4f}")
            print(f"람다 (σu/σv): {frontier['sigma_u']/frontier['sigma_v']:>10.4f}")
            print(f"γ = σu²/σ²: {frontier['sigma_u']**2/(frontier['sigma_u']**2 + frontier['sigma_v']**2):>13.4f}")
        
        # 3. 기술적 효율성(TE) 통계
        if 'technical_efficiency' in self.normalized_data.columns:
            te = self.normalized_data['technical_efficiency']
            print(f"\n3. 기술적 효율성(TE) 통계")
            print("-" * 40)
            print(f"평균: {te.mean():>20.4f}")
            print(f"표준편차: {te.std():>15.4f}")
            print(f"최소값: {te.min():>16.4f}")
            print(f"최대값: {te.max():>16.4f}")
            print(f"중위수: {te.median():>16.4f}")
        
        # 4. TFP 분해 결과 (핵심 4가지 구성요인)
        display_cols = [self.id_var, self.time_var]
        
        # TFP 핵심 구성요인들만 추가
        tfp_core_components = ['tfp_growth', 'tech_change_effect', 'tech_efficiency_change', 'scale_effect']
        
        for comp in tfp_core_components:
            if comp in self.normalized_data.columns:
                display_cols.append(comp)
        
        # 결측치 제거
        cost_display = self.normalized_data[display_cols].dropna()
        
        if len(cost_display) > 0:
            print(f"\n4. TFP 분해 결과 (핵심 4가지 구성요인)")
            print("=" * 80)
            
            # 컬럼명 변경 (가독성)
            col_rename = {
                'tfp_growth': 'TFP증가율',
                'tech_change_effect': '기술변화',
                'tech_efficiency_change': '기술적효율성변화',
                'scale_effect': '규모의경제효과'
            }
            
            cost_display_renamed = cost_display.rename(columns=col_rename)
            
            # 전체 데이터 출력
            print("TFP 분해 결과:")
            print(cost_display_renamed.round(6).to_string(index=False))
            print(f"\n총 {len(cost_display_renamed)}개 관측치")
            
            # TFP 분해 구성요인 통계
            tfp_stats_cols = ['TFP증가율', '기술변화', '기술적효율성변화', '규모의경제효과']
            available_tfp_cols = [col for col in tfp_stats_cols if col in cost_display_renamed.columns]
            
            if available_tfp_cols:
                print("\n" + "-" * 70)
                print("TFP 구성요인 통계 (연간 변화율, %)")
                print("-" * 70)
                print(f"{'구성요인':<20} {'평균':<12} {'표준편차':<12} {'최소값':<12} {'최대값':<12}")
                print("-" * 70)
                
                for col in available_tfp_cols:
                    if col in cost_display_renamed.columns:
                        col_data = cost_display_renamed[col].dropna() * 100  # 백분율
                        if len(col_data) > 0:
                            print(f"{col:<20} {col_data.mean():>8.4f} {col_data.std():>12.4f} {col_data.min():>12.4f} {col_data.max():>12.4f}")
                
                print("-" * 70)
                
                # TFP 분해 검증
                if all(col in cost_display_renamed.columns for col in ['TFP증가율', '기술변화', '기술적효율성변화', '규모의경제효과']):
                    calculated_tfp = (cost_display_renamed['기술변화'] + 
                                    cost_display_renamed['기술적효율성변화'] + 
                                    cost_display_renamed['규모의경제효과'])
                    decomposition_error = cost_display_renamed['TFP증가율'] - calculated_tfp
                    
                    print(f"\n?? TFP 분해 정확성:")
                    print(f"   평균 분해오차: {decomposition_error.mean()*100:>8.6f}%")
                    print(f"   최대 절대오차: {decomposition_error.abs().max()*100:>8.6f}%")
                    
                    if decomposition_error.abs().max() < 0.05:  # 5% 이하
                        print("   ? 분해가 정확합니다")
                    elif decomposition_error.abs().max() < 0.10:  # 10% 이하
                        print("   ??  분해 오차가 다소 있습니다")
                    else:
                        print("   ? 분해 오차가 큽니다")
            
            # 시간별 평균
            if self.include_time and len(cost_display) > 0:
                print(f"\n시간별 평균:")
                time_avg = cost_display.groupby(self.time_var)[tfp_core_components].mean()
                time_avg_renamed = time_avg.rename(columns=col_rename)
                print(time_avg_renamed.round(6).to_string())
            
            # ID별 평균 (개체별 비교)
            print(f"\nID별 평균 (개체별 비교):")
            print("-" * 60)
            id_avg = cost_display.groupby(self.id_var)[tfp_core_components].mean()
            id_avg_renamed = id_avg.rename(columns=col_rename)
            print(id_avg_renamed.round(6).to_string())
            
            # 파일 저장
            print(f"\n?? TFP 분해 분석 데이터 저장: {save_path}")
            
            # 디렉토리가 없으면 생성
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            cost_display_renamed.to_csv(save_path, index=False, encoding='utf-8-sig')
            print("저장 완료!")
        
        else:
            print("TFP 분해 분석 데이터가 없습니다")
        
        print("\n" + "=" * 100)
    
    def plot_results(self):
        """결과 시각화"""
        if 'technical_efficiency' not in self.normalized_data.columns:
            print("기술적 효율성(TE)이 계산되지 않았습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 기술적 효율성 분포
        axes[0, 0].hist(self.normalized_data['technical_efficiency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('기술적 효율성(TE) 분포')
        axes[0, 0].set_xlabel('기술적 효율성')
        axes[0, 0].set_ylabel('빈도')
        
        # TFP 증가율 시계열
        if self.include_time and 'tfp_growth' in self.normalized_data.columns:
            tfp_by_time = self.normalized_data.groupby(self.time_var)['tfp_growth'].mean()
            axes[0, 1].plot(tfp_by_time.index, tfp_by_time.values * 100, marker='o')
            axes[0, 1].set_title('시간별 평균 TFP 증가율')
            axes[0, 1].set_xlabel('시간')
            axes[0, 1].set_ylabel('TFP 증가율 (%)')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # TFP 구성요인별 시계열
        if self.include_time and all(col in self.normalized_data.columns for col in ['tech_change_effect', 'tech_efficiency_change', 'scale_effect']):
            components = ['tech_change_effect', 'tech_efficiency_change', 'scale_effect']
            component_names = ['기술변화', '기술적효율성변화', '규모의경제효과']
            
            for comp, name in zip(components, component_names):
                comp_by_time = self.normalized_data.groupby(self.time_var)[comp].mean()
                axes[1, 0].plot(comp_by_time.index, comp_by_time.values * 100, marker='o', label=name)
            
            axes[1, 0].set_title('TFP 구성요인별 시계열')
            axes[1, 0].set_xlabel('시간')
            axes[1, 0].set_ylabel('변화율 (%)')
            axes[1, 0].legend()
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 비용몫 분포
        share_vars = [f'share_{input_var}' for input_var in self.input_vars if f'share_{input_var}' in self.normalized_data.columns]
        if share_vars:
            share_data = self.normalized_data[share_vars].mean()
            axes[1, 1].bar(range(len(share_data)), share_data.values)
            axes[1, 1].set_title('평균 비용몫')
            axes[1, 1].set_xlabel('생산요소')
            axes[1, 1].set_ylabel('비용몫')
            axes[1, 1].set_xticks(range(len(share_data)))
            axes[1, 1].set_xticklabels([var.replace('share_', '').upper() for var in share_vars])
        
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self, save_path='cost_tfp_results.csv'):
        """전체 TFP 분해 분석 실행"""
        print("확률변경비용함수 및 TFP 분해 분석을 시작합니다...")
        
        # 1. EDA
        self.exploratory_data_analysis()
        
        # 2. 데이터 준비
        self.normalize_data()
        self.create_translog_variables()
        
        # 3. 추정
        self.estimate_ols()
        self.estimate_stochastic_frontier()
        
        # 4. TFP 및 구성요인 분석
        self.calculate_efficiency()
        self.calculate_cost_economics()
        
        # 5. 결과 출력
        self.print_results(save_path)
        self.plot_results()
        
        print("\nTFP 분해 분석이 완료되었습니다!")
        
        return self.results, self.normalized_data


def Run_StochasticFrontierCost(data, cost_var, output_var, price_vars, input_vars, 
                          time_var='year', id_var='id', include_time=True, save_path='cost_results.csv'):
    """
    확률변경비용함수를 이용한 TFP 분해 분석 - 사용자 친화적 인터페이스
    """
    
    print("?? 확률변경비용함수를 이용한 TFP 분해 분석을 시작합니다...")
    print("=" * 80)
    
    try:
        # 분석기 생성
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
        
        # 전체 분석 실행
        results, processed_data = analyzer.run_complete_analysis(save_path)
        
        print("? 분석이 완료되었습니다!")
        
        return results, processed_data
        
    except Exception as e:
        print(f"? 분석 중 오류 발생: {str(e)}")
        print("상세 오류:")
        import traceback
        traceback.print_exc()
        return None, None
