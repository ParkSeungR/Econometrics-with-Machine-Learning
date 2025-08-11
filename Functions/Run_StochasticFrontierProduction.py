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
    확률변경생산함수 추정과 생산성 측정을 위한 내부 클래스 (사용자가 직접 호출하지 않음)
    """
    
    def __init__(self, data, output_var, input_vars, time_var='t', id_var='id', include_time=True):
        self.data = data.copy()
        self.output_var = output_var
        self.input_vars = input_vars
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
        required_vars = [self.output_var] + self.input_vars + [self.id_var]
        if self.include_time:
            required_vars.append(self.time_var)
            
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"다음 변수들이 데이터에 없습니다: {missing_vars}")
            
        # 로그 변환을 위해 양수 체크
        for var in [self.output_var] + self.input_vars:
            if (self.data[var] <= 0).any():
                raise ValueError(f"변수 {var}에 0 이하의 값이 있습니다. 로그 변환이 불가능합니다.")
    
    def exploratory_data_analysis(self):
        """탐색적 데이터 분석 수행"""
        print("=" * 60)
        print("탐색적 데이터 분석 (EDA)")
        print("=" * 60)
        
        # 기본 변수들
        analysis_vars = [self.output_var] + self.input_vars
        if self.include_time:
            analysis_vars.append(self.time_var)
        
        # 1. 기술통계량
        print("\n1. 기술통계량")
        print("-" * 40)
        desc_stats = self.data[analysis_vars].describe()
        print(desc_stats.round(4))
        
        # 2. 상관관계
        print("\n2. 상관관계 행렬")
        print("-" * 40)
        corr_matrix = self.data[analysis_vars].corr()
        print(corr_matrix.round(4))
        
        return desc_stats, corr_matrix
    
    def normalize_data(self):
        """개체별 평균으로 데이터 표준화"""
        print("\n데이터 표준화 수행 중...")
        
        self.normalized_data = self.data.copy()
        
        # 개체별 평균 계산 (투입/산출 변수만)
        vars_to_normalize = [self.output_var] + self.input_vars
        
        for var in vars_to_normalize:
            # 개체별 평균
            mean_by_id = self.data.groupby(self.id_var)[var].transform('mean')
            # 표준화
            self.normalized_data[f'nm_{var}'] = self.data[var] / mean_by_id
            # 로그 변환
            self.normalized_data[f'ln_{var}'] = np.log(self.normalized_data[f'nm_{var}'])
        
        if self.include_time:
            # 시간 변수는 로그 변환 없이 그대로 사용 (1, 2, 3, ...)
            self.normalized_data[self.time_var] = self.data[self.time_var]
        
        print("데이터 표준화 완료")
    
    def create_translog_variables(self):
        """초월대수 함수를 위한 교차항 변수 생성"""
        print("초월대수 함수 변수 생성 중...")
        
        if self.normalized_data is None:
            self.normalize_data()
        
        # 로그 변환된 투입 변수들 (시간 제외)
        ln_input_vars = [f'ln_{var}' for var in self.input_vars]
        
        # 모든 회귀변수 (투입변수 + 시간변수)
        all_vars = ln_input_vars.copy()
        if self.include_time:
            all_vars.append(self.time_var)  # 시간은 그냥 t
        
        # 2차항 생성
        for var in all_vars:
            self.normalized_data[f'{var}2'] = 0.5 * self.normalized_data[var] ** 2
        
        # 교차항 생성 (투입변수들 간)
        n_inputs = len(ln_input_vars)
        for i in range(n_inputs):
            for j in range(i+1, n_inputs):
                var1 = ln_input_vars[i]
                var2 = ln_input_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.normalized_data[var_name] = self.normalized_data[var1] * self.normalized_data[var2]
        
        # 시간과 투입변수의 교차항 (기술변화)
        if self.include_time:
            for var in ln_input_vars:
                var_name = f'{var}_{self.time_var}'
                self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[self.time_var]
        
        # 회귀분석용 변수 리스트 생성
        self.translog_vars = []
        
        # 1차항
        self.translog_vars.extend(all_vars)
        
        # 2차항
        for var in all_vars:
            self.translog_vars.append(f'{var}2')
        
        # 투입변수들 간 교차항
        for i in range(n_inputs):
            for j in range(i+1, n_inputs):
                var1 = ln_input_vars[i]
                var2 = ln_input_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.translog_vars.append(var_name)
        
        # 시간-투입 교차항
        if self.include_time:
            for var in self.input_vars:
                self.translog_vars.append(f'ln_{var}_{self.time_var}')
        
        print(f"생성된 변수 수: {len(self.translog_vars)}")
        print("초월대수 함수 변수 생성 완료")
    
    def estimate_ols(self):
        """OLS 추정 (초기값용)"""
        print("\nOLS 추정 수행 중...")
        
        if self.translog_vars is None:
            self.create_translog_variables()
        
        # 종속변수
        y = self.normalized_data[f'ln_{self.output_var}']
        
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
        """확률변경함수 추정"""
        print(f"\n확률변경함수 추정 수행 중 (분포: {distribution})...")
        
        if 'ols' not in self.results:
            self.estimate_ols()
        
        # 초기값 설정
        ols_params = self.results['ols'].params.values
        
        # 종속변수와 독립변수
        y = self.normalized_data[f'ln_{self.output_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])  # 상수항 추가
        
        # 최대우도추정
        if distribution == 'half_normal':
            result = self._ml_estimation_half_normal(y, X, ols_params)
        else:
            raise ValueError("현재는 half-normal 분포만 지원합니다.")
        
        self.results['frontier'] = result
        
        print("확률변경함수 추정 완료")
        
        return result
    
    def _ml_estimation_half_normal(self, y, X, initial_params):
        """Half-normal 분포를 가정한 최대우도추정"""
        
        def log_likelihood(params):
            n_beta = X.shape[1]
            beta = params[:n_beta]
            sigma_u = np.exp(params[n_beta])  # 양수 제약
            sigma_v = np.exp(params[n_beta + 1])  # 양수 제약
            
            # 잔차
            residuals = y - X @ beta
            
            # 복합오차의 분산
            sigma_sq = sigma_u**2 + sigma_v**2
            sigma = np.sqrt(sigma_sq)
            
            # 람다
            lambd = sigma_u / sigma_v
            
            # 로그우도함수
            epsilon_star = residuals * lambd / sigma
            
            log_phi = norm.logpdf(residuals / sigma)
            log_Phi = norm.logcdf(-epsilon_star)
            
            log_likelihood = (np.log(2) - np.log(sigma) + log_phi + log_Phi).sum()
            
            return -log_likelihood  # 최소화를 위해 음수
        
        # 초기값 설정
        initial_sigma_u = 0.1
        initial_sigma_v = 0.1
        initial_vals = np.concatenate([initial_params, [np.log(initial_sigma_u), np.log(initial_sigma_v)]])
        
        # 최적화
        result = minimize(log_likelihood, initial_vals, method='BFGS')
        
        # 표준오차 계산 (간단한 근사)
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
        
        # 결과 정리
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
        """기술적 효율성 계산"""
        print("\n기술적 효율성 계산 중...")
        
        if 'frontier' not in self.results:
            self.estimate_stochastic_frontier()
        
        # 파라미터 추출
        beta = self.results['frontier']['beta']
        sigma_u = self.results['frontier']['sigma_u']
        sigma_v = self.results['frontier']['sigma_v']
        
        # 종속변수와 독립변수
        y = self.normalized_data[f'ln_{self.output_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        
        # 잔차
        residuals = y - X @ beta
        
        # 효율성 계산 (Jondrow et al., 1982)
        sigma_sq = sigma_u**2 + sigma_v**2
        sigma = np.sqrt(sigma_sq)
        lambd = sigma_u / sigma_v
        
        mu_star = -residuals * sigma_u**2 / sigma_sq
        sigma_star = sigma_u * sigma_v / sigma
        
        # 조건부 기댓값
        ratio = mu_star / sigma_star
        efficiency = np.exp(-mu_star + 0.5 * sigma_star**2) * (1 - norm.cdf(-ratio + sigma_star)) / (1 - norm.cdf(-ratio))
        
        self.normalized_data['efficiency'] = efficiency
        
        print("기술적 효율성 계산 완료")
        
        return efficiency
    
    def calculate_productivity_components(self):
        """총요소생산성 구성요인 계산"""
        print("\n총요소생산성 구성요인 계산 중...")
        
        if 'efficiency' not in self.normalized_data.columns:
            self.calculate_efficiency()
        
        # 파라미터
        beta = self.results['frontier']['beta']
        
        # 산출탄력성 계산
        self._calculate_output_elasticities(beta)
        
        # 기술변화 계산
        if self.include_time:
            self._calculate_technical_change(beta)
        
        # 규모의 경제 계산
        self._calculate_scale_effects()
        
        # 기술적 효율성 변화 계산
        self._calculate_efficiency_change()
        
        # 총요소생산성 계산
        self._calculate_tfp()
        
        print("총요소생산성 구성요인 계산 완료")
    
    def _calculate_output_elasticities(self, beta):
        """산출탄력성 계산"""
        data = self.normalized_data
        
        for i, var in enumerate(self.input_vars):
            ln_var = f'ln_{var}'
            
            # 1차항 계수
            beta_idx = 1 + i  # 상수항 제외
            elasticity = beta[beta_idx]
            
            # 2차항 추가
            var2_idx = self.translog_vars.index(f'{ln_var}2')
            elasticity += beta[1 + var2_idx] * data[ln_var]
            
            # 교차항 추가 (다른 투입변수들과)
            for j, other_var in enumerate(self.input_vars):
                if i != j:
                    other_ln_var = f'ln_{other_var}'
                    cross_term = f'{ln_var}_{other_var}' if i < j else f'{other_ln_var}_{var}'
                    if cross_term in self.translog_vars:
                        cross_idx = self.translog_vars.index(cross_term)
                        elasticity += beta[1 + cross_idx] * data[other_ln_var]
            
            # 시간과의 교차항
            if self.include_time:
                time_cross = f'{ln_var}_{self.time_var}'
                if time_cross in self.translog_vars:
                    time_cross_idx = self.translog_vars.index(time_cross)
                    elasticity += beta[1 + time_cross_idx] * data[self.time_var]
            
            data[f'eta_{var}'] = elasticity
    
    def _calculate_technical_change(self, beta):
        """기술변화 계산"""
        if not self.include_time:
            return
        
        data = self.normalized_data
        
        # 시간의 1차항 계수 찾기
        time_idx = self.translog_vars.index(self.time_var)
        tech_change = beta[1 + time_idx]  # 상수항 제외
        
        # 시간의 2차항
        time2_idx = self.translog_vars.index(f'{self.time_var}2')
        tech_change += beta[1 + time2_idx] * data[self.time_var]
        
        # 시간과 투입변수의 교차항
        for var in self.input_vars:
            time_cross = f'ln_{var}_{self.time_var}'
            if time_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_cross)
                tech_change += beta[1 + cross_idx] * data[f'ln_{var}']
        
        data['tech_change'] = tech_change
    
    def _calculate_scale_effects(self):
        """규모의 경제 효과 계산"""
        data = self.normalized_data
        
        # 총 산출탄력성 (규모의 경제)
        rts = sum(data[f'eta_{var}'] for var in self.input_vars)
        data['rts'] = rts
        
        # 각 요소의 비중 계산
        for var in self.input_vars:
            data[f'lambda_{var}'] = data[f'eta_{var}'] / rts
        
        # 투입증가율 계산
        data_sorted = data.sort_values([self.id_var, self.time_var])
        for var in self.input_vars:
            ln_var = f'ln_{var}'
            data_sorted[f'gr_{var}'] = data_sorted.groupby(self.id_var)[ln_var].diff()
        
        # 규모효과 계산
        scale_effect = (rts - 1) * sum(
            data_sorted[f'lambda_{var}'] * data_sorted[f'gr_{var}'] 
            for var in self.input_vars
        )
        
        data_sorted['scale_effect'] = scale_effect
        self.normalized_data = data_sorted
    
    def _calculate_efficiency_change(self):
        """기술적 효율성 변화 계산"""
        data = self.normalized_data.sort_values([self.id_var, self.time_var])
        
        # 효율성 변화율
        data['efficiency_change'] = data.groupby(self.id_var)['efficiency'].pct_change()
        
        self.normalized_data = data
    
    def _calculate_tfp(self):
        """총요소생산성 계산"""
        data = self.normalized_data
        
        # TFP = 규모효과 + 기술변화 + 효율성변화
        tfp_components = ['scale_effect']
        
        if self.include_time:
            tfp_components.append('tech_change')
        
        if 'efficiency_change' in data.columns:
            tfp_components.append('efficiency_change')
        
        data['tfp'] = sum(data[comp].fillna(0) for comp in tfp_components)
        
        self.normalized_data = data
    
    def print_results(self, save_path='tfp_results.csv'):
        """결과 출력"""
        print("\n" + "=" * 100)
        print("확률변경생산함수 추정 결과")
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
        
        # 3. 효율성 통계
        if 'efficiency' in self.normalized_data.columns:
            eff = self.normalized_data['efficiency']
            print(f"\n3. 기술적 효율성 통계")
            print("-" * 40)
            print(f"평균: {eff.mean():>20.4f}")
            print(f"표준편차: {eff.std():>15.4f}")
            print(f"최솟값: {eff.min():>16.4f}")
            print(f"최댓값: {eff.max():>16.4f}")
            print(f"중위수: {eff.median():>16.4f}")
        
        # 4. 총요소생산성 분해 결과 - 전체 시리즈 출력
        if 'tfp' in self.normalized_data.columns:
            print(f"\n4. 총요소생산성 분해 결과 (전체 시리즈)")
            print("=" * 80)
            
            # 시간별로 정렬
            tfp_data = self.normalized_data.sort_values([self.time_var, self.id_var]).copy()
            
            # 개체별 ID와 시간 포함
            display_cols = [self.id_var, self.time_var]
            
            # TFP 구성요소들
            tfp_components = ['tfp']
            if 'scale_effect' in tfp_data.columns:
                tfp_components.append('scale_effect')
            if 'tech_change' in tfp_data.columns:
                tfp_components.append('tech_change')
            if 'efficiency_change' in tfp_data.columns:
                tfp_components.append('efficiency_change')
            
            display_cols.extend(tfp_components)
            
            # 결측치 제거
            tfp_display = tfp_data[display_cols].dropna()
            
            if len(tfp_display) > 0:
                # 컬럼명 변경 (가독성)
                col_rename = {
                    'tfp': 'TFP',
                    'scale_effect': '규모효과',
                    'tech_change': '기술변화', 
                    'efficiency_change': '효율성변화'
                }
                
                tfp_display_renamed = tfp_display.rename(columns=col_rename)
                
                # **전체 데이터 출력**
                print("전체 TFP 분해 결과:")
                print(tfp_display_renamed.round(6).to_string(index=False))
                print(f"\n총 {len(tfp_display_renamed)}개 관측치")
                
                print("\n" + "-" * 60)
                print("TFP 구성요소 요약통계")
                print("-" * 60)
                print(f"{'구성요소':<15} {'평균':<12} {'표준편차':<12} {'최솟값':<12} {'최댓값':<12}")
                print("-" * 60)
                
                for comp in tfp_components:
                    if comp in tfp_data.columns:
                        comp_data = tfp_data[comp].dropna()
                        if len(comp_data) > 0:
                            comp_name = col_rename.get(comp, comp)
                            print(f"{comp_name:<15} {comp_data.mean():>8.6f} {comp_data.std():>12.6f} {comp_data.min():>12.6f} {comp_data.max():>12.6f}")
                
                print("-" * 60)
                
                # 시간별 평균
                if self.include_time and len(tfp_display) > 0:
                    print(f"\n시간별 평균:")
                    time_avg = tfp_data.groupby(self.time_var)[tfp_components].mean()
                    time_avg_renamed = time_avg.rename(columns=col_rename)
                    print(time_avg_renamed.round(6).to_string())
                
                # ID별 평균 (개체별 비교)
                print(f"\nID별 평균 (개체별 비교):")
                print("-" * 60)
                id_avg = tfp_data.groupby(self.id_var)[tfp_components].mean()
                id_avg_renamed = id_avg.rename(columns=col_rename)
                print(id_avg_renamed.round(6).to_string())
                
                # 효율성도 ID별로 보여주기
                if 'efficiency' in tfp_data.columns:
                    print(f"\nID별 평균 효율성:")
                    id_eff = tfp_data.groupby(self.id_var)['efficiency'].mean()
                    print(f"{'ID':<5} {'평균효율성':<12}")
                    print("-" * 20)
                    for id_val, eff_val in id_eff.items():
                        print(f"{id_val:<5} {eff_val:>8.6f}")
                
                # TFP 데이터 저장 (지정된 경로에)
                print(f"\n?? TFP 데이터 저장: {save_path}")
                tfp_display_renamed.to_csv(save_path, index=False, encoding='utf-8-sig')
                print("저장 완료!")
            
            else:
                print("TFP 데이터가 없습니다")
        
        # 5. 산출탄력성 통계
        elasticity_vars = [f'eta_{var}' for var in self.input_vars if f'eta_{var}' in self.normalized_data.columns]
        if elasticity_vars:
            print(f"\n5. 산출탄력성 통계")
            print("-" * 50)
            print(f"{'투입요소':<10} {'평균탄력성':<15} {'표준편차':<15}")
            print("-" * 50)
            
            for var in self.input_vars:
                eta_var = f'eta_{var}'
                if eta_var in self.normalized_data.columns:
                    eta_data = self.normalized_data[eta_var]
                    print(f"{var.upper():<10} {eta_data.mean():>10.6f} {eta_data.std():>15.6f}")
            
            # 규모의 경제
            if 'rts' in self.normalized_data.columns:
                rts_data = self.normalized_data['rts']
                rts_mean = rts_data.mean()
                print("-" * 50)
                print(f"{'규모합계':<10} {rts_mean:>10.6f} {rts_data.std():>15.6f}")
                
                if rts_mean > 1.01:
                    print("  → 규모수익 증가 (Increasing Returns to Scale)")
                elif rts_mean < 0.99:
                    print("  → 규모수익 감소 (Decreasing Returns to Scale)")
                else:
                    print("  → 규모수익 불변 (Constant Returns to Scale)")
        
        print("\n" + "=" * 100)
    
    def plot_results(self):
        """결과 시각화"""
        if 'efficiency' not in self.normalized_data.columns:
            print("효율성이 계산되지 않았습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 효율성 분포
        axes[0, 0].hist(self.normalized_data['efficiency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('기술적 효율성 분포')
        axes[0, 0].set_xlabel('효율성')
        axes[0, 0].set_ylabel('빈도')
        
        # 효율성 시계열
        if self.include_time:
            eff_by_time = self.normalized_data.groupby(self.time_var)['efficiency'].mean()
            axes[0, 1].plot(eff_by_time.index, eff_by_time.values, marker='o')
            axes[0, 1].set_title('시간별 평균 효율성')
            axes[0, 1].set_xlabel('시간')
            axes[0, 1].set_ylabel('평균 효율성')
        
        # TFP 분포
        if 'tfp' in self.normalized_data.columns:
            tfp_data = self.normalized_data['tfp'].dropna()
            if len(tfp_data) > 0:
                axes[1, 0].hist(tfp_data, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('총요소생산성 분포')
                axes[1, 0].set_xlabel('TFP')
                axes[1, 0].set_ylabel('빈도')
        
        # TFP 구성요소
        if all(col in self.normalized_data.columns for col in ['scale_effect', 'tech_change']):
            components = ['scale_effect']
            comp_labels = ['규모효과']
            
            if self.include_time:
                components.append('tech_change')
                comp_labels.append('기술변화')
            
            comp_means = [self.normalized_data[comp].dropna().mean() for comp in components]
            axes[1, 1].bar(comp_labels, comp_means)
            axes[1, 1].set_title('TFP 구성요소 평균')
            axes[1, 1].set_ylabel('기여도')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self, save_path='tfp_results.csv'):
        """전체 분석 실행"""
        print("확률변경생산함수 분석을 시작합니다...")
        
        # 1. EDA
        self.exploratory_data_analysis()
        
        # 2. 데이터 준비
        self.normalize_data()
        self.create_translog_variables()
        
        # 3. 추정
        self.estimate_ols()
        self.estimate_stochastic_frontier()
        
        # 4. 생산성 분석
        self.calculate_efficiency()
        self.calculate_productivity_components()
        
        # 5. 결과 출력
        self.print_results(save_path)
        self.plot_results()
        
        print("\n분석이 완료되었습니다!")
        
        return self.results, self.normalized_data


# 확률변경생산함수 분석 함수
def Run_StochasticFrontierProduction(data, output_var, input_vars, time_var='t', id_var='id', include_time=True, save_path='tfp_results.csv'):
    """
    확률변경생산함수 분석 실행 함수 - 한 번에 모든 결과 출력
    
    사용법:
    -------
    # 기본 사용
    StochasticFrontierProduction(data, 'y', ['l', 'k', 'm'])
    
    # 저장 위치 지정
    StochasticFrontierProduction(data, 'y', ['l', 'k', 'm'], save_path='my_analysis.csv')
    
    # 시간 트렌드 제외
    StochasticFrontierProduction(data, 'y', ['l', 'k'], include_time=False)
    
    결과:
    -----
    - 모든 분석 결과가 화면에 출력됨
    - TFP 분해 결과가 지정된 경로로 저장됨
    - ID별 평균이 추가로 출력됨
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

