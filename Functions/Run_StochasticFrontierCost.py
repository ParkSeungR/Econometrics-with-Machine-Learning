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
    ?뺣쪧蹂寃쎈퉬?⑺븿??遺꾩꽍, 湲곗닠 ?⑥쑉?? TFP 援ъ꽦?붿씤 遺꾪빐瑜??꾪븳 ?대? ?대옒??
    (?ъ슜?먭? 吏곸젒 ?몄텧?섏? ?딆쓬)
    """
    
    def __init__(self, data, cost_var, price_vars, input_vars, output_var, time_var='t', id_var='id', include_time=True):
        self.data = data.copy()
        self.cost_var = cost_var  # 珥앸퉬??
        self.price_vars = price_vars  # ?붿냼媛寃⑸뱾 
        self.input_vars = input_vars  # ?ъ엯?됰뱾
        self.output_var = output_var  # ?곗텧??
        self.time_var = time_var
        self.id_var = id_var
        self.include_time = include_time
        
        # 寃곌낵 ??μ슜 ?뺤뀛?덈━
        self.results = {}
        self.normalized_data = None
        self.translog_vars = None
        
        # ?곗씠??寃利?
        self._validate_data()
        
    def _validate_data(self):
        """?곗씠???좏슚??寃??""
        required_vars = [self.cost_var, self.output_var] + self.price_vars + self.input_vars + [self.id_var]
        if self.include_time:
            required_vars.append(self.time_var)
            
        missing_vars = [var for var in required_vars if var not in self.data.columns]
        if missing_vars:
            raise ValueError(f"?ㅼ쓬 蹂?섎뱾???곗씠?곗뿉 ?놁뒿?덈떎: {missing_vars}")
        
        # 媛寃⑸??섏? ?ъ엯??蹂?섏쓽 媛쒖닔媛 媛숈?吏 ?뺤씤
        if len(self.price_vars) != len(self.input_vars):
            raise ValueError(f"媛寃⑸???媛쒖닔({len(self.price_vars)})? ?ъ엯?됰???媛쒖닔({len(self.input_vars)})媛 ?ㅻ쫭?덈떎.")
            
        # 濡쒓렇 蹂?섏쓣 ?꾪빐 ?묒닔 泥댄겕
        for var in [self.cost_var, self.output_var] + self.price_vars + self.input_vars:
            if (self.data[var] <= 0).any():
                raise ValueError(f"蹂??{var}??0 ?댄븯??媛믪씠 ?덉뒿?덈떎. 濡쒓렇 蹂?섏씠 遺덇??ν빀?덈떎.")
        
        print(f"? ?곗씠??寃利??꾨즺:")
        print(f"   媛寃⑸??? {self.price_vars}")
        print(f"   ?ъ엯?됰??? {self.input_vars}")
        print(f"   珥앸퉬?⑸??? {self.cost_var}")
        print(f"   ?곗텧?됰??? {self.output_var}")
    
    def exploratory_data_analysis(self):
        """?먯깋???곗씠??遺꾩꽍 ?섑뻾"""
        print("=" * 60)
        print("?먯깋???곗씠??遺꾩꽍 (EDA) - ?뺣쪧蹂寃쎈퉬?⑺븿??)
        print("=" * 60)
        
        # 湲곕낯 蹂?섎뱾
        analysis_vars = [self.cost_var, self.output_var] + self.price_vars
        if self.include_time:
            analysis_vars.append(self.time_var)
        
        # 1. 湲곗큹?듦퀎??
        print("\n1. 湲곗큹?듦퀎??)
        print("-" * 40)
        desc_stats = self.data[analysis_vars].describe()
        print(desc_stats.round(4))
        
        # 2. ?곴?愿怨?
        print("\n2. ?곴?愿怨?議고쉶")
        print("-" * 40)
        corr_matrix = self.data[analysis_vars].corr()
        print(corr_matrix.round(4))
        
        return desc_stats, corr_matrix
    
    def normalize_data(self):
        """媛쒖껜蹂??됯퇏?쇰줈 ?곗씠???쒖???""
        print("\n?곗씠???쒖????섑뻾 以?..")
        
        self.normalized_data = self.data.copy()
        
        # 媛쒖껜蹂??됯퇏 怨꾩궛 (鍮꾩슜, 媛寃? ?곗텧??
        vars_to_normalize = [self.cost_var, self.output_var] + self.price_vars
        
        for var in vars_to_normalize:
            # 媛쒖껜蹂??됯퇏
            mean_by_id = self.data.groupby(self.id_var)[var].transform('mean')
            # ?쒖???
            self.normalized_data[f'nm_{var}'] = self.data[var] / mean_by_id
            # 濡쒓렇 蹂??
            self.normalized_data[f'ln_{var}'] = np.log(self.normalized_data[f'nm_{var}'])
        
        # ?ъ엯?됰룄 蹂듭궗 (鍮꾩슜紐?怨꾩궛??
        for input_var in self.input_vars:
            self.normalized_data[input_var] = self.data[input_var]
        
        if self.include_time:
            # ?쒓컙 蹂?섎뒗 濡쒓렇 蹂???놁씠 洹몃?濡??ъ슜 (1, 2, 3, ...)
            self.normalized_data[self.time_var] = self.data[self.time_var]
        
        print("?곗씠???쒖????꾨즺")
    
    def create_translog_variables(self):
        """珥덉썡???鍮꾩슜?⑥닔瑜??꾪븳 援먯감??蹂???앹꽦"""
        print("珥덉썡???鍮꾩슜?⑥닔 蹂???앹꽦 以?..")
        
        if self.normalized_data is None:
            self.normalize_data()
        
        # 濡쒓렇 蹂?섎맂 媛寃?蹂?섎뱾怨??곗텧??
        ln_price_vars = [f'ln_{var}' for var in self.price_vars]
        ln_output_var = f'ln_{self.output_var}'
        
        # 紐⑤뱺 ?듭떖蹂??(媛寃?+ ?곗텧??+ ?쒓컙蹂??
        all_vars = ln_price_vars + [ln_output_var]
        if self.include_time:
            all_vars.append(self.time_var)  # ?쒓컙? 洹몃깷 t
        
        # 2李⑦빆 ?앹꽦
        for var in all_vars:
            self.normalized_data[f'{var}2'] = 0.5 * self.normalized_data[var] ** 2
        
        # 媛寃⑸??섎뱾 媛?援먯감??
        n_prices = len(ln_price_vars)
        for i in range(n_prices):
            for j in range(i+1, n_prices):
                var1 = ln_price_vars[i]
                var2 = ln_price_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.normalized_data[var_name] = self.normalized_data[var1] * self.normalized_data[var2]
        
        # 媛寃⑷낵 ?곗텧?됱쓽 援먯감??
        for var in ln_price_vars:
            var_name = f'{var}_{self.output_var}'
            self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[ln_output_var]
        
        # ?쒓컙怨??ㅻⅨ 蹂?섎뱾??援먯감??
        if self.include_time:
            for var in ln_price_vars + [ln_output_var]:
                var_name = f'{var}_{self.time_var}'
                self.normalized_data[var_name] = self.normalized_data[var] * self.normalized_data[self.time_var]
        
        # ?듭떖遺꾩꽍??蹂??由ъ뒪???앹꽦
        self.translog_vars = []
        
        # 1李⑦빆
        self.translog_vars.extend(all_vars)
        
        # 2李⑦빆
        for var in all_vars:
            self.translog_vars.append(f'{var}2')
        
        # 媛寃⑸??섎뱾 媛?援먯감??
        for i in range(n_prices):
            for j in range(i+1, n_prices):
                var1 = ln_price_vars[i]
                var2 = ln_price_vars[j]
                var_name = f'{var1}_{var2.split("_")[1]}'
                self.translog_vars.append(var_name)
        
        # 媛寃??곗텧??援먯감??
        for var in self.price_vars:
            self.translog_vars.append(f'ln_{var}_{self.output_var}')
        
        # ?쒓컙 援먯감??
        if self.include_time:
            for var in self.price_vars + [self.output_var]:
                self.translog_vars.append(f'ln_{var}_{self.time_var}')
        
        print(f"?앹꽦??蹂???? {len(self.translog_vars)}")
        print("珥덉썡???鍮꾩슜?⑥닔 蹂???앹꽦 ?꾨즺")
    
    def estimate_ols(self):
        """OLS 異붿젙 (珥덇린媛믪슜)"""
        print("\nOLS 異붿젙 ?섑뻾 以?..")
        
        if self.translog_vars is None:
            self.create_translog_variables()
        
        # 醫낆냽蹂??(珥앸퉬??
        y = self.normalized_data[f'ln_{self.cost_var}']
        
        # ?낅┰蹂??
        X = self.normalized_data[self.translog_vars]
        X = sm.add_constant(X)
        
        # OLS 異붿젙
        ols_model = sm.OLS(y, X).fit()
        
        self.results['ols'] = ols_model
        
        print("OLS 異붿젙 ?꾨즺")
        print(f"R-squared: {ols_model.rsquared:.4f}")
        
        return ols_model
    
    def estimate_stochastic_frontier(self, distribution='half_normal'):
        """?뺣쪧蹂寃쎈퉬?⑺븿??異붿젙 - 媛쒖꽑??踰꾩쟾"""
        print(f"\n?뺣쪧蹂寃쎈퉬?⑺븿??異붿젙 ?섑뻾 以?(遺꾪룷: {distribution})...")
        
        if 'ols' not in self.results:
            print("OLS 異붿젙??癒쇱? ?섑뻾?⑸땲??..")
            self.estimate_ols()
        
        # OLS 寃곌낵 寃利?
        if not hasattr(self.results['ols'], 'params'):
            print("? OLS 異붿젙 寃곌낵媛 ?놁뒿?덈떎.")
            return self._create_fallback_result()
        
        # 珥덇린媛??ㅼ젙
        ols_params = self.results['ols'].params.values
        
        # OLS 寃곌낵 寃利?
        if np.any(np.isnan(ols_params)) or np.any(np.isinf(ols_params)):
            print("?? OLS ?뚮씪誘명꽣??NaN/Inf媛 ?덉뒿?덈떎. 湲곕낯媛믪쓣 ?ъ슜?⑸땲??")
            ols_params = np.zeros(len(ols_params))
            ols_params[0] = np.log(self.normalized_data[f'ln_{self.cost_var}'].mean())
        
        # 醫낆냽蹂?섏? ?낅┰蹂??
        y = self.normalized_data[f'ln_{self.cost_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])  # ?곸닔??異붽?
        
        # ?곗씠??寃利?
        if np.any(np.isnan(y)) or np.any(np.isnan(X)):
            print("? ?곗씠?곗뿉 NaN???덉뒿?덈떎.")
            return self._create_fallback_result()
        
        print(f"   ?곗씠???ш린: y={y.shape}, X={X.shape}")
        print(f"   OLS R짼: {self.results['ols'].rsquared:.4f}")
        
        # 理쒕??곕룄異붿젙
        print("?? 理쒕??곕룄異붿젙 ?쒖옉...")
        try:
            if distribution == 'half_normal':
                result = self._ml_estimation_half_normal_cost(y, X, ols_params)
            else:
                raise ValueError("?꾩옱??half-normal 遺꾪룷留?吏?먰빀?덈떎.")
        except Exception as e:
            print(f"? 理쒕??곕룄異붿젙 以??ㅻ쪟: {str(e)}")
            return self._create_fallback_result()
        
        # 寃곌낵 寃利?
        if result['success']:
            # ?뚮씪誘명꽣 ?⑸━??寃利?
            if (result['sigma_u'] > 0 and result['sigma_v'] > 0 and 
                result['sigma_u'] < 100 and result['sigma_v'] < 100 and
                np.isfinite(result['log_likelihood'])):
                
                self.results['frontier'] = result
                print("? ?뺣쪧蹂寃쎈퉬?⑺븿??異붿젙 ?꾨즺")
                
                # 紐⑤뜽 ?곹빀???뺣낫
                gamma = result['sigma_u']**2 / (result['sigma_u']**2 + result['sigma_v']**2)
                print(f"   款 = ?짼u/?짼: {gamma:.4f}")
                if gamma > 0.5:
                    print("   ??鍮꾪슚?⑥꽦???ㅼ감??二쇱슂 ?먯씤?낅땲??")
                else:
                    print("   ???뺣쪧???ㅼ감媛 二쇱슂 ?먯씤?낅땲??")
                
                return result
            else:
                print("?? 異붿젙???뚮씪誘명꽣媛 鍮꾪빀由ъ쟻?낅땲??")
                return self._create_fallback_result()
        else:
            print(f"?? 異붿젙 ?ㅽ뙣: {result.get('message', 'Unknown error')}")
            return self._create_fallback_result()
    
    def _create_fallback_result(self):
        """異붿젙 ?ㅽ뙣???泥?寃곌낵 ?앹꽦"""
        print("   ?泥?寃곌낵 ?앹꽦 以?..")
        
        # 湲곕낯 ?뚮씪誘명꽣 ?ㅼ젙
        n_params = len(self.translog_vars) + 1  # ?곸닔???ы븿
        fallback_beta = np.zeros(n_params)
        
        # ?곸닔??? ?됯퇏 鍮꾩슜?쇰줈 ?ㅼ젙
        fallback_beta[0] = self.normalized_data[f'ln_{self.cost_var}'].mean()
        
        # 湲곕낯 sigma 媛?
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
        print("   ?? ?泥??뚮씪誘명꽣 ?ъ슜 - 寃곌낵 ?댁꽍??二쇱쓽?섏꽭??")
        
        return fallback_result
    
    def _ml_estimation_half_normal_cost(self, y, X, initial_params):
        """鍮꾩슜?⑥닔??Half-normal 遺꾪룷瑜?媛?뺥븳 理쒕??곕룄異붿젙 - 媛쒖꽑??踰꾩쟾"""
        
        def log_likelihood(params):
            try:
                n_beta = X.shape[1]
                beta = params[:n_beta]
                log_sigma_u = params[n_beta]      
                log_sigma_v = params[n_beta + 1]  
                
                # sigma 怨꾩궛 (???덉젙??
                sigma_u = np.exp(np.clip(log_sigma_u, -10, 5))  # 洹밴컪 ?쒗븳
                sigma_v = np.exp(np.clip(log_sigma_v, -10, 5))
                
                # 理쒖냼媛?蹂댁옣
                sigma_u = np.maximum(sigma_u, 1e-4)
                sigma_v = np.maximum(sigma_v, 1e-4)
                
                # ?붿감 怨꾩궛
                residuals = y - X @ beta
                
                # ?붿감???ㅼ???泥댄겕
                if np.std(residuals) > 100 or np.std(residuals) < 1e-6:
                    return 1e8
                
                # 蹂듯빀?ㅼ감 ?뚮씪誘명꽣
                sigma_sq = sigma_u**2 + sigma_v**2
                sigma = np.sqrt(sigma_sq)
                
                if sigma < 1e-4 or sigma > 100:
                    return 1e8
                
                # ?뚮떎 (鍮꾩쑉 ?쒗븳)
                lambd = np.clip(sigma_u / sigma_v, 0.01, 100)
                
                # ?쒖??붾맂 ?붿감
                residuals_std = residuals / sigma
                residuals_std = np.clip(residuals_std, -8, 8)
                
                # epsilon* 怨꾩궛 (鍮꾩슜?⑥닔)
                epsilon_star = residuals_std * lambd
                epsilon_star = np.clip(epsilon_star, -8, 8)
                
                # 濡쒓렇 ?뺣쪧諛??怨꾩궛
                log_phi = -0.5 * np.log(2 * np.pi) - 0.5 * residuals_std**2
                
                # 濡쒓렇 ?꾩쟻遺꾪룷 怨꾩궛 (?덉젙??諛⑸쾿)
                log_Phi = np.where(epsilon_star > -5, 
                                  np.log(norm.cdf(epsilon_star) + 1e-15),
                                  epsilon_star - 0.5 * epsilon_star**2 - np.log(np.sqrt(2*np.pi)))
                
                # 濡쒓렇?곕룄 怨꾩궛
                log_likelihood_val = (np.log(2) - np.log(sigma) + log_phi + log_Phi).sum()
                
                # 理쒖쥌 寃利?
                if not np.isfinite(log_likelihood_val) or log_likelihood_val < -1e6:
                    return 1e8
                
                return -log_likelihood_val
                
            except Exception as e:
                return 1e8
        
        # 媛쒖꽑??珥덇린媛??ㅼ젙
        print("   珥덇린媛??ㅼ젙 以?..")
        ols_residuals = y - X @ initial_params
        ols_sigma = np.std(ols_residuals)
        
        # ????珥덇린媛??ъ슜 (?덉젙???섎졃???꾪빐)
        initial_sigma_u = np.clip(ols_sigma * 0.8, 0.05, 1.0)  # ????媛?
        initial_sigma_v = np.clip(ols_sigma * 0.6, 0.05, 1.0)  # ????媛?
        
        # log scale濡?珥덇린媛??ㅼ젙
        initial_vals = np.concatenate([
            initial_params, 
            [np.log(initial_sigma_u), np.log(initial_sigma_v)]
        ])
        
        print(f"   OLS ?: {ols_sigma:.4f}")
        print(f"   珥덇린 ?_u: {initial_sigma_u:.4f}, ?_v: {initial_sigma_v:.4f}")
        
        # ??愿???bounds ?ㅼ젙
        n_beta = len(initial_params)
        bounds = []
        
        # beta ?뚮씪誘명꽣: ???볦? 踰붿쐞
        for i in range(n_beta):
            bounds.append((-20, 20))  
        
        # log_sigma: ???볦? 踰붿쐞
        bounds.append((-8, 3))   # exp(-8) ? 0.0003, exp(3) ? 20
        bounds.append((-8, 3))
        
        # 理쒖쟻???쒕룄 (??愿????ㅼ젙)
        print("   理쒖쟻???쒖옉...")
        result = None
        
        # 諛⑸쾿 1: L-BFGS-B (??愿????ㅼ젙)
        try:
            print("   ?쒕룄 1: L-BFGS-B")
            result = minimize(log_likelihood, initial_vals, method='L-BFGS-B', 
                            bounds=bounds, 
                            options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-4})
            
            if result.success and result.fun < 1e7:
                print(f"   ? L-BFGS-B ?깃났: f={result.fun:.4f}")
            else:
                print(f"   L-BFGS-B ?ㅽ뙣: f={result.fun:.4f}")
                result = None
        except Exception as e:
            print(f"   L-BFGS-B ?ㅻ쪟: {str(e)}")
            result = None
        
        # 諛⑸쾿 2: ?ㅻⅨ 珥덇린媛믪쑝濡??ъ떆??
        if result is None:
            try:
                print("   ?쒕룄 2: ?ㅻⅨ 珥덇린媛믪쑝濡?L-BFGS-B")
                # 珥덇린媛믪쓣 ?쎄컙 蹂寃?
                alt_initial_vals = initial_vals.copy()
                alt_initial_vals[-2:] += np.random.normal(0, 0.5, 2)  # sigma 珥덇린媛?蹂寃?
                
                result = minimize(log_likelihood, alt_initial_vals, method='L-BFGS-B', 
                                bounds=bounds, 
                                options={'maxiter': 2000, 'ftol': 1e-6})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? ???L-BFGS-B ?깃났: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   ???L-BFGS-B ?ㅽ뙣: {str(e)}")
                result = None
        
        # 諛⑸쾿 3: BFGS (bounds ?놁쓬)
        if result is None:
            try:
                print("   ?쒕룄 3: BFGS")
                result = minimize(log_likelihood, initial_vals, method='BFGS', 
                                options={'maxiter': 1500, 'gtol': 1e-4})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? BFGS ?깃났: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   BFGS ?ㅽ뙣: {str(e)}")
                result = None
        
        # 諛⑸쾿 4: Powell (derivative-free)
        if result is None:
            try:
                print("   ?쒕룄 4: Powell")
                result = minimize(log_likelihood, initial_vals, method='Powell', 
                                options={'maxiter': 1000, 'ftol': 1e-6})
                
                if result.success and result.fun < 1e7:
                    print(f"   ? Powell ?깃났: f={result.fun:.4f}")
                else:
                    result = None
            except Exception as e:
                print(f"   Powell ?ㅽ뙣: {str(e)}")
                result = None
        
        # 紐⑤뱺 諛⑸쾿 ?ㅽ뙣??湲곕낯媛?諛섑솚
        if result is None:
            print("   ?? 紐⑤뱺 理쒖쟻??諛⑸쾿 ?ㅽ뙣 - 湲곕낯媛??ъ슜")
            result = type('obj', (object,), {
                'x': initial_vals, 
                'fun': 1e10, 
                'success': False,
                'message': 'All optimization methods failed'
            })()
        
        # 媛쒖꽑???쒖??ㅼ감 怨꾩궛
        print("   ?쒖??ㅼ감 怨꾩궛 以?..")
        try:
            if result.success and result.fun < 1e9:
                # Hessian 怨꾩궛 (???덉젙?곸씤 諛⑸쾿)
                eps = 1e-5
                n_params = len(result.x)
                hessian = np.zeros((n_params, n_params))
                
                f0 = log_likelihood(result.x)
                
                # diagonal elements留?怨꾩궛 (???덉젙??
                for i in range(n_params):
                    x_plus = result.x.copy()
                    x_minus = result.x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    
                    f_plus = log_likelihood(x_plus)
                    f_minus = log_likelihood(x_minus)
                    
                    second_deriv = (f_plus - 2*f0 + f_minus) / (eps**2)
                    hessian[i, i] = abs(second_deriv)  # ?덈뙎媛??ъ슜
                
                # ?쒖??ㅼ감 怨꾩궛
                std_errors = np.zeros(n_params)
                for i in range(n_params):
                    if hessian[i, i] > 1e-10:
                        std_errors[i] = 1.0 / np.sqrt(hessian[i, i])
                    else:
                        std_errors[i] = np.nan
                
                # 鍮꾪빀由ъ쟻???쒖??ㅼ감 ?쒗븳
                std_errors = np.where(std_errors > 100, np.nan, std_errors)
                
            else:
                std_errors = np.full(len(result.x), np.nan)
                
        except Exception as e:
            print(f"   ?쒖??ㅼ감 怨꾩궛 ?ㅻ쪟: {str(e)}")
            std_errors = np.full(len(result.x), np.nan)
        
        # 寃곌낵 ?뺣━
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
        
        # 寃곌낵 異쒕젰
        if estimated_params['success']:
            print(f"   ? 理쒖쟻???깃났!")
            print(f"   濡쒓렇?곕룄: {estimated_params['log_likelihood']:.4f}")
            print(f"   ?_u: {estimated_params['sigma_u']:.4f}")
            print(f"   ?_v: {estimated_params['sigma_v']:.4f}")
            print(f"   貫 (?_u/?_v): {estimated_params['sigma_u']/estimated_params['sigma_v']:.4f}")
        else:
            print(f"   ?? 理쒖쟻???ㅽ뙣: {estimated_params['message']}")
            print(f"   ?⑥닔媛? {result.fun:.4f}")
        
        return estimated_params
    
    def calculate_efficiency(self):
        """湲곗닠???⑥쑉??Technical Efficiency) 怨꾩궛 - 媛쒖꽑??踰꾩쟾"""
        print("\n湲곗닠???⑥쑉??TE) 怨꾩궛 以?..")
        
        if 'frontier' not in self.results:
            self.estimate_stochastic_frontier()
        
        # 異붿젙???ㅽ뙣??寃쎌슦 泥섎━
        if not self.results['frontier']['success']:
            print("?? ?뺣쪧蹂寃쏀븿??異붿젙???ㅽ뙣?섏뿬 ?⑥쑉?깆쓣 怨꾩궛?????놁뒿?덈떎.")
            # 湲곕낯媛믪쑝濡?1.0 (?꾩쟾 ?⑥쑉?? ?좊떦
            self.normalized_data['technical_efficiency'] = 1.0
            return np.ones(len(self.normalized_data))
        
        # ?뚮씪誘명꽣 異붿텧
        beta = self.results['frontier']['beta']
        sigma_u = self.results['frontier']['sigma_u']
        sigma_v = self.results['frontier']['sigma_v']
        
        # numerical stability 泥댄겕
        if sigma_u < 1e-6 or sigma_v < 1e-6:
            print("?? ? 媛믪씠 ?덈Т ?묒븘 ?⑥쑉??怨꾩궛??嫄대꼫?곷땲??")
            self.normalized_data['technical_efficiency'] = 1.0
            return np.ones(len(self.normalized_data))
        
        # 醫낆냽蹂?섏? ?낅┰蹂??
        y = self.normalized_data[f'ln_{self.cost_var}'].values
        X = self.normalized_data[self.translog_vars].values
        X = np.column_stack([np.ones(len(X)), X])
        
        # ?붿감
        residuals = y - X @ beta
        
        # 湲곗닠???⑥쑉??怨꾩궛 (鍮꾩슜?⑥닔??- Jondrow et al., 1982)
        sigma_sq = sigma_u**2 + sigma_v**2
        sigma = np.sqrt(sigma_sq)
        lambd = sigma_u / sigma_v
        
        # numerical stability瑜??꾪븳 泥섎━
        try:
            mu_star = residuals * sigma_u**2 / sigma_sq  # 鍮꾩슜?⑥닔??遺??諛섎?
            sigma_star = sigma_u * sigma_v / sigma
            
            # numerical stability: 洹밴컪 ?쒗븳
            mu_star = np.clip(mu_star, -10, 10)
            sigma_star = np.maximum(sigma_star, 1e-6)
            
            # 議곌굔遺 湲곕뙎媛?(湲곗닠???⑥쑉??
            ratio = mu_star / sigma_star
            ratio = np.clip(ratio, -10, 10)  # 洹밴컪 ?쒗븳
            
            # ?덉젙?곸씤 怨꾩궛???꾪빐 ?④퀎蹂꾨줈 怨꾩궛
            exp_term = np.exp(-mu_star + 0.5 * sigma_star**2)
            
            # CDF 怨꾩궛 ??numerical stability
            cdf_term1 = norm.cdf(ratio + sigma_star)
            cdf_term2 = norm.cdf(ratio)
            
            # 0?쇰줈 ?섎늻湲?諛⑹?
            denominator = 1 - cdf_term2
            denominator = np.maximum(denominator, 1e-10)
            
            numerator = 1 - cdf_term1
            
            technical_efficiency = exp_term * numerator / denominator
            
            # 寃곌낵 寃利?諛??꾩쿂由?
            technical_efficiency = np.clip(technical_efficiency, 1e-6, 1.0)
            
            # NaN?대굹 inf 泥댄겕
            invalid_mask = ~np.isfinite(technical_efficiency)
            if invalid_mask.any():
                print(f"?? {invalid_mask.sum()}媛?愿痢↔컪?먯꽌 ?⑥쑉??怨꾩궛 ?ㅻ쪟 - 湲곕낯媛?0.5 ?좊떦")
                technical_efficiency[invalid_mask] = 0.5
            
            self.normalized_data['technical_efficiency'] = technical_efficiency
            
            print("? 湲곗닠???⑥쑉??TE) 怨꾩궛 ?꾨즺")
            print(f"   ?됯퇏 TE: {technical_efficiency.mean():.4f}")
            print(f"   理쒖냼 TE: {technical_efficiency.min():.4f}")
            print(f"   理쒕? TE: {technical_efficiency.max():.4f}")
            
            return technical_efficiency
            
        except Exception as e:
            print(f"? ?⑥쑉??怨꾩궛 以??ㅻ쪟: {str(e)}")
            # ?ㅻ쪟 ??湲곕낯媛??좊떦
            technical_efficiency = np.full(len(residuals), 0.8)
            self.normalized_data['technical_efficiency'] = technical_efficiency
            return technical_efficiency
    
    def calculate_cost_economics(self):
        """鍮꾩슜?⑥닔 寃쎌젣?숈쟻 吏??怨꾩궛"""
        print("\n鍮꾩슜?⑥닔 寃쎌젣?숈쟻 吏??怨꾩궛 以?..")
        
        if 'technical_efficiency' not in self.normalized_data.columns:
            self.calculate_efficiency()
        
        # ?뚮씪誘명꽣
        beta = self.results['frontier']['beta']
        
        # ?ㅼ젣 鍮꾩슜紐?怨꾩궛 (?щ컮瑜?諛⑸쾿)
        self._calculate_actual_cost_shares()
        
        # 媛寃⑺깂?μ꽦 怨꾩궛
        self._calculate_price_elasticities(beta)
        
        # 湲곗닠蹂??怨꾩궛 (?섏젙??- 鍮꾩슜?⑥닔 湲곗?)
        if self.include_time:
            self._calculate_technical_change_cost_corrected(beta)
        
        # 洹쒕え??寃쎌젣 怨꾩궛 (?섏젙??
        self._calculate_scale_economies(beta)
        
        # TFP 遺꾪빐 怨꾩궛
        self._calculate_tfp_decomposition()
        
        print("鍮꾩슜?⑥닔 寃쎌젣?숈쟻 吏??怨꾩궛 ?꾨즺")
        
        print("\n?? TFP 遺꾪빐 怨듭떇:")
        print("TFP 利앷???= 湲곗닠蹂??+ 湲곗닠?곹슚?⑥꽦蹂??+ 洹쒕え?섍꼍?쒗슚怨?)
        print("?ш린??")
        print("  ? 湲곗닠蹂?? -?굃nC/?굏 (鍮꾩슜 媛먯냼 ?④낵)")  
        print("  ? 湲곗닠?곹슚?⑥꽦蹂?? ?굃n(TE)/?굏")
        print("  ? 洹쒕え?섍꼍?쒗슚怨? (1-1/洹쒕え?섍꼍?? 횞 ?곗텧?됱쬆媛??)
    
    def _calculate_actual_cost_shares(self):
        """?ㅼ젣 鍮꾩슜紐?怨꾩궛 (?щ컮瑜?諛⑸쾿)"""
        data = self.normalized_data
        
        print("\n?ㅼ젣 鍮꾩슜紐?怨꾩궛 以?..")
        
        # 珥앸퉬??怨꾩궛 諛?寃利?
        total_cost_calculated = sum(data[price_var] * data[input_var] 
                                  for price_var, input_var in zip(self.price_vars, self.input_vars))
        data['calculated_total_cost'] = total_cost_calculated
        
        # ?ㅼ젣 珥앸퉬?⑷낵 怨꾩궛??珥앸퉬??鍮꾧탳
        actual_total_cost = data[self.cost_var]
        cost_diff = abs(actual_total_cost - total_cost_calculated).mean()
        print(f"   珥앸퉬??寃利? ?ㅼ젣 vs 怨꾩궛 李⑥씠 ?됯퇏 = {cost_diff:.6f}")
        
        # ?ㅼ젣 鍮꾩슜紐?怨꾩궛: si = pi * xi / TC
        for price_var, input_var in zip(self.price_vars, self.input_vars):
            input_cost = data[price_var] * data[input_var]
            cost_share = input_cost / actual_total_cost  # ?ㅼ젣 珥앸퉬???ъ슜
            data[f'share_{input_var}'] = cost_share
        
        # 鍮꾩슜紐??⑷퀎 寃利?
        total_shares = sum(data[f'share_{input_var}'] for input_var in self.input_vars)
        data['total_shares'] = total_shares
        
        print(f"\n? 鍮꾩슜紐?寃利?")
        print(f"   鍮꾩슜紐??⑷퀎 ?됯퇏: {total_shares.mean():.6f}")
        print(f"   鍮꾩슜紐??⑷퀎 ?쒖??몄감: {total_shares.std():.6f}")
        print(f"   鍮꾩슜紐??⑷퀎 踰붿쐞: [{total_shares.min():.6f}, {total_shares.max():.6f}]")
        
        # 媛쒕퀎 鍮꾩슜紐?異쒕젰
        for input_var in self.input_vars:
            share_mean = data[f'share_{input_var}'].mean()
            print(f"   ?됯퇏 {input_var.upper()} 鍮꾩슜紐? {share_mean:.6f}")
        
        if abs(total_shares.mean() - 1.0) < 0.01:
            print("   ? 鍮꾩슜紐??⑷퀎媛 ?щ컮由낅땲??(? 1)")
        else:
            print(f"   ??  鍮꾩슜紐??⑷퀎媛 1?먯꽌 踰쀬뼱?ъ뒿?덈떎: {total_shares.mean():.6f}")
    
    def _calculate_price_elasticities(self, beta):
        """媛寃⑺깂?μ꽦 怨꾩궛"""
        data = self.normalized_data
        
        # ?먭린媛寃⑺깂?μ꽦怨?援먯감媛寃⑺깂?μ꽦
        for i, (price_var, input_var) in enumerate(zip(self.price_vars, self.input_vars)):
            share_i = data[f'share_{input_var}']
            
            for j, (price_var_j, input_var_j) in enumerate(zip(self.price_vars, self.input_vars)):
                if i == j:
                    # ?먭린媛寃⑺깂?μ꽦
                    ln_var = f'ln_{price_var}'
                    beta_idx = 1 + i
                    elasticity = beta[beta_idx]
                    
                    # 2李⑦빆
                    var2_idx = self.translog_vars.index(f'{ln_var}2')
                    elasticity += beta[1 + var2_idx] * data[ln_var]
                    
                    # 理쒖쥌 ?꾨젰??
                    elasticity = elasticity / share_i - 1
                    
                else:
                    # 援먯감媛寃⑺깂?μ꽦
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
        """湲곗닠蹂??怨꾩궛 (鍮꾩슜?⑥닔) - 怨꾩닔 ?뺤씤 諛??뺤젙??怨꾩궛"""
        if not self.include_time:
            return
        
        data = self.normalized_data
        
        print(f"\n?? 湲곗닠蹂??愿??怨꾩닔 ?뺤씤:")
        print("-" * 50)
        
        # ?쒓컙??1李⑦빆 怨꾩닔 ?뺤씤
        time_idx = self.translog_vars.index(self.time_var)
        beta_t = beta[1 + time_idx]  # ?곸닔???쒖쇅
        print(f"   棺_t (?쒓컙 1李⑦빆 怨꾩닔): {beta_t:.6f}")
        
        # ?쒓컙??2李⑦빆 怨꾩닔 ?뺤씤
        time2_idx = self.translog_vars.index(f'{self.time_var}2')
        beta_t2 = beta[1 + time2_idx]
        print(f"   棺_tt (?쒓컙 2李⑦빆 怨꾩닔): {beta_t2:.6f}")
        
        # ?쒓컙怨?媛寃⑸??섏쓽 援먯감??怨꾩닔 ?뺤씤
        print("   ?쒓컙-媛寃?援먯감??怨꾩닔:")
        time_price_coeffs = {}
        for var in self.price_vars:
            time_cross = f'ln_{var}_{self.time_var}'
            if time_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_cross)
                beta_cross = beta[1 + cross_idx]
                time_price_coeffs[var] = beta_cross
                print(f"     棺_{var}_t: {beta_cross:.6f}")
        
        # ?쒓컙怨??곗텧?됱쓽 援먯감??怨꾩닔 ?뺤씤
        time_output_cross = f'ln_{self.output_var}_{self.time_var}'
        beta_yt = 0
        if time_output_cross in self.translog_vars:
            cross_idx = self.translog_vars.index(time_output_cross)
            beta_yt = beta[1 + cross_idx]
            print(f"   棺_y_t (?쒓컙-?곗텧??援먯감??: {beta_yt:.6f}")
        
        print("-" * 50)
        
        # 湲곗닠蹂??怨꾩궛: TECH = -?굃nC/?굏 (PDF 諛⑸쾿濡?
        # ?굃nC/?굏 = 棺_t + 棺_tt*t + 誇棺_it*ln(pi) + 棺_yt*ln(y)
        # ?곕씪??TECH = -(棺_t + 棺_tt*t + 誇棺_it*ln(pi) + 棺_yt*ln(y))
        
        # 1李⑦빆: -棺_t
        tech_change = -beta_t
        
        # 2李⑦빆: -棺_tt * t
        tech_change += -beta_t2 * data[self.time_var]
        
        # ?쒓컙-媛寃?援먯감?? -誇棺_it * ln(pi)
        for var in self.price_vars:
            if var in time_price_coeffs:
                tech_change += -time_price_coeffs[var] * data[f'ln_{var}']
        
        # ?쒓컙-?곗텧??援먯감?? -棺_yt * ln(y)
        if beta_yt != 0:
            tech_change += -beta_yt * data[f'ln_{self.output_var}']
        
        data['tech_change_cost'] = tech_change
        
        print(f"\n?? 湲곗닠蹂??怨꾩궛 寃곌낵:")
        print(f"   怨듭떇: TECH = -?굃nC/?굏")
        print(f"   ?됯퇏 湲곗닠蹂?? {tech_change.mean():.6f}")
        print(f"   湲곗닠蹂??踰붿쐞: [{tech_change.min():.6f}, {tech_change.max():.6f}]")
        
        # ?댁꽍 ?꾩?留?
        if tech_change.mean() > 0:
            print("   ? ?묒닔 ??湲곗닠吏꾨낫 (鍮꾩슜 媛먯냼)")
        elif tech_change.mean() < 0:
            print("   ??  ?뚯닔 ??湲곗닠?대낫 ?먮뒗 鍮꾩슜 利앷?")
            print("   ?? 媛?ν븳 ?먯씤:")
            print("      - ?쒓컙怨꾩닔 棺_t > 0 (?쒓컙???곕씪 鍮꾩슜利앷?)")
            print("      - ?곗씠?곗뿉 湲곗닠?대낫 諛섏쁺")
            print("      - 異붿젙 紐⑤뜽???쒓퀎")
        else:
            print("   ??  ????湲곗닠蹂???놁쓬")
        
        # 援ъ꽦?붿냼蹂?湲곗뿬??遺꾩꽍
        print(f"\n?? 湲곗닠蹂??援ъ꽦?붿냼 遺꾩꽍:")
        component1 = -beta_t
        component2 = (-beta_t2 * data[self.time_var]).mean()
        print(f"   1李⑦빆 湲곗뿬??(-棺_t): {component1:.6f}")
        print(f"   2李⑦빆 湲곗뿬???됯퇏 (-棺_tt*t): {component2:.6f}")
        
        total_cross_effect = 0
        for var in self.price_vars:
            if var in time_price_coeffs:
                cross_effect = (-time_price_coeffs[var] * data[f'ln_{var}']).mean()
                total_cross_effect += cross_effect
                print(f"   {var} 援먯감???됯퇏: {cross_effect:.6f}")
        
        if beta_yt != 0:
            output_cross_effect = (-beta_yt * data[f'ln_{self.output_var}']).mean()
            total_cross_effect += output_cross_effect
            print(f"   ?곗텧??援먯감???됯퇏: {output_cross_effect:.6f}")
        
        print(f"   珥?援먯감???④낵: {total_cross_effect:.6f}")
        print(f"   ?꾩껜 ?됯퇏: {component1 + component2 + total_cross_effect:.6f}")
        print("-" * 50)
    
    def _calculate_scale_economies(self, beta):
        """洹쒕え??寃쎌젣 怨꾩궛 (鍮꾩슜?⑥닔)"""
        data = self.normalized_data
        
        # ?곗텧?됱뿉 ???鍮꾩슜?꾨젰??(IRTS = ?굃nC/?굃nY)
        ln_output_var = f'ln_{self.output_var}'
        output_idx = self.translog_vars.index(ln_output_var)
        
        # 1李⑦빆: 棺_y
        output_elasticity = beta[1 + output_idx]  # ?곸닔???쒖쇅
        
        # 2李⑦빆: 棺_yy * ln(y)
        output2_idx = self.translog_vars.index(f'{ln_output_var}2')
        output_elasticity += beta[1 + output2_idx] * data[ln_output_var]
        
        # 媛寃??곗텧??援먯감?? 誇棺_iy * ln(wi)
        for var in self.price_vars:
            output_cross = f'ln_{var}_{self.output_var}'
            if output_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(output_cross)
                output_elasticity += beta[1 + cross_idx] * data[f'ln_{var}']
        
        # ?쒓컙-?곗텧??援먯감?? 棺_yt * t
        if self.include_time:
            time_output_cross = f'ln_{self.output_var}_{self.time_var}'
            if time_output_cross in self.translog_vars:
                cross_idx = self.translog_vars.index(time_output_cross)
                output_elasticity += beta[1 + cross_idx] * data[self.time_var]
        
        # 洹쒕え??寃쎌젣 = 1 / IRTS (?ш린??IRTS = ?곗텧??鍮꾩슜?꾨젰??
        scale_economies = 1 / output_elasticity
        
        # ?곗씠?곗뿉 ???(TFP 遺꾪빐?먯꽌 ?ъ슜)
        data['irts'] = output_elasticity  # ?곗텧??鍮꾩슜?꾨젰??(1/RTS)
        data['output_elasticity'] = output_elasticity  # 湲곗〈 ?명솚??
        data['scale_economies'] = scale_economies
    
    def _calculate_tfp_decomposition(self):
        """TFP 利앷??④낵 援ъ꽦?붿씤 遺꾪빐 怨꾩궛"""
        data = self.normalized_data.sort_values([self.id_var, self.time_var])
        
        # 1. 湲곗닠???⑥쑉??蹂??怨꾩궛 (TE??濡쒓렇 李⑤텇)
        data['ln_te'] = np.log(data['technical_efficiency'])
        data['tech_efficiency_change'] = data.groupby(self.id_var)['ln_te'].diff()
        
        # 2. ?곗텧??利앷???怨꾩궛 (洹쒕え?④낵 怨꾩궛??
        data['ln_output'] = data[f'ln_{self.output_var}']
        data['output_growth'] = data.groupby(self.id_var)['ln_output'].diff()
        
        # 3. 湲곗닠蹂???④낵 (?섏젙??- ?대? ?щ컮瑜?遺?몃줈 怨꾩궛??
        if 'tech_change_cost' in data.columns:
            # ?대? Stata 諛⑹떇?쇰줈 怨꾩궛?섏뼱 ?щ컮瑜?遺?몃? 媛吏?
            data['tech_change_effect'] = data['tech_change_cost']
        else:
            data['tech_change_effect'] = 0
        
        # 4. 洹쒕え?④낵 怨꾩궛 = (1-IRTS) 횞 ?곗텧利앷???(PDF 諛⑸쾿濡?
        # Stata: gen SCALE = (1-IRTS)*gr_y
        if 'irts' in data.columns:
            data['scale_effect'] = (1 - data['irts']) * data['output_growth']
        else:
            data['scale_effect'] = 0
        
        # 5. TFP 利앷???怨꾩궛 = SCALE + TECH + TEFF (PDF 諛⑸쾿濡?
        # Stata: gen TFP = SCALE + TECH + TEFF
        data['tfp_growth'] = (data['scale_effect'] + 
                             data['tech_change_effect'] + 
                             data['tech_efficiency_change'])
        
        # 6. 諛깅텇??蹂?섏쓣 ?꾪븳 蹂?섎뱾
        for var in ['tech_efficiency_change', 'tech_change_effect', 'scale_effect', 
                   'tfp_growth', 'output_growth']:
            if var in data.columns:
                data[f'{var}_pct'] = data[var] * 100
        
        self.normalized_data = data
        
        print(f"\n?? TFP 遺꾪빐 怨꾩궛 ?꾨즺:")
        valid_data = data.dropna(subset=['tfp_growth', 'tech_change_effect', 'tech_efficiency_change', 'scale_effect'])
        if len(valid_data) > 0:
            print(f"   ?됯퇏 TFP 利앷??? {valid_data['tfp_growth'].mean()*100:.4f}%")
            print(f"   ?됯퇏 湲곗닠蹂?? {valid_data['tech_change_effect'].mean()*100:.4f}%")
            print(f"   ?됯퇏 湲곗닠?곹슚?⑥꽦蹂?? {valid_data['tech_efficiency_change'].mean()*100:.4f}%")
            print(f"   ?됯퇏 洹쒕え?섍꼍?쒗슚怨? {valid_data['scale_effect'].mean()*100:.4f}%")
    
    def print_results(self, save_path='cost_results.csv'):
        """寃곌낵 異쒕젰 - TFP 遺꾪빐 4媛吏 ?붿냼留?異쒕젰"""
        print("\n" + "=" * 100)
        print("?뺣쪧蹂寃쎈퉬?⑺븿??異붿젙 寃곌낵")
        print("=" * 100)
        
        # 1. ?뚮씪誘명꽣 異붿젙移?異쒕젰
        if 'frontier' in self.results:
            frontier = self.results['frontier']
            beta = frontier['beta']
            std_errors = frontier.get('std_errors', np.full(len(beta), np.nan))
            
            print("\n1. ?뚮씪誘명꽣 異붿젙移?)
            print("-" * 100)
            print(f"{'蹂?섎챸':<20} {'怨꾩닔':<12} {'?쒖??ㅼ감':<12} {'t-媛?:<10} {'p-媛?:<10} {'?좎쓽??:<8}")
            print("-" * 100)
            
            # ?곸닔??
            t_stat = beta[0] / std_errors[0] if not np.isnan(std_errors[0]) and std_errors[0] != 0 else np.nan
            p_value = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
            significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
            
            print(f"{'?곸닔??:<20} {beta[0]:>8.6f} {std_errors[0]:>8.4f} {t_stat:>8.3f} {p_value:>8.4f} {significance:<8}")
            
            # 蹂?섎퀎 怨꾩닔
            for i, var_name in enumerate(self.translog_vars):
                coef = beta[i + 1]  # ?곸닔???쒖쇅
                se = std_errors[i + 1] if i + 1 < len(std_errors) else np.nan
                t_stat = coef / se if not np.isnan(se) and se != 0 else np.nan
                p_value = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
                significance = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.1 else ""
                
                print(f"{var_name:<20} {coef:>8.6f} {se:>8.4f} {t_stat:>8.3f} {p_value:>8.4f} {significance:<8}")
            
            print("-" * 100)
            print("?좎쓽?? *** p<0.01, ** p<0.05, * p<0.1")
            
            # ?? 湲곗닠蹂??愿??怨꾩닔 ?댁꽍 異붽?
            print(f"\n?? 湲곗닠蹂??愿??怨꾩닔 ?댁꽍:")
            print("-" * 50)
            
            # ?쒓컙 蹂??怨꾩닔 李얘린 諛??댁꽍
            if self.include_time and self.time_var in self.translog_vars:
                time_idx = self.translog_vars.index(self.time_var)
                beta_t = beta[time_idx + 1]  # ?곸닔???쒖쇅
                
                print(f"   ?쒓컙(t) 怨꾩닔 棺_t = {beta_t:.6f}")
                if beta_t > 0:
                    print("   ??棺_t > 0: ?쒓컙???곕씪 鍮꾩슜利앷? (湲곗닠?대낫 ?먮뒗 鍮꾩슜?곸듅 ?붿씤)")
                    print("   ??湲곗닠蹂??TECH = -棺_t = {:.6f} (?뚯닔)".format(-beta_t))
                elif beta_t < 0:
                    print("   ??棺_t < 0: ?쒓컙???곕씪 鍮꾩슜媛먯냼 (湲곗닠吏꾨낫)")
                    print("   ??湲곗닠蹂??TECH = -棺_t = {:.6f} (?묒닔)".format(-beta_t))
                else:
                    print("   ??棺_t = 0: ?쒓컙???곕Ⅸ 鍮꾩슜蹂???놁쓬")
                
                print(f"   ?? 湲곗닠蹂??怨듭떇: TECH = -?굃nC/?굏 = -棺_t - 棺_tt횞t - 誇棺_it횞ln(pi) - 棺_yt횞ln(y)")
            
            print("-" * 50)
            
            # 2. 紐⑤뜽 ?듦퀎??
            print(f"\n2. 紐⑤뜽 ?듦퀎??)
            print("-" * 40)
            print(f"愿痢≪닔: {frontier.get('n_obs', 'N/A'):>20}")
            print(f"?뚮씪誘명꽣 ?? {frontier.get('n_params', 'N/A'):>15}")
            print(f"濡쒓렇?곕룄: {frontier['log_likelihood']:>15.4f}")
            print(f"?쒓렇留?u: {frontier['sigma_u']:>15.4f}")
            print(f"?쒓렇留?v: {frontier['sigma_v']:>15.4f}")
            print(f"?쒓렇留댟? {frontier['sigma_u']**2 + frontier['sigma_v']**2:>15.4f}")
            print(f"?뚮떎 (?u/?v): {frontier['sigma_u']/frontier['sigma_v']:>10.4f}")
            print(f"款 = ?u짼/?짼: {frontier['sigma_u']**2/(frontier['sigma_u']**2 + frontier['sigma_v']**2):>13.4f}")
        
        # 3. 湲곗닠???⑥쑉??TE) ?듦퀎
        if 'technical_efficiency' in self.normalized_data.columns:
            te = self.normalized_data['technical_efficiency']
            print(f"\n3. 湲곗닠???⑥쑉??TE) ?듦퀎")
            print("-" * 40)
            print(f"?됯퇏: {te.mean():>20.4f}")
            print(f"?쒖??몄감: {te.std():>15.4f}")
            print(f"理쒖냼媛? {te.min():>16.4f}")
            print(f"理쒕?媛? {te.max():>16.4f}")
            print(f"以묒쐞?? {te.median():>16.4f}")
        
        # 4. TFP 遺꾪빐 寃곌낵 (?듭떖 4媛吏 援ъ꽦?붿씤)
        display_cols = [self.id_var, self.time_var]
        
        # TFP ?듭떖 援ъ꽦?붿씤?ㅻ쭔 異붽?
        tfp_core_components = ['tfp_growth', 'tech_change_effect', 'tech_efficiency_change', 'scale_effect']
        
        for comp in tfp_core_components:
            if comp in self.normalized_data.columns:
                display_cols.append(comp)
        
        # 寃곗륫移??쒓굅
        cost_display = self.normalized_data[display_cols].dropna()
        
        if len(cost_display) > 0:
            print(f"\n4. TFP 遺꾪빐 寃곌낵 (?듭떖 4媛吏 援ъ꽦?붿씤)")
            print("=" * 80)
            
            # 而щ읆紐?蹂寃?(媛?낆꽦)
            col_rename = {
                'tfp_growth': 'TFP利앷???,
                'tech_change_effect': '湲곗닠蹂??,
                'tech_efficiency_change': '湲곗닠?곹슚?⑥꽦蹂??,
                'scale_effect': '洹쒕え?섍꼍?쒗슚怨?
            }
            
            cost_display_renamed = cost_display.rename(columns=col_rename)
            
            # ?꾩껜 ?곗씠??異쒕젰
            print("TFP 遺꾪빐 寃곌낵:")
            print(cost_display_renamed.round(6).to_string(index=False))
            print(f"\n珥?{len(cost_display_renamed)}媛?愿痢≪튂")
            
            # TFP 遺꾪빐 援ъ꽦?붿씤 ?듦퀎
            tfp_stats_cols = ['TFP利앷???, '湲곗닠蹂??, '湲곗닠?곹슚?⑥꽦蹂??, '洹쒕え?섍꼍?쒗슚怨?]
            available_tfp_cols = [col for col in tfp_stats_cols if col in cost_display_renamed.columns]
            
            if available_tfp_cols:
                print("\n" + "-" * 70)
                print("TFP 援ъ꽦?붿씤 ?듦퀎 (?곌컙 蹂?붿쑉, %)")
                print("-" * 70)
                print(f"{'援ъ꽦?붿씤':<20} {'?됯퇏':<12} {'?쒖??몄감':<12} {'理쒖냼媛?:<12} {'理쒕?媛?:<12}")
                print("-" * 70)
                
                for col in available_tfp_cols:
                    if col in cost_display_renamed.columns:
                        col_data = cost_display_renamed[col].dropna() * 100  # 諛깅텇??
                        if len(col_data) > 0:
                            print(f"{col:<20} {col_data.mean():>8.4f} {col_data.std():>12.4f} {col_data.min():>12.4f} {col_data.max():>12.4f}")
                
                print("-" * 70)
                
                # TFP 遺꾪빐 寃利?
                if all(col in cost_display_renamed.columns for col in ['TFP利앷???, '湲곗닠蹂??, '湲곗닠?곹슚?⑥꽦蹂??, '洹쒕え?섍꼍?쒗슚怨?]):
                    calculated_tfp = (cost_display_renamed['湲곗닠蹂??] + 
                                    cost_display_renamed['湲곗닠?곹슚?⑥꽦蹂??] + 
                                    cost_display_renamed['洹쒕え?섍꼍?쒗슚怨?])
                    decomposition_error = cost_display_renamed['TFP利앷???] - calculated_tfp
                    
                    print(f"\n?? TFP 遺꾪빐 ?뺥솗??")
                    print(f"   ?됯퇏 遺꾪빐?ㅼ감: {decomposition_error.mean()*100:>8.6f}%")
                    print(f"   理쒕? ?덈??ㅼ감: {decomposition_error.abs().max()*100:>8.6f}%")
                    
                    if decomposition_error.abs().max() < 0.05:  # 5% ?댄븯
                        print("   ? 遺꾪빐媛 ?뺥솗?⑸땲??)
                    elif decomposition_error.abs().max() < 0.10:  # 10% ?댄븯
                        print("   ??  遺꾪빐 ?ㅼ감媛 ?ㅼ냼 ?덉뒿?덈떎")
                    else:
                        print("   ? 遺꾪빐 ?ㅼ감媛 ?쎈땲??)
            
            # ?쒓컙蹂??됯퇏
            if self.include_time and len(cost_display) > 0:
                print(f"\n?쒓컙蹂??됯퇏:")
                time_avg = cost_display.groupby(self.time_var)[tfp_core_components].mean()
                time_avg_renamed = time_avg.rename(columns=col_rename)
                print(time_avg_renamed.round(6).to_string())
            
            # ID蹂??됯퇏 (媛쒖껜蹂?鍮꾧탳)
            print(f"\nID蹂??됯퇏 (媛쒖껜蹂?鍮꾧탳):")
            print("-" * 60)
            id_avg = cost_display.groupby(self.id_var)[tfp_core_components].mean()
            id_avg_renamed = id_avg.rename(columns=col_rename)
            print(id_avg_renamed.round(6).to_string())
            
            # ?뚯씪 ???
            print(f"\n?? TFP 遺꾪빐 遺꾩꽍 ?곗씠????? {save_path}")
            
            # ?붾젆?좊━媛 ?놁쑝硫??앹꽦
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            cost_display_renamed.to_csv(save_path, index=False, encoding='utf-8-sig')
            print("????꾨즺!")
        
        else:
            print("TFP 遺꾪빐 遺꾩꽍 ?곗씠?곌? ?놁뒿?덈떎")
        
        print("\n" + "=" * 100)
    
    def plot_results(self):
        """寃곌낵 ?쒓컖??""
        if 'technical_efficiency' not in self.normalized_data.columns:
            print("湲곗닠???⑥쑉??TE)??怨꾩궛?섏? ?딆븯?듬땲??")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 湲곗닠???⑥쑉??遺꾪룷
        axes[0, 0].hist(self.normalized_data['technical_efficiency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('湲곗닠???⑥쑉??TE) 遺꾪룷')
        axes[0, 0].set_xlabel('湲곗닠???⑥쑉??)
        axes[0, 0].set_ylabel('鍮덈룄')
        
        # TFP 利앷????쒓퀎??
        if self.include_time and 'tfp_growth' in self.normalized_data.columns:
            tfp_by_time = self.normalized_data.groupby(self.time_var)['tfp_growth'].mean()
            axes[0, 1].plot(tfp_by_time.index, tfp_by_time.values * 100, marker='o')
            axes[0, 1].set_title('?쒓컙蹂??됯퇏 TFP 利앷???)
            axes[0, 1].set_xlabel('?쒓컙')
            axes[0, 1].set_ylabel('TFP 利앷???(%)')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # TFP 援ъ꽦?붿씤蹂??쒓퀎??
        if self.include_time and all(col in self.normalized_data.columns for col in ['tech_change_effect', 'tech_efficiency_change', 'scale_effect']):
            components = ['tech_change_effect', 'tech_efficiency_change', 'scale_effect']
            component_names = ['湲곗닠蹂??, '湲곗닠?곹슚?⑥꽦蹂??, '洹쒕え?섍꼍?쒗슚怨?]
            
            for comp, name in zip(components, component_names):
                comp_by_time = self.normalized_data.groupby(self.time_var)[comp].mean()
                axes[1, 0].plot(comp_by_time.index, comp_by_time.values * 100, marker='o', label=name)
            
            axes[1, 0].set_title('TFP 援ъ꽦?붿씤蹂??쒓퀎??)
            axes[1, 0].set_xlabel('?쒓컙')
            axes[1, 0].set_ylabel('蹂?붿쑉 (%)')
            axes[1, 0].legend()
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 鍮꾩슜紐?遺꾪룷
        share_vars = [f'share_{input_var}' for input_var in self.input_vars if f'share_{input_var}' in self.normalized_data.columns]
        if share_vars:
            share_data = self.normalized_data[share_vars].mean()
            axes[1, 1].bar(range(len(share_data)), share_data.values)
            axes[1, 1].set_title('?됯퇏 鍮꾩슜紐?)
            axes[1, 1].set_xlabel('?앹궛?붿냼')
            axes[1, 1].set_ylabel('鍮꾩슜紐?)
            axes[1, 1].set_xticks(range(len(share_data)))
            axes[1, 1].set_xticklabels([var.replace('share_', '').upper() for var in share_vars])
        
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self, save_path='cost_tfp_results.csv'):
        """?꾩껜 TFP 遺꾪빐 遺꾩꽍 ?ㅽ뻾"""
        print("?뺣쪧蹂寃쎈퉬?⑺븿??諛?TFP 遺꾪빐 遺꾩꽍???쒖옉?⑸땲??..")
        
        # 1. EDA
        self.exploratory_data_analysis()
        
        # 2. ?곗씠??以鍮?
        self.normalize_data()
        self.create_translog_variables()
        
        # 3. 異붿젙
        self.estimate_ols()
        self.estimate_stochastic_frontier()
        
        # 4. TFP 諛?援ъ꽦?붿씤 遺꾩꽍
        self.calculate_efficiency()
        self.calculate_cost_economics()
        
        # 5. 寃곌낵 異쒕젰
        self.print_results(save_path)
        self.plot_results()
        
        print("\nTFP 遺꾪빐 遺꾩꽍???꾨즺?섏뿀?듬땲??")
        
        return self.results, self.normalized_data


def Run_StochasticFrontierCost(data, cost_var, output_var, price_vars, input_vars, 
                          time_var='year', id_var='id', include_time=True, save_path='cost_results.csv'):
    """
    ?뺣쪧蹂寃쎈퉬?⑺븿?섎? ?댁슜??TFP 遺꾪빐 遺꾩꽍 - ?ъ슜??移쒗솕???명꽣?섏씠??
    """
    
    print("?? ?뺣쪧蹂寃쎈퉬?⑺븿?섎? ?댁슜??TFP 遺꾪빐 遺꾩꽍???쒖옉?⑸땲??..")
    print("=" * 80)
    
    try:
        # 遺꾩꽍湲??앹꽦
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
        
        # ?꾩껜 遺꾩꽍 ?ㅽ뻾
        results, processed_data = analyzer.run_complete_analysis(save_path)
        
        print("? 遺꾩꽍???꾨즺?섏뿀?듬땲??")
        
        return results, processed_data
        
    except Exception as e:
        print(f"? 遺꾩꽍 以??ㅻ쪟 諛쒖깮: {str(e)}")
        print("?곸꽭 ?ㅻ쪟:")
        import traceback
        traceback.print_exc()
        return None, None
