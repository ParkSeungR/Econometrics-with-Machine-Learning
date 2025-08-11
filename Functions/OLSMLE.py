# Calculating the coefficients with ML
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import scipy.stats as stats 

# Creating the OLS Object using Maximum Likelihood 
class OLSMLE(GenericLikelihoodModel): 
    def loglike(self, params): 
        exog = self.exog 
        endog = self.endog 
        k = exog.shape[1]
        resids = endog - np.dot(exog, params[0:k]) 
        sigma = np.std(resids, ddof=0)
        return stats.norm.logpdf(resids, loc=0, scale=sigma).sum()
    