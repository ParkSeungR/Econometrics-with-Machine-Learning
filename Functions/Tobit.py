# Tobit Function
# Source: Florian Heiss and Daniel Brunner p.280
class Tobit(smclass.GenericLikelihoodModel):
    def nloglikeobs(self, params):
        X = self.exog
        y = self.endog
        p = X.shape[1]
        beta = params[0:p]
        sigma = np.exp(params[p])
        y_hat = np.dot(X, beta)
        y_eq = (y == 0)
        y_g = (y > 0)
        ll = np.empty(len(y))
        ll[y_eq] = np.log(stats.norm.cdf(-y_hat[y_eq] / sigma))
        ll[y_g] = np.log(stats.norm.pdf((y - y_hat)[y_g] / sigma)) - np.log(sigma)
        return -ll