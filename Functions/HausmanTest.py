# ########## DEfine Function ################
def HausmanTest(Results_FE, Results_RE):
    
    # (1) 
    C_coef = list(set(Results_FE.params.index).intersection(Results_RE.params.index))
    # (2) FE and RE
    D_b = np.array(Results_FE.params[C_coef] - Results_RE.params[C_coef])
    dof = len(D_b)
    D_b.reshape((dof, 1))
    # (3) FE and RE
    D_cov = np.array(Results_FE.cov.loc[C_coef, C_coef] -
                     Results_RE.cov.loc[C_coef, C_coef])
    D_cov.reshape((dof, dof))
    # (4) Chi2 
    Chi2 = abs(np.transpose(D_b) @ np.linalg.inv(D_cov) @ D_b)
    pvalue = 1 - stats.chi2.cdf(Chi2, dof)

    return [Chi2, dof, pvalue]

