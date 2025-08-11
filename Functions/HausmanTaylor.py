import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from statsmodels.sandbox.regression.gmm import IV2SLS

# ########## DEfine Function ################
def HausmanTaylor(y, X1, X2, Z1, Z2):
    # Time length 
    T = len(y.index.get_level_values(1).unique())
    
    # 1) Step 1: Fixed effects equation
    X = pd.concat([X1, X2], axis=1)
    XF = sm.add_constant(X)
    
    # PanelOLS requires entity (individual) and time effects to be specified if needed
    Model_FE = PanelOLS(y, XF, entity_effects=True)
    Results_FE = Model_FE.fit()
    
    # Step 2 
    # Residuals
    FE_resid = Results_FE.resids
    sig1 = Results_FE.s2

    
    ZZ1 = sm.add_constant(Z1)
    Z = pd.concat([ZZ1, Z2], axis=1)
    INST = pd.concat([X1, ZZ1], axis=1)

    # IV2SLS requires dependent variable, exogenous regressors, endogenous regressors, and instruments
    Results_IV = IV2SLS(FE_resid, Z, INST).fit()
    sig2 = Results_IV.scale
    
    # 3) Step 3
    ahat = 1 - np.sqrt(sig1/(T * sig2 + sig1))
    df = pd.concat([y, X1, X2, Z1, Z2], axis=1)
    
     # 1)  _y 
    df_m = df.groupby('id').mean()
    df_m = pd.merge(df, df_m, on = 'id', how = 'left')
    
    # 2) (xy - xy_bar): _d 
    for name in df.columns:
        df_m[name +'_d'] = df_m[name + '_x'] - df_m[name + '_y']
    
    # 3) (xy - ahat * xy_bar): _s
    for name in df.columns:                          
        df_m[name +'_s'] = df_m[name + '_x'] - ahat * df_m[name + '_y']
     
    # (_s)
    # define dep, indep var.
    df_s = df_m[[var for var in df_m if var.endswith('_s')]]
    y_s = df_s.iloc[:,0]
    X_s = df_s.iloc[:,1:]
    X_s = sm.add_constant(X_s)
    
    # instrumental var (_d, _y, Z1)
    ZZ_d = df_m[[var for var in df_m if var.endswith('_d')]]
    ZZ_d = ZZ_d.iloc[:,1:]
    ZZ_dd= ZZ_d.reset_index()
    ZZ_dd = ZZ_dd.iloc[:,1:]
          
    ZZ_y = df_m[[var for var in df_m if var.endswith('_y')]]
    ZZ_y = ZZ_y.iloc[:,1:]
    ZZ_yy = ZZ_y.reset_index()
    ZZ_yy = ZZ_yy.iloc[:,1:]
    
    Z1_z = Z1.reset_index()
    Z1_z = Z1_z.iloc[:,2:]

    ZZ_ss = pd.concat([ZZ_dd, ZZ_yy, Z1_z], join='inner', axis=1)
    variances = ZZ_ss.var()
    columns_to_keep = variances[variances != 0].index
    INST = ZZ_ss[columns_to_keep]
    INST = sm.add_constant(INST)
    
    # IV2SLS requires dependent variable, exogenous regressors, endogenous regressors, and instruments
    Results_OLS = sm.OLS(y_s, X_s).fit()
    Results_HT = IV2SLS(y_s, X_s, INST).fit()
    print(Results_HT.summary())
    return []

