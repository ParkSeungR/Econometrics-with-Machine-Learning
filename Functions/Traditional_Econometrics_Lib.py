# Libraries for the Analysis of Traditional Econometrics
# Call this file "exec(open('Functions/Traditional_Econometrics_Lib').read()"
import arch
import os
import numpy as np                                       # Numerical calculations
import pandas as pd                                      # Data handling
import math as someAlias
import matplotlib.dates as mdates                        # Turn dates into numbers
import matplotlib.pyplot as plt                          # Lower-level graphics
import patsy as pt
import seaborn as sns
import stargazer.stargazer as sg
import statsmodels.api as sm
import statsmodels.base.model as smclass
import statsmodels.formula.api as smf                    # Econometrics
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as dg
import statsmodels.stats.outliers_influence as smo
import linearmodels as lm                                # Panel model, Simultaneous Eq. Model
import linearmodels.iv as iv
import linearmodels.system as iv3
import scipy.stats as stats                              # Statistics
import random

from pandas import set_option                      
from scipy.optimize import Bounds
from scipy.optimize import curve_fit                    # Nonlinear regression
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics import tsaplots               # Time series
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.eval_measures import mse, rmse, meanabs, aic, bic
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima_model import ARMA, ARMAResults, ARIMA, ARIMAResults
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.stattools import acovf, acf, pacf, pacf_yw, pacf_ols
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.stattools import ccovf, ccf
from statsmodels.tsa.stattools import adfuller, kpss, coint, bds, q_stat, grangercausalitytests, levinson_durbin

from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.vector_ar.svar_model import SVAR
from statsmodels.tsa.vector_ar.vecm import select_coint_rank  
from statsmodels.tsa.vector_ar.vecm import select_order 
from statsmodels.tsa.vector_ar.vecm import VECM

from linearmodels import BetweenOLS
from linearmodels.panel.results import compare
from linearmodels import FirstDifferenceOLS
from linearmodels import PanelOLS
from linearmodels import PooledOLS
from linearmodels import RandomEffects
from linearmodels import IV2SLS, IV3SLS, SUR, IVSystemGMM

from arch import arch_model
from arch.unitroot import ADF, DFGLS, PhillipsPerron, KPSS
from arch.unitroot import engle_granger

from pmdarima import auto_arima

import wooldridge as woo
#from imfpy.retrievals import dots
#from imfpy.tools import dotsplot
#import wbdata
#import wbgapi as wb            
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

# Korean Fonts
import matplotlib as mpl
mpl.rc('font', family='NanumGothic')
mpl.rc('axes', unicode_minus=False)