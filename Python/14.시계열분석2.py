#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # ARMA(p,q) and ARIMA(p,d,q)
# # Autoregressive Moving Averages
# This section covers <em>Autoregressive Moving Averages</em> (ARMA) and <em>Autoregressive Integrated Moving Averages</em> (ARIMA).
# 
# Recall that an <strong>AR(1)</strong> model follows the formula
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = c + \phi_{1}y_{t-1} + \varepsilon_{t}$
# 
# while an <strong>MA(1)</strong> model follows the formula
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = \mu + \theta_{1}\varepsilon_{t-1} + \varepsilon_{t}$
# 
# where $c$ is a constant, $\mu$ is the expectation of $y_{t}$ (often assumed to be zero), $\phi_1$ (phi-sub-one) is the AR lag coefficient, $\theta_1$ (theta-sub-one) is the MA lag coefficient, and $\varepsilon$ (epsilon) is white noise.
# 
# An <strong>ARMA(1,1)</strong> model therefore follows
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y_{t} = c + \phi_{1}y_{t-1} + \theta_{1}\varepsilon_{t-1} + \varepsilon_{t}$
# 
# ARMA models can be used on stationary datasets.
# 
# For non-stationary datasets with a trend component, ARIMA models apply a differencing coefficient as well.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARMA.html'>arima_model.ARMA</a></strong><font color=black>(endog, order[, exog, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Autoregressive Moving Average ARMA(p,q) model<br>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARMAResults.html'>arima_model.ARMAResults</a></strong><font color=black>(model, params[, …])</font>&nbsp;&nbsp;&nbsp;Class to hold results from fitting an ARMA model<br>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html'>arima_model.ARIMA</a></strong><font color=black>(endog, order[, exog, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;Autoregressive Integrated Moving Average ARIMA(p,d,q) model<br>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.html'>arima_model.ARIMAResults</a></strong><font color=black>(model, params[, …])</font>&nbsp;&nbsp;Class to hold results from fitting an ARIMA model<br>	
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.kalmanf.kalmanfilter.KalmanFilter.html'>kalmanf.kalmanfilter.KalmanFilter</a></strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Kalman Filter code intended for use with the ARMA model</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model'>Wikipedia</a></strong>&nbsp;&nbsp;<font color=black>Autoregressive–moving-average model</font><br>
# <strong>
# <a href='https://otexts.com/fpp2/non-seasonal-arima.html'>Forecasting: Principles and Practice</a></strong>&nbsp;&nbsp;<font color=black>Non-seasonal ARIMA models</font></div>

# ## Perform standard imports and load datasets

# In[1]:


# 시계열 분석을 위한 라이브러리 불러오기
exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[5]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[6]:


data['Gr_gdp'] = 100* data['gdp'].pct_change(periods=4)
#data['Gr_gdp'] = 100* np.log(data['gdp']).diff()
df = data.dropna()
df


# In[ ]:





# ## Automate the augmented Dickey-Fuller Test
# Since we'll be using it a lot to determine if an incoming time series is stationary, let's write a function that performs the augmented Dickey-Fuller Test.

# In[11]:


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# ___
# ## Autoregressive Moving Average - ARMA(p,q)
# In this first section we'll look at a stationary dataset, determine (p,q) orders, and run a forecasting ARMA model fit to the data. In practice it's rare to find stationary data with no trend or seasonal component, but the first four months of the <em>Daily Total Female Births</em> dataset should work for our purposes.
# ### Plot the source data

# In[57]:


df['Gr_gdp'].plot(figsize=(12,5));


# ### Run the augmented Dickey-Fuller Test to confirm stationarity

# In[58]:


adf_test(df['Gr_gdp'])


# In[69]:


result = seasonal_decompose(df['Gr_gdp'], model='additive')  
result.plot();


# ### Determine the (p,q) ARMA Orders using <tt>pmdarima.auto_arima</tt>
# This tool should give just $p$ and $q$ value recommendations for this dataset.

# In[59]:


auto_arima(df['Gr_gdp'],seasonal=False).summary()


# ### Split the data into train/test sets
# As a general rule you should set the length of your test set equal to your intended forecast size. For this dataset we'll attempt a 1-month forecast.

# In[73]:


# Set one month for testing
train = df.loc[:'2015-12-31']
test = df.loc['2016-03-31':]


# In[74]:


train


# In[75]:


test


# ### Fit an ARMA(p,q) Model
# If you want you can run <tt>help(ARMA)</tt> to learn what incoming arguments are available/expected, and what's being returned.

# In[76]:


model = ARIMA(train['Gr_gdp'],order=(4,1,2))
results = model.fit()
results.summary()


# ### Obtain a month's worth of predicted values

# In[79]:


start='2016-3-31'
end='2023-12-31'
predictions = results.predict(start=start, end=end).rename('ARIMA(4,1,2) Predictions')
predictions


# ### Plot predictions against known values

# In[80]:


title = '한국 GDP 증가율'
ylabel='증가율'
xlabel='연/분기'
ax = test['Gr_gdp'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# In[ ]:


# 예측치를 원래 레벨변수로 만드는 법? 이하설명


# Since our starting dataset exhibited no trend or seasonal component, this prediction makes sense. In the next section we'll take additional steps to evaluate the performance of our predictions, and forecast into the future.

# ___
# ## Autoregressive Integrated Moving Average - ARIMA(p,d,q)
# The steps are the same as for ARMA(p,q), except that we'll apply a differencing component to make the dataset stationary.<br>
# First let's take a look at the <em>Real Manufacturing and Trade Inventories</em> dataset.
# ### Plot the Source Data

# In[82]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index

ax = data['con'].plot(figsize=(12,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# ### Run an ETS Decomposition (optional)
# We probably won't learn a lot from it, but it never hurts to run an ETS Decomposition plot.

# In[83]:


result = seasonal_decompose(data['con'], model='mul')
result.plot();


# Here we see that the seasonal component does not contribute significantly to the behavior of the series.
# ### Use <tt>pmdarima.auto_arima</tt> to determine ARIMA Orders

# In[8]:


auto_arima(data['gdp'],seasonal=False).summary()


# This suggests that we should fit an ARIMA(1,1,1) model to best forecast future values of the series. Before we train the model, let's look at augmented Dickey-Fuller Test, and the ACF/PACF plots to see if they agree. These steps are optional, and we would likely skip them in practice.

# ### Run the augmented Dickey-Fuller Test on the First Difference

# In[9]:


from statsmodels.tsa.statespace.tools import diff
data['D_con'] = diff(data['con'], k_diff=1)
data


# In[12]:


adf_test(data['D_con'].dropna(),'Consumption')


# This confirms that we reached stationarity after the first difference.
# ### Run the ACF and PACF plots
# A <strong>PACF Plot</strong> can reveal recommended AR(p) orders, and an <strong>ACF Plot</strong> can do the same for MA(q) orders.<br>
# Alternatively, we can compare the stepwise <a href='https://en.wikipedia.org/wiki/Akaike_information_criterion'>Akaike Information Criterion (AIC)</a> values across a set of different (p,q) combinations to choose the best combination.

# In[6]:


# GDP의 ACF(Autocorrelation Function)와 PACF(Partial Autocorrelation Function)
fig, ax = plt.subplots(2,1,  figsize = (12,8))
plot_acf(data['gdp'], lags=40, ax=ax[0], zero=False)
plot_pacf(data['gdp'], lags=40, ax=ax[1], zero=False)  
fig.tight_layout()
plt.show()


# In[7]:


# Autocorrelation: CON
fig, ax = plt.subplots(2,1,  figsize = (12,8))
plot_acf(data['con'], lags=40, ax=ax[0], zero=False)
plot_pacf(data['con'], lags=40, ax=ax[1], zero=False)  
fig.tight_layout()
plt.show()


# In[14]:


# Autocorrelation: CON
fig, ax = plt.subplots(2,1,  figsize = (12,8))
plot_acf(data['D_con'].dropna(), lags=40, ax=ax[0], zero=False)
plot_pacf(data['D_con'].dropna(), lags=40, ax=ax[1], zero=False)  
fig.tight_layout()
plt.show()


# This tells us that the AR component should be more important than MA. From the <a href='https://people.duke.edu/~rnau/411arim3.htm'>Duke University Statistical Forecasting site</a>:<br>
# > <em>If the PACF displays a sharp cutoff while the ACF decays more slowly (i.e., has significant spikes at higher lags), we    say that the stationarized series displays an "AR signature," meaning that the autocorrelation pattern can be explained more    easily by adding AR terms than by adding MA terms.</em><br>
# 
# Let's take a look at <tt>pmdarima.auto_arima</tt> done stepwise to see if having $p$ and $q$ terms the same still makes sense:

# In[4]:


stepwise_fit = auto_arima(data['gdp'].dropna(), start_p=0, start_q=0,
                          max_p=8, max_q=4, m=4,
                          seasonal=False,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# Looks good from here! Now let's train & test the ARIMA(1,1,1) model, evaluate it, then produce a forecast of future values.
# ### Split the data into train/test sets

# In[6]:


train = data.loc[:'2015-12-31']
test = data.loc['2016-03-31':]


# ### Fit an ARIMA(1,1,1) Model

# In[7]:


model = ARIMA(train['gdp'],order=(5,2,3))
results = model.fit()
results.summary()


# In[8]:


start='2016-3-31'
end='2023-12-31'
predictions = results.predict(start=start, end=end).rename('ARIMA(5,2,3) Predictions')
predictions


# Passing <tt>dynamic=False</tt> means that forecasts at each point are generated using the full history up to that point (all lagged values).
# 
# Passing <tt>typ='levels'</tt> predicts the levels of the original endogenous variables. If we'd used the default <tt>typ='linear'</tt> we would have seen linear predictions in terms of the differenced endogenous variables.
# 
# For more information on these arguments visit https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html

# In[10]:


# 사후예측과 실적치 
title = 'Korea"s GDP'
ylabel='level of GDP'
xlabel='연/분기'

ax = test['gdp'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# ### Evaluate the Model

# In[103]:


mse = mean_squared_error(test['con'], predictions)
rmse = rmse(test['con'], predictions)
print(mse, rmse)


# ### Retrain the model on the full data, and forecast the future

# In[16]:


model = ARIMA(data['gdp'],order=(5,2,3))
results = model.fit()
fcast = results.predict('2024-3-31','2030-12-31',typ='levels').rename('ARIMA(5,2,3) Forecast')


# In[17]:


# 사전예측 그리프
title = 'Korea"s GDP'
ylabel='Level'
xlabel='연/분기' 

ax = data['gdp'].loc['2000-03-31':].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# In[ ]:





# # SARIMA(p,d,q)(P,D,Q)m
# # Seasonal Autoregressive Integrated Moving Averages
# We have finally reached one of the most fascinating aspects of time series analysis: seasonality.
# 
# Where ARIMA accepts the parameters $(p,d,q)$, SARIMA accepts an <em>additional</em> set of parameters $(P,D,Q)m$ that specifically describe the seasonal components of the model. Here $P$, $D$ and $Q$ represent the seasonal regression, differencing and moving average coefficients, and $m$ represents the number of data points (rows) in each seasonal cycle.
# 
# <strong>NOTE:</strong> The statsmodels implementation of SARIMA is called SARIMAX. The “X” added to the name means that the function also supports <em>exogenous</em> regressor variables. We'll cover these in the next section.
# 
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html'>sarimax.SARIMAX</a></strong><font color=black>(endog[, exog, order, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html'>sarimax.SARIMAXResults</a></strong><font color=black>(model, params, …[, …])</font>&nbsp;&nbsp;Class to hold results from fitting a SARIMAX model.</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://www.statsmodels.org/stable/statespace.html'>Statsmodels Tutorial:</a></strong>&nbsp;&nbsp;<font color=black>Time Series Analysis by State Space Methods</font></div>

# ## Perform standard imports and load datasets

# In[3]:


# 시계열 분석을 위한 라이브러리 불러오기
exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[4]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# ### Inspect the data, create a DatetimeIndex

# In[21]:


data


# We need to combine two integer columns (year and month) into a DatetimeIndex. We can do this by passing a dictionary into <tt>pandas.to_datetime()</tt> with year, month and day values.<br>
# For more information visit https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

# ### Plot the source data

# In[22]:


title = 'Korean GDP'
ylabel='Level'
xlabel='연/분기' 
ax = data['gdp'].plot(figsize=(12,6),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# ### Run an ETS Decomposition

# In[23]:


result = seasonal_decompose(data['gdp'], model='add')
result.plot();


# In[24]:


result = seasonal_decompose(data['gdp'], model='mul')
result.plot();


# Although small in scale compared to the overall values, there is a definite annual seasonality.

# ### Run <tt>pmdarima.auto_arima</tt> to obtain recommended orders
# This may take awhile as there are a lot more combinations to evaluate.

# In[5]:


# 계절적 요인을 무시한 SARIMA(ARIMA) 
results = auto_arima(data['gdp'], seasonal=False, m=4)
print(results.summary())


# In[7]:


# 계절적 요인을 고려한 SARIMA
results = auto_arima(data['gdp'], seasonal=True, m=4)
print(results.summary())


# Excellent! This provides an ARIMA Order of (0,1,3) combined with a seasonal order of (1,0,1,12) Now let's train & test the SARIMA(0,1,3)(1,0,1,12) model, evaluate it, then produce a forecast of future values.
# ### Split the data into train/test sets

# In[26]:


# Set one month for testing
train = data.loc[:'2015-12-31']
test = data.loc['2016-03-31':]


# ### Fit a SARIMA(0,1,3)(1,0,1,12) Model

# In[27]:


model = SARIMAX(data['gdp'],order=(0,1,0),seasonal_order=(1, 1, [1, 2], 4))
results = model.fit()
results.summary()


# In[28]:


start='2016-3-31'
end='2023-12-31'
predictions = results.get_prediction(start=start, end=end)
pred_point = results.predict(start=start, end=end)
pred_ci = predictions.conf_int()
pred_ci['point'] =pred_point
pred_ci['Actual'] =data['gdp']
print(pred_ci)


# Passing <tt>dynamic=False</tt> means that forecasts at each point are generated using the full history up to that point (all lagged values).
# 
# Passing <tt>typ='levels'</tt> predicts the levels of the original endogenous variables. If we'd used the default <tt>typ='linear'</tt> we would have seen linear predictions in terms of the differenced endogenous variables.
# 
# For more information on these arguments visit https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html

# In[29]:


# 사후예측 그래프 그리기
title = 'Korean GDP'
ylabel='Level'
xlabel='연/분기'
ax = pred_ci.plot(figsize=(12,6),legend=True,title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel); 


# ### Evaluate the Model

# In[30]:


mse = mse(test['gdp'], pred_ci['point'])
rmse = rmse(test['gdp'], pred_ci['point'])
print(mse, rmse)


# These are outstanding results!
# ### Retrain the model on the full data, and forecast the future

# In[31]:


model = SARIMAX(data['gdp'],order=(0,1,0),seasonal_order=(1,1,[1,2],4))
results = model.fit()
predictions = results.get_prediction('2023-12-31','2030-12-31',typ='levels')
pred_point = results.predict('2023-12-31','2030-12-31')
pred_ci = predictions.conf_int()
pred_ci['point'] =pred_point
pred_ci


# In[33]:


# 샘플범위 밖 예측 및 예측오차의 그래프
title = 'Korean GDP'
ylabel='Level'
xlabel='연/분기'
fig, ax = plt.subplots()
ax = data['gdp'].loc['2010-03-31':].plot(legend=True,figsize=(12,6),title=title)
ax.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], color='grey' , alpha=0.2)
pred_ci.plot(legend=True, ax=ax)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);
plt.show()


# In[170]:


# 코스피 예측(빈날짜가 있기 때문에 datetime설정 불가, 계절분해 불가, 방법확인 필요), 월별, 분기별은 가능할 듯)
# 비트코인은 빈날짜 없이 데이터가 있기 때문에 가능하지만 계절적 요인 파악이 힘든지 관련 파라미터, 예측곤란)
start = '2016-01-01' 
end = '2024-02-21'
assets = ['BTC-USD'] 
#assets = ['KS11','^GSPC', 'BTC-USD'] 
# Downloading price data
data = yf.download(assets, start=start, end=end) 
data = data.loc[:, ('Adj Close')] 
data.columns = assets
df = pd.DataFrame(data)

df.columns = ["pric"]
display(df)

df.index = pd.to_datetime(new_index)
df.index


# In[171]:


#title = 'Korean GDP'
ylabel='Level'
xlabel='연/분기' 
ax = df['pric'].plot(figsize=(12,6),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# In[172]:


result = seasonal_decompose(df['pric'], model='add')
result.plot();


# In[173]:


# For SARIMA Orders we set seasonal=True and pass in an m value
auto_arima(df['pric'],seasonal=True).summary()


# In[174]:


model = SARIMAX(df['pric'],order=(3, 1, 2))
results = model.fit()
results.summary()


# In[175]:


model = SARIMAX(df['pric'],order=(3,1,2))
results = model.fit()
fcast = results.predict('2024-02-21','2024-12-29',typ='levels').rename('SARIMA(3,1,2) Forecast')


# In[177]:


# Plot predictions against known values
title = 'BTC'
ylabel='Level'
xlabel='연/분기/일'

ax = df['pric'].loc['2020-01-01':].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# In[ ]:





# # SARIMAX
# 
# ## Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
# So far the models we've looked at consider past values of a dataset and past errors to determine future trends, seasonality and forecasted values. We look now to models that encompass these non-seasonal (p,d,q) and seasonal (P,D,Q,m) factors, but introduce the idea that external factors (environmental, economic, etc.) can also influence a time series, and be used in forecasting.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html'>sarimax.SARIMAX</a></strong><font color=black>(endog[, exog, order, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VARResults.html'>sarimax.SARIMAXResults</a></strong><font color=black>(model, params, …[, …])</font>&nbsp;&nbsp;Class to hold results from fitting a SARIMAX model.</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://www.statsmodels.org/stable/statespace.html'>Statsmodels Tutorial:</a></strong>&nbsp;&nbsp;<font color=black>Time Series Analysis by State Space Methods</font><br>
# <strong>
# <a href='https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_sarimax_stata.html'>Statsmodels Example:</a></strong>&nbsp;&nbsp;<font color=black>SARIMAX</font></div>

# ## Perform standard imports and load datasets

# In[2]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
df = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(df), freq='Q')
df.index = pd.to_datetime(new_index)
df.index


# ### Inspect the data
# For this section we've built a Restaurant Visitors dataset that was inspired by a <a href='https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting'>recent Kaggle competition</a>. The data considers daily visitors to four restaurants located in the United States, subject to American holidays. For the exogenous variable we'll see how holidays affect patronage. The dataset contains 478 days of restaurant data, plus an additional 39 days of holiday data for forecasting purposes.

# In[3]:


df


# Notice that even though the restaurant visitor columns contain integer data, they appear as floats. This is because the bottom of the dataframe has 39 rows of NaN data to accommodate the extra holiday data we'll use for forecasting, and pandas won't allow NaN's as integers. We could leave it like this, but since we have to drop NaN values anyway, let's also convert the columns to dtype int64.

# ### Plot the source data

# In[4]:


title='Consumption function'
ylabel='Level'
xlabel='연/분기'

ax = df[['gdp', 'con']].plot(figsize=(16,5),title=title)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);


# ## Look at holidays
# Rather than prepare a separate plot, we can use matplotlib to shade holidays behind our restaurant data.

# ### Run an ETS Decomposition

# In[5]:


result = seasonal_decompose(df['gdp'])
result.plot();


# In[6]:


result = seasonal_decompose(df['con'])
result.plot();


# ## Test for stationarity

# In[189]:


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[190]:


adf_test(df['gdp'])


# In[191]:


adf_test(df['con'])


# ### Run <tt>pmdarima.auto_arima</tt> to obtain recommended orders
# This may take awhile as there are a lot of combinations to evaluate.

# In[7]:


# For SARIMA Orders we set seasonal=True and pass in an m value
auto_arima(df['con'],seasonal=True,m=4).summary()


# Excellent! This provides an ARIMA Order of (1,0,0) and a seasonal order of (2,0,0,7) Now let's train & test the SARIMA model, evaluate it, then compare the result to a model that uses an exogenous variable.
# ### Split the data into train/test sets
# We'll assign 42 days (6 weeks) to the test set so that it includes several holidays.

# In[9]:


# Set one month for testing
train = df.loc[:'2015-12-31']
test = df.loc['2016-03-31':]


# ## Now add the exog variable

# In[10]:


model = SARIMAX(train['con'],exog=train['gdp'],order=(3,2,0),seasonal_order=(0,0,1,4),enforce_invertibility=False)
results = model.fit()
results.summary()


# In[11]:


start='2016-3-31'
end='2023-12-31'


# In[14]:


predictions = results.predict(start=start, end=end, exog=test[['gdp']])


# In[203]:


# Plot predictions against known values
title = 'Consumption Function'
ylabel='Level'
xlabel='연/분기'

ax = test['con'].plot(legend=True,figsize=(12,6),title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)


# We can see that the exogenous variable (holidays) had a positive impact on the forecast by raising predicted values at 3/17, 4/14, 4/16 and 4/17! Let's compare evaluations:
# ### Evaluate the Model

# In[209]:


#from statsmodels.tools.eval_measures import mse, rmse, meanabs
mse = mse(test['con'], predictions)
rmse = rmse(test['con'], predictions)
print(mse, rmse)


# ### Retrain the model on the full data, and forecast the future
# We're going to forecast 39 days into the future, and use the additional holiday data

# In[213]:


model = SARIMAX(df['con'],exog=df['gdp'],order=(3,2,0),seasonal_order=(0,0,1,4),enforce_invertibility=False)
results = model.fit()
exog_forecast = df['2024-01-31':'2024-012-31'][['gdp']]
fcast = results.predict('2024-01-31', '2024-012-31',exog=exog_forecast).rename('SARIMAX(1,0,0)(2,0,0,7) Forecast')
# GDP의 미래값이 있어야... 


# In[24]:


# Plot the forecast alongside historical values
title='Restaurant Visitors'
ylabel='Visitors per day'
xlabel=''

ax = df1['total'].plot(legend=True,figsize=(16,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
for x in df.query('holiday==1').index: 
    ax.axvline(x=x, color='k', alpha = 0.3);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # SVAR

# In[4]:


import numpy as np
import pandas as pd

import statsmodels.datasets.macrodata
from statsmodels.tsa.vector_ar.svar_model import SVAR


# In[5]:


mdatagen = statsmodels.datasets.macrodata.load().data
mdata = mdatagen[['realgdp','realcons','realinv']]
mdata


# In[15]:


#names = mdata.name.dtype


# In[6]:


#qtr = pd.DatetimeIndex(start=start, end=end, freq=pd.datetools.BQuarterEnd())
qtr = pd.date_range('1959-03-31', '2009-09-30', freq='Q')
qtr


# In[7]:


mdata.index = qtr
mdata


# In[8]:


df = (np.log(mdata)).diff().dropna()
df


# In[9]:


df.plot(figsize=(10,4))


# In[10]:


#define structural inputs
A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
A_guess = np.asarray([0.5, 0.25, -0.38])
B_guess = np.asarray([0.5, 0.1, 0.05])
mymodel = SVAR(df, svar_type='AB', A=A, B=B, freq='Q')
res = mymodel.fit(maxlags=3, maxiter=10000, maxfun=10000, solver='bfgs')


# In[11]:


fevd = res.fevd(10)
fevd.summary()
res.fevd(20).plot()


# In[12]:


res.irf(periods=20).plot(impulse='realcons', plot_stderr=True,
                         stderr_type='mc', repl=100);


# In[28]:


res.irf(periods=20).plot(impulse='realgdp', plot_stderr=True,
                         stderr_type='mc', repl=100);


# In[29]:


res.irf(periods=20).plot(impulse='realinv', plot_stderr=True,
                         stderr_type='mc', repl=100);


# In[11]:


import numpy as np
import pandas as pd

import statsmodels.datasets.macrodata
from statsmodels.tsa.vector_ar.svar_model import SVAR

mdatagen = statsmodels.datasets.macrodata.load().data
mdata = mdatagen[['realgdp','realcons','realinv']]
qtr = pd.date_range('1959-03-31', '2009-09-30', freq='Q')
data = pd.DataFrame(mdata, index=qtr)
data = (np.log(data)).diff().dropna()

#define structural inputs
A = np.asarray([[1, 0, 0],['E', 1, 0],['E', 'E', 1]])
B = np.asarray([['E', 0, 0], [0, 'E', 0], [0, 0, 'E']])
A_guess = np.asarray([0.5, 0.25, -0.38])
B_guess = np.asarray([0.5, 0.1, 0.05])
mymodel = SVAR(data, svar_type='AB', A=A, B=B, freq='Q')
res = mymodel.fit(maxlags=3, maxiter=10000, maxfun=10000, solver='bfgs')
res.irf(periods=30).plot(impulse='realgdp', plot_stderr=True,
                         stderr_type='mc', repl=100)


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




