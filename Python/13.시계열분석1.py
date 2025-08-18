#!/usr/bin/env python
# coding: utf-8

# ![%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png](attachment:%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%98%20%EA%B3%84%EB%9F%89%EA%B2%BD%EC%A0%9C%ED%95%99%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%ED%8C%8C%EC%9D%BC%20%ED%91%9C%EC%A7%80%20%EA%B7%B8%EB%A6%BC.png)

# # Introduction to Statsmodels
# 
# Statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration. An extensive list of result statistics are available for each estimator. The results are tested against existing statistical packages to ensure that they are correct. The package is released under the open source Modified BSD (3-clause) license. The online documentation is hosted at <a href='https://www.statsmodels.org/stable/index.html'>statsmodels.org</a>. The statsmodels version used in the development of this course is 0.9.0.
# 
# <div class="alert alert-info"><h3>For Further Reading:</h3>
# <strong>
# <a href='http://www.statsmodels.org/stable/tsa.html'>Statsmodels Tutorial:</a></strong>&nbsp;&nbsp;<font color=black>Time Series Analysis</font></div>
# 
# Let's walk through a very simple example of using statsmodels!

# In[1]:


# 작업공간(working directory)지정  
import os  
os.chdir("E:\JupyterWorkingDirectory\MyStock")
os.getcwd()


# In[2]:


exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# # 다양한 형태의 시계열 자료 생성과 도표그리기

# In[18]:


# 인위적인 AR(1) 시계열 자료 생성
N = 501 
x = range(N)
a = 1
l = 0.1
rho = 0.7

# 확률변수 생성을 위한 seed부여
random.seed(123456)
# 백색잡음(white noise) 오차항 생성
e = stats.norm.rvs(0, 1, N)
y = np.zeros(N)

fig, ax = plt.subplots(3, 2, figsize = [12,10], sharex = True)

# 상수항 0인 안정적 시계열(평균=0)
for t in range(1, N):
    y[t]= rho*y[t-1]+e[t]
ax[0,0].plot(y)
ax[0,0].set_title(r'$y_t = \rho y_{t-1}+\varepsilon_t$', fontsize = 11)
ax[0,0].axhline(0, xmin = 0, xmax = N)

# 상수항 1인 안정적 시계열(평균=1)
for t in range(1, N):
    y[t]= a + rho*y[t-1] + e[t]
ax[0,1].plot(y)
ax[0,1].set_title(r'$y_t = 1+\rho y_{t-1}+\varepsilon_t$', fontsize = 11)
ax[0,1].axhline(a, xmin = 0, xmax = N)

# 상수항, 추세항, rho<1의 시계열
for t in range(1, N):
    y[t]= a+l*x[t]+rho*y[t-1]+e[t]
ax[1,0].plot(y)
ax[1,0].set_title(r'$y_t = 1+l*t +\rho y_{t-1}+\varepsilon_t$', fontsize = 11)

# 상수항X, 추세항X, rho=1의 시계열(확률보행)
for t in range(1, N):
    y[t]= y[t-1]+e[t]
ax[1,1].plot(y)
ax[1,1].set_title(r'$y_t = y_{t-1}+v_t$', fontsize = 11)

# 상수항 0.1, 추세항X, rho=1의 시계열(확률보행)
a = 0.1
for t in range(1, N):
    y[t]= a+y[t-1]+e[t]
ax[2,0].plot(y)
ax[2,0].set_title(r'$y_t=0.1+y_{t-1}+\varepsilon_t$', fontsize = 11)
ax[2,0].axhline(0, xmin = 0, xmax = N)
ax[2,0].set_xlabel('Time', fontsize = 11)

# 상수항 0.1, 추세항, rho=1의 시계열(확률보행)
for t in range(1, N):
    y[t]= a+l*x[t]+y[t-1]+e[t]
ax[2,1].plot(y)
ax[2,1].set_title(r'$y_t=0.1+l*t+ y_{t-1}+\varepsilon_t$', fontsize = 11)
ax[2,1].set_xlabel('Time', fontsize = 11) 

fig.tight_layout()
plt.show()

# plt.savefig('Figs/ar1artificial12.pdf'); plt.close()
plt.rcParams.update({'font.size': 11}) # Restore font size 


# In[26]:


# 상수항 0.1, 추세항X, rho=1의 시계열(확률보행)의 ACF, PACF 그래프 그리기
a = 0.1
for t in range(1, N):
    y[t]= a+y[t-1]+e[t]
    
fig, ax = plt.subplots(2,1,  figsize = (12,8))
plot_acf(y, lags=40, ax=ax[0], zero=False)
plot_pacf(y, lags=40, ax=ax[1], zero=False)  
fig.tight_layout()
plt.show()


# In[3]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)

new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# ### Plot the dataset

# In[4]:


# 증가율 
data['Gr_gdp'] = 100*data['gdp'].pct_change(periods=4)

# 단순 이동평균(SMA)
data['SMA_gdp']    = data['gdp'].rolling(window=4).mean()
data['SMA_Gr_gdp'] = data['Gr_gdp'].rolling(window=4).mean()

# 지수 가중이동평균(EWMA) 
data['EWMA_gdp']    = data['gdp'].ewm(span=4, adjust=False).mean()
data['EWMA_Gr_gdp'] = data['Gr_gdp'].ewm(span=4, adjust=False).mean()
data


# In[5]:


df = data.dropna()
df


# In[6]:


fig, ax = plt.subplots(2,1, figsize = (12,8))
ax[0].plot(df['gdp'])
ax[1].plot(df['Gr_gdp'])
fig.tight_layout()
plt.show()


# In[7]:


fig, ax = plt.subplots(2,1, figsize = (12,8))
ax[0].plot(df['gdp'], label='Actual')
ax[0].plot(df['SMA_gdp'], label='Moving Average')
ax[0].plot(df['EWMA_gdp'], label='Exponentially Weighted Moving Average')
ax[0].set_title('한국의 국민소득(GDP)')
ax[0].set_ylabel('국민소득(GDP)')
ax[0].legend(loc='upper left')

ax[1].plot(df['Gr_gdp'], label='Actual')
ax[1].plot(df['SMA_Gr_gdp'], label='Moving Average')
ax[1].plot(df['EWMA_Gr_gdp'], label='Exponentially Weighted Moving Average')
ax[1].set_title('한국의 국민소득 증가율(GDP Growth Rate)')
ax[1].set_ylabel('국민소득 증가율(GDP Growth Rate)')
ax[1].legend(loc='upper left')
fig.tight_layout()
plt.show();


# ## Statsmodels to get the trend(Hodrick-Prescott filter)
# <div class="alert alert-info"><h3>Related Function:</h3>
# <tt><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html'><strong>statsmodels.tsa.filters.hp_filter.hpfilter</strong></a><font color=black>(X, lamb=1600)</font>&nbsp;&nbsp;Hodrick-Prescott filter</div>
#     
# The <a href='https://en.wikipedia.org/wiki/Hodrick%E2%80%93Prescott_filter'>Hodrick-Prescott filter</a> separates a time-series  $y_t$ into a trend component $\tau_t$ and a cyclical component $c_t$
# 
# $y_t = \tau_t + c_t$
# 
# The components are determined by minimizing the following quadratic loss function, where $\lambda$ is a smoothing parameter:
# 
# $\min_{\\{ \tau_{t}\\} }\sum_{t=1}^{T}c_{t}^{2}+\lambda\sum_{t=1}^{T}\left[\left(\tau_{t}-\tau_{t-1}\right)-\left(\tau_{t-1}-\tau_{t-2}\right)\right]^{2}$
# 
# 
# The $\lambda$ value above handles variations in the growth rate of the trend component.<br>When analyzing quarterly data, the default lambda value of 1600 is recommended. Use 6.25 for annual data, and 129,600 for monthly data.

# ### Hodrick-Prescott filter

# In[8]:


# Hodrick-Prescott filter
gdp_cycle, gdp_trend = hpfilter(df['gdp'], lamb=1600)
df['gdp_trend'] = gdp_trend
df['gdp_cycle'] = gdp_cycle

ax =df[['gdp', 'gdp_trend', 'gdp_cycle']].plot(figsize=(14,8))
ax.autoscale(axis='x',tight=True)
ax.set_ylabel('실질국민소득(REAL GDP)')
ax.set_title('한국의 국민소득 추세와 경기요인(GDP Trend and Cycle)')
plt.show();


# In[9]:


# 증가율지표에 대한 HP
Gr_gdp_cycle, Gr_gdp_trend = hpfilter(df['Gr_gdp'], lamb=1600)


# In[10]:


df['Gr_gdp_trend'] = Gr_gdp_trend
df['Gr_gdp_cycle'] = Gr_gdp_cycle
df


# In[11]:


ax =df[['Gr_gdp', 'Gr_gdp_trend', 'Gr_gdp_cycle']].plot(figsize=(14,8))
ax.autoscale(axis='x',tight=True)
ax.set(ylabel='GDP증가율(Growth Rate of GDP)');


# In[12]:


fig, ax = plt.subplots(3,1, figsize = (12,8))
ax[0].plot(df['Gr_gdp'], label='GDP Growth Rate')
ax[0].legend(loc='upper left')
ax[1].plot(df['Gr_gdp_trend'], label='GDP Trend')
ax[1].legend(loc='upper left')
ax[2].plot(df['Gr_gdp_cycle'], label='GDP Cycle')
ax[2].legend(loc='upper left')
fig.tight_layout()
plt.show()


# ## ETS
# 
# ## Error/Trend/Seasonality Models
# As we begin working with <em>endogenous</em> data ("endog" for short) and start to develop forecasting models, it helps to identify and isolate factors working within the system that influence behavior. Here the name "endogenous" considers internal factors, while "exogenous" would relate to external forces. These fall under the category of <em>state space models</em>, and include <em>decomposition</em> (described below), and <em>exponential smoothing</em> (described in an upcoming section).
# 
# The <a href='https://en.wikipedia.org/wiki/Decomposition_of_time_series'>decomposition</a> of a time series attempts to isolate individual components such as <em>error</em>, <em>trend</em>, and <em>seasonality</em> (ETS). We've already seen a simplistic example of this in the <strong>Introduction to Statsmodels</strong> section with the Hodrick-Prescott filter. There we separated data into a trendline and a cyclical feature that mapped observed data back to the trend.
# 
# <div class="alert alert-info"><h3>Related Function:</h3>
# <tt><strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html'>statsmodels.tsa.seasonal.seasonal_decompose</a></strong><font color=black>(x, model)</font>&nbsp;&nbsp;
# Seasonal decomposition using moving averages</tt>
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://otexts.com/fpp2/ets.html'>Forecasting: Principles and Practice</a></strong>&nbsp;&nbsp;<font color=black>Innovations state space models for exponential smoothing</font><br>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Decomposition_of_time_series'>Wikipedia</a></strong>&nbsp;&nbsp;<font color=black>Decomposition of time series</font></div>
# 
# ## Seasonal Decomposition
# Statsmodels provides a <em>seasonal decomposition</em> tool we can use to separate out the different components. This lets us see quickly and visually what each component contributes to the overall behavior.
# 
# 
# We apply an <strong>additive</strong> model when it seems that the trend is more linear and the seasonality and trend components seem to be constant over time (e.g. every year we add 10,000 passengers).<br>
# A <strong>multiplicative</strong> model is more appropriate when we are increasing (or decreasing) at a non-linear rate (e.g. each year we double the amount of passengers).
# 
# For these examples we'll use the International Airline Passengers dataset, which gives monthly totals in thousands from January 1949 to December 1960.

# In[13]:


result = seasonal_decompose(df['gdp'], model='multiplicative', period=4) 
# result
#dir(result)
df['Seasonal_gdp'] = result.seasonal
df['Trend_gdp'] = result.trend
df['Resid_gdp'] = result.resid
df


# In[14]:


result.plot();


# In[15]:


resultr = seasonal_decompose(df['Gr_gdp'], model='addtive', period=4) 
# result
#dir(result)
df['Seasonal_GR_gdp'] = resultr.seasonal
df['Trend_Gr_gdp'] = resultr.trend
df['Resid_Gr_gdp'] = resultr.resid
df


# In[16]:


resultr.plot();


# # MA
# ## Moving Averages
# In this section we'll compare <em>Simple Moving Averages</em> to <em>Exponentially Weighted Moving Averages</em> in terms of complexity and performance.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong><a href='https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html'>pandas.DataFrame.rolling</a></strong><font color=black>(window)</font>&nbsp;&nbsp;
# Provides rolling window calculations<br>
# <strong><a href='https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html'>pandas.DataFrame.ewm</a></strong><font color=black>(span)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# Provides exponential weighted functions</tt></div></div>
# 
# ### Perform standard imports and load the dataset
# For these examples we'll use the International Airline Passengers dataset, which gives monthly totals in thousands from January 1949 to December 1960.

# ___
# # SMA
# ## Simple Moving Average
# 
# We've already shown how to create a <a href='https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average'>simple moving average</a> by applying a <tt>mean</tt> function to a rolling window.
# 
# For a quick review:

# ___
# # EWMA
# ## Exponentially Weighted Moving Average 
# 
# We just showed how to calculate the SMA based on some window. However, basic SMA has some weaknesses:
# * Smaller windows will lead to more noise, rather than signal
# * It will always lag by the size of the window
# * It will never reach to full peak or valley of the data due to the averaging.
# * Does not really inform you about possible future behavior, all it really does is describe trends in your data.
# * Extreme historical values can skew your SMA significantly
# 
# To help fix some of these issues, we can use an <a href='https://en.wikipedia.org/wiki/Exponential_smoothing'>EWMA (Exponentially weighted moving average)</a>.
# 
# EWMA will allow us to reduce the lag effect from SMA and it will put more weight on values that occured more recently (by applying more weight to the more recent values, thus the name). The amount of weight applied to the most recent values will depend on the actual parameters used in the EWMA and the number of periods given a window size.
# [Full details on Mathematics behind this can be found here](http://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows).
# Here is the shorter version of the explanation behind EWMA.
# 
# The formula for EWMA is:
# ### $y_t =   \frac{\sum\limits_{i=0}^t w_i x_{t-i}}{\sum\limits_{i=0}^t w_i}$
# 
# Where $x_t$ is the input value, $w_i$ is the applied weight (Note how it can change from $i=0$ to $t$), and $y_t$ is the output.
# 
# Now the question is, how to we define the weight term $w_i$?
# 
# This depends on the <tt>adjust</tt> parameter you provide to the <tt>.ewm()</tt> method.
# 
# When <tt>adjust=True</tt> (default) is used, weighted averages are calculated using weights equal to $w_i = (1 - \alpha)^i$
# 
# which gives
# 
# ### $y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
# + (1 - \alpha)^t x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
# + (1 - \alpha)^t}$
# When <tt>adjust=False</tt> is specified, moving averages are calculated as:
# 
# ### $\begin{split}y_0 &= x_0 \\
# y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,\end{split}$
# 
# which is equivalent to using weights:
# 
#  \begin{split}w_i = \begin{cases}
#     \alpha (1 - \alpha)^i & \text{if } i < t \\
#     (1 - \alpha)^i        & \text{if } i = t.
# \end{cases}\end{split}
# When <tt>adjust=True</tt> we have $y_0=x_0$ and from the last representation above we have 
# $y_t=\alpha x_t+(1−α)y_{t−1}$, therefore there is an assumption that $x_0$ is not an ordinary value but rather an exponentially weighted moment of the infinite series up to that point.
# 
# For the smoothing factor $\alpha$ one must have $0<\alpha≤1$, and while it is possible to pass <em>alpha</em> directly, it’s often easier to think about either the <em>span</em>, <em>center of mass</em> (com) or <em>half-life</em> of an EW moment:
# 
# \begin{split}\alpha =
#  \begin{cases}
#      \frac{2}{s + 1},               & \text{for span}\ s \geq 1\\
#      \frac{1}{1 + c},               & \text{for center of mass}\ c \geq 0\\
#      1 - \exp^{\frac{\log 0.5}{h}}, & \text{for half-life}\ h > 0
#  \end{cases}\end{split}
#  * <strong>Span</strong> corresponds to what is commonly called an “N-day EW moving average”.
# * <strong>Center of mass</strong> has a more physical interpretation and can be thought of in terms of span: $c=(s−1)/2$
# * <strong>Half-life</strong> is the period of time for the exponential weight to reduce to one half.
# * <strong>Alpha</strong> specifies the smoothing factor directly.
# 
# We have to pass precisely one of the above into the <tt>.ewm()</tt> function. For our data we'll use <tt>span=12</tt>.
#  

# ## Simple Exponential Smoothing
# The above example employed <em>Simple Exponential Smoothing</em> with one smoothing factor <strong>α</strong>. Unfortunately, this technique does a poor job of forecasting when there is a trend in the data as seen above. In the next section we'll look at <em>Double</em> and <em>Triple Exponential Smoothing</em> with the Holt-Winters Methods.

# # Holt-Winters Methods
# In the previous section on <strong>Exponentially Weighted Moving Averages</strong> (EWMA) we applied <em>Simple Exponential Smoothing</em> using just one smoothing factor $\alpha$ (alpha). This failed to account for other contributing factors like trend and seasonality.
# 
# In this section we'll look at <em>Double</em> and <em>Triple Exponential Smoothing</em> with the <a href='https://otexts.com/fpp2/holt-winters.html'>Holt-Winters Methods</a>. 
# 
# In <strong>Double Exponential Smoothing</strong> (aka Holt's Method) we introduce a new smoothing factor $\beta$ (beta) that addresses trend:
# 
# \begin{split}l_t &= (1 - \alpha) l_{t-1} + \alpha x_t, & \text{    level}\\
# b_t &= (1-\beta)b_{t-1} + \beta(l_t-l_{t-1}) & \text{    trend}\\
# y_t &= l_t + b_t & \text{    fitted model}\\
# \hat y_{t+h} &= l_t + hb_t & \text{    forecasting model (} h = \text{# periods into the future)}\end{split}
# 
# Because we haven't yet considered seasonal fluctuations, the forecasting model is simply a straight sloped line extending from the most recent data point. We'll see an example of this in upcoming lectures.
# 
# With <strong>Triple Exponential Smoothing</strong> (aka the Holt-Winters Method) we introduce a smoothing factor $\gamma$ (gamma) that addresses seasonality:
# 
# \begin{split}l_t &= (1 - \alpha) l_{t-1} + \alpha x_t, & \text{    level}\\
# b_t &= (1-\beta)b_{t-1} + \beta(l_t-l_{t-1}) & \text{    trend}\\
# c_t &= (1-\gamma)c_{t-L} + \gamma(x_t-l_{t-1}-b_{t-1}) & \text{    seasonal}\\
# y_t &= (l_t + b_t) c_t & \text{    fitted model}\\
# \hat y_{t+m} &= (l_t + mb_t)c_{t-L+1+(m-1)modL} & \text{    forecasting model (} m = \text{# periods into the future)}\end{split}
# 
# Here $L$ represents the number of divisions per cycle. In our case looking at monthly data that displays a repeating pattern each year, we would use $L=12$.
# 
# In general, higher values for $\alpha$, $\beta$ and $\gamma$ (values closer to 1), place more emphasis on recent data.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html'>statsmodels.tsa.holtwinters.SimpleExpSmoothing</a></strong><font color=black>(endog)</font>&nbsp;&nbsp;&nbsp;&nbsp;
# Simple Exponential Smoothing<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html'>statsmodels.tsa.holtwinters.ExponentialSmoothing</a></strong><font color=black>(endog)</font>&nbsp;&nbsp;
#     Holt-Winters Exponential Smoothing</tt>
#     
# <h3>For Further Reading:</h3>
# <tt>
# <strong>
# <a href='https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm'>NIST/SEMATECH e-Handbook of Statistical Methods</a></strong>&nbsp;&nbsp;<font color=black>What is Exponential Smoothing?</font></tt></div>

# ___
# ## Simple Exponential Smoothing
# 
# A variation of the statmodels Holt-Winters function provides Simple Exponential Smoothing. We'll show that it performs the same calculation of the weighted moving average as the pandas <tt>.ewm()</tt> method:<br>
# $\begin{split}y_0 &= x_0 \\
# y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,\end{split}$

# In[17]:


span = 4
alpha = 2/(span+1)

data['SES_gdp']=SimpleExpSmoothing(data['gdp']).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)
data.head()


# In[18]:


data['HWESADD_gdp'] = ExponentialSmoothing(data['gdp'], trend='add').fit().fittedvalues.shift(-1)
data['HWESMUL_gdp'] = ExponentialSmoothing(data['gdp'], trend='mul').fit().fittedvalues.shift(-1)

data[['gdp', 'HWESADD_gdp', 'HWESMUL_gdp']].plot(figsize=(15,10))


# In[19]:


data['HW3ESADD_gdp'] = ExponentialSmoothing(data['gdp'],trend='add',seasonal='add',seasonal_periods=4).fit().fittedvalues
data['HW3ESMUL_gdp'] = ExponentialSmoothing(data['gdp'],trend='mul',seasonal='mul',seasonal_periods=4).fit().fittedvalues
data[['gdp', 'HW3ESADD_gdp', 'HW3ESMUL_gdp']].plot(figsize=(15,10))


# In[45]:


# 일정구간만 그래프로 그리는 방법
# data[['gdp', 'HW3ESADD_gdp', 'HW3ESMUL_gdp']].loc['2020-03-31':].plot(figsize=(15,10))
# data[['gdp', 'HW3ESADD_gdp', 'HW3ESMUL_gdp']].iloc[:12].plot(figsize=(15,10))
# data[['gdp', 'HW3ESADD_gdp', 'HW3ESMUL_gdp']].iloc[-12:].plot(figsize=(15,10))


# ## 예측 : 다음 단원에서 좀 더 자세히

# In[33]:


fitted_model = ExponentialSmoothing(data['gdp'],trend='add',seasonal='add',seasonal_periods=4).fit()
FORCST_gdp  = fitted_model.forecast(12)
FORCST_gdp


# In[37]:


# 예측치와 데이터프레임 결합해서 그래프 그리는 방법?: 데이터 프레임 이용
forcast = pd.DataFrame(FORCST_gdp, columns=['FORCST_gdp'])
all = pd.concat([data, forcast], axis=1)
all[['gdp', 'HW3ESMUL_gdp', 'FORCST_gdp']].plot(figsize=(15,10),xlim=['2000-03-31','2026-12-31'])

all[['gdp', 'HW3ESMUL_gdp', 'FORCST_gdp']].iloc[-36:].plot(figsize=(15,10))

# 예측치와 데이터프레임 결합해서 그래프 그리는 방법?: 별도로 그려서 합쳐지게 하는 방법, 다음 단원 참조


# # Introduction to Forecasting
# In the previous section we fit various smoothing models to existing data. The purpose behind this is to predict what happens next.<br>
# What's our best guess for next month's value? For the next six months?
# 
# In this section we'll look to extend our models into the future. First we'll divide known data into training and testing sets, and evaluate the performance of a trained model on known test data.
# 
# * Goals
#   * Compare a Holt-Winters forecasted model to known data
#   * Understand <em>stationarity</em>, <em>differencing</em> and <em>lagging</em>
#   * Introduce ARIMA and describe next steps
#   
#   ### <font color=blue>Simple Exponential Smoothing / Simple Moving Average</font>
# This is the simplest to forecast. $\hat{y}$ is equal to the most recent value in the dataset, and the forecast plot is simply a horizontal line extending from the most recent value.
# ### <font color=blue>Double Exponential Smoothing / Holt's Method</font>
# This model takes trend into account. Here the forecast plot is still a straight line extending from the most recent value, but it has slope.
# ### <font color=blue>Triple Exponential Smoothing / Holt-Winters Method</font>
# This model has (so far) the "best" looking forecast plot, as it takes seasonality into account. When we expect regular fluctuations in the future, this model attempts to map the seasonal behavior.

# ## Forecasting with the Holt-Winters Method
# For this example we'll use the same airline_passengers dataset, and we'll split the data into 108 training records and 36 testing records. Then we'll evaluate the performance of the model.

# In[41]:


new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
# data.index = pd.to_datetime(new_index)
data.index


# In[5]:


# 데이터셋을 train set과 test set으로 나누기 
train_data = data.loc[:'2020-1-31']
test_data = data.loc['2020-1-31':]
train_data


# In[6]:


test_data


# In[6]:


#from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['gdp'],trend='mul',seasonal='mul',seasonal_periods=4).fit()


# In[7]:


test_predictions = fitted_model.forecast(16)


# In[8]:


train_data['gdp'].plot(legend=True,label='TRAIN')
test_data['gdp'].plot(legend=True,label='TEST',figsize=(12,5));


# In[9]:


# 도표 범위 지정 1
train_data['gdp'].loc['2010-03-31':].plot(legend=True,label='TRAIN')
test_data['gdp'].loc['2010-03-31':].plot(legend=True,label='TEST',figsize=(12,5))
test_predictions.plot(legend=True,label='PREDICTION');


# In[62]:


# 도표 범위 지정 2
train_data['gdp'].plot(legend=True,label='TRAIN')
test_data['gdp'].plot(legend=True,label='TEST',figsize=(12,5))
test_predictions.plot(legend=True,label='PREDICTION',xlim=['2010-12-31','2023-06-30']);


# In[67]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(test_data['gdp'])
print(test_predictions)


# In[25]:


mae = mean_absolute_error(test_data['gdp'],test_predictions)
print(mae)


# In[73]:


mse = mean_squared_error(test_data['gdp'],test_predictions)
print(mse)


# In[74]:


rmse = np.sqrt(mean_squared_error(test_data['gdp'],test_predictions))
print(rmse)


# In[75]:


r2 = r2_score(test_data['gdp'],test_predictions)
print(r2)


# In[10]:


# from statsmodels.tools.eval_measures import mse, rmse, meanabs, aic, bic
mae = meanabs(test_data['gdp'],test_predictions)
mse = mse(test_data['gdp'],test_predictions)
rmse = rmse(test_data['gdp'],test_predictions)
#aic = aic(test_data['gdp'],test_predictions)
#bic = bic(test_data['gdp'],test_predictions)
print(mae, mse, rmse)


# ### 미래예측법

# In[77]:


forcst_model = ExponentialSmoothing(data['gdp'],trend='mul',seasonal='mul',seasonal_periods=4).fit()
forecast = forcst_model.forecast(20)


# In[79]:


data['gdp'].loc['2000-03-31':].plot(figsize=(12,5))
forecast.plot();


# # Stationarity
# Time series data is said to be <em>stationary</em> if it does <em>not</em> exhibit trends or seasonality. That is, the mean, variance and covariance should be the same for any segment of the series, and are not functions of time.<br>
# The file <tt>samples.csv</tt> contains made-up datasets that illustrate stationary and non-stationary data.
# 
# <div class="alert alert-info"><h3>For Further Reading:</h3>
# <strong>
# <a href='https://otexts.com/fpp2/stationarity.html'>Forecasting: Principles and Practice</a></strong>&nbsp;&nbsp;<font color=black>Stationarity and differencing</font></div>

# In[1]:


exec(open('E:\JupyterWorkingDirectory\MyStock\Functions/Traditional_Econometrics_Lib.py').read())


# In[73]:


df = pd.read_csv('../Data/samples.csv',index_col=0,parse_dates=True)
df


# In[89]:


df['a'].plot(ylim=[0,100],title="안정적 데이터(STATIONARY DATA)").autoscale(axis='x',tight=True);


# In[90]:


df2['b'].plot(ylim=[0,100],title="불안정 시계열(NON-STATIONARY DATA)").autoscale(axis='x',tight=True);


# In[91]:


df2['c'].plot(ylim=[0,10000],title="심한 불안정 시계열(MORE NON-STATIONARY DATA)").autoscale(axis='x',tight=True);


# # Differencing
# ## First Order Differencing
# Non-stationary data can be made to look stationary through <em>differencing</em>. A simple method called <em>first order differencing</em> calculates the difference between consecutive observations.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$y^{\prime}_t = y_t - y_{t-1}$
# 
# In this way a linear trend is transformed into a horizontal set of values.

# In[94]:


# Calculate the first difference of the non-stationary dataset "b"
df2['d1b'] = df2['b'].diff() 

df2[['b','d1b']].head()


# In[95]:


df2['d1b'].plot(title="1계차분(FIRST ORDER DIFFERENCE)").autoscale(axis='x',tight=True);


# ## Second order differencing
# Sometimes the first difference is not enough to attain stationarity, particularly if the trend is not linear. We can difference the already differenced values again to obtain a second order set of values.
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$\begin{split}y_{t}^{\prime\prime} &= y_{t}^{\prime} - y_{t-1}^{\prime} \\
# &= (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) \\
# &= y_t - 2y_{t-1} + y_{t-2}\end{split}$

# In[98]:


# First we'll look at the first order difference of dataset "c"
df2['d1c'] = df2['c'].diff()

df2['d1c'].plot(title="FIRST ORDER DIFFERENCE").autoscale(axis='x',tight=True);


# In[100]:


# We can do this from the original time series in one step
df2['d2c'] = df2['c'].diff().diff()
df2[['c','d1c','d2c']].head()


# In[101]:


df2['d2c'].plot(title="2계 차분(SECOND ORDER DIFFERENCE)").autoscale(axis='x',tight=True);


# # Introduction to ARIMA Models
# We'll investigate a variety of different forecasting models in upcoming sections, but they all stem from ARIMA.
# 
# <strong>ARIMA</strong>, or <em>Autoregressive Integrated Moving Average</em> is actually a combination of 3 models:
# * <strong>AR(p)</strong> Autoregression - a regression model that utilizes the dependent relationship between a current observation and observations over a previous period
# * <strong>I(d)</strong> Integration - uses differencing of observations (subtracting an observation from an observation at the previous time step) in order to make the time series stationary
# * <strong>MA(q)</strong> Moving Average - a model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# 
# <strong>Moving Averages</strong> we've already seen with EWMA and the Holt-Winters Method.<br>
# <strong>Integration</strong> will apply differencing to make a time series stationary, which ARIMA requires.<br>
# <strong>Autoregression</strong> is explained in detail in the next section. Here we're going to correlate a current time series with a lagged version of the same series.<br>
# Once we understand the components, we'll investigate how to best choose the $p$, $d$ and $q$ values required by the model.

# # ACF and PACF
# # Autocorrelation Function / Partial Autocorrelation Function
# Before we can investigate <em>autoregression</em> as a modeling tool, we need to look at <em>covariance</em> and <em>correlation</em> as they relate to lagged (shifted) samples of a time series.
# 
# 
# ### Goals
#  * Be able to create ACF and PACF charts
#  * Create these charts for multiple times series, one with seasonality and another without
#  * Be able to calculate Orders PQD terms for ARIMA off these charts (highlight where they cross the x axis)
#  
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acovf.html'>stattools.acovf</a></strong><font color=black>(x[, unbiased, demean, fft, …])</font>&nbsp;Autocovariance for 1D<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html'>stattools.acf</a></strong><font color=black>(x[, unbiased, nlags, qstat, …])</font>&nbsp;&nbsp;Autocorrelation function for 1d arrays<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html'>stattools.pacf</a></strong><font color=black>(x[, nlags, method, alpha])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Partial autocorrelation estimated<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf_yw.html'>stattools.pacf_yw</a></strong><font color=black>(x[, nlags, method])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Partial autocorrelation estimated with non-recursive yule_walker<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf_ols.html'>stattools.pacf_ols</a></strong><font color=black>(x[, nlags])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Calculate partial autocorrelations</tt>
#    
# <h3>Related Plot Methods:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html'>tsaplots.plot_acf</a></strong><font color=black>(x)</font>&nbsp;&nbsp;&nbsp;Plot the autocorrelation function<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_pacf.html'>tsaplots.plot_pacf</a></strong><font color=black>(x)</font>&nbsp;&nbsp;Plot the partial autocorrelation function</tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Autocovariance'>Wikipedia:</a></strong>&nbsp;&nbsp;<font color=black>Autocovariance</font><br>
# <strong>
# <a href='https://otexts.com/fpp2/autocorrelation.html'>Forecasting: Principles and Practice</a></strong>&nbsp;&nbsp;<font color=black>Autocorrelation</font><br>
# <strong>
# <a href='https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4463.htm'>NIST Statistics Handbook</a></strong>&nbsp;&nbsp;<font color=black>Partial Autocorrelation Plot</font></div>
# 

# In[80]:


exec(open('E:\JupyterWorkingDirectory\MyStock\Functions/Traditional_Econometrics_Lib.py').read())


# ## Autocovariance for 1D
# In a <em>deterministic</em> process, like $y=sin(x)$, we always know the value of $y$ for a given value of $x$. However, in a <em>stochastic</em> process there is always some randomness that prevents us from knowing the value of $y$. Instead, we analyze the past (or <em>lagged</em>) behavior of the system to derive a probabilistic estimate for $\hat{y}$.
# 
# One useful descriptor is <em>covariance</em>. When talking about dependent and independent $x$ and $y$ variables, covariance describes how the variance in $x$ relates to the variance in $y$. Here the size of the covariance isn't really important, as $x$ and $y$ may have very different scales. However, if the covariance is positive it means that $x$ and $y$ are changing in the same direction, and may be related.
# 
# With a time series, $x$ is a fixed interval. Here we want to look at the variance of $y_t$ against lagged or shifted values of $y_{t+k}$
# 
# For a stationary time series, the autocovariance function for $\gamma$ (gamma) is given as:
# 
# ${\displaystyle {\gamma}_{XX}(t_{1},t_{2})=\operatorname {Cov} \left[X_{t_{1}},X_{t_{2}}\right]=\operatorname {E} [(X_{t_{1}}-\mu _{t_{1}})(X_{t_{2}}-\mu _{t_{2}})]}$
# 
# We can calculate a specific $\gamma_k$ with:
# 
# ${\displaystyle \gamma_k = \frac 1 n \sum\limits_{t=1}^{n-k} (y_t - \bar{y})(y_{t+k}-\bar{y})}$

# ### Autocovariance Example:
# Say we have a time series with five observations: {13, 5, 11, 12, 9}.<br>
# We can quickly see that $n = 5$, the mean $\bar{y} = 10$, and we'll see that the variance $\sigma^2 = 8$.<br>
# The following calculations give us our covariance values:
# <br><br>
# $\gamma_0 = \frac {(13-10)(13-10)+(5-10)(5-10)+(11-10)(11-10)+(12-10)(12-10)+(9-10)(9-10)} 5 = \frac {40} 5 = 8.0 \\
# \gamma_1 = \frac {(13-10)(5-10)+(5-10)(11-10)+(11-10)(12-10)+(12-10)(9-10)} 5 = \frac {-20} 5 = -4.0 \\
# \gamma_2 = \frac {(13-10)(11-10)+(5-10)(12-10)+(11-10)(9-10)} 5 = \frac {-8} 5 = -1.6 \\
# \gamma_3 = \frac {(13-10)(12-10)+(5-10)(9-10)} 5 = \frac {11} 5 = 2.2 \\
# \gamma_4 = \frac {(13-10)(9-10)} 5 = \frac {-3} 5 = -0.6$
# <br><br>
# Note that $\gamma_0$ is just the population variance $\sigma^2$
# 
# Let's see if statsmodels gives us the same results! For this we'll create a <strong>fake</strong> dataset:

# ## Autocorrelation for 1D
# The correlation $\rho$ (rho) between two variables $y_1,y_2$ is given as:
# 
# ### $\rho = \frac {\operatorname E[(y_1−\mu_1)(y_2−\mu_2)]} {\sigma_{1}\sigma_{2}} = \frac {\operatorname {Cov} (y_1,y_2)} {\sigma_{1}\sigma_{2}}$,
# 
# where $E$ is the expectation operator, $\mu_{1},\sigma_{1}$ and $\mu_{2},\sigma_{2}$ are the means and standard deviations of $y_1$ and $y_2$.
# 
# When working with a single variable (i.e. <em>autocorrelation</em>) we would consider $y_1$ to be the original series and $y_2$ a lagged version of it. Note that with autocorrelation we work with $\bar y$, that is, the full population mean, and <em>not</em> the means of the reduced set of lagged factors (see note below).
# 
# Thus, the formula for $\rho_k$ for a time series at lag $k$ is:
# 
# ${\displaystyle \rho_k = \frac {\sum\limits_{t=1}^{n-k} (y_t - \bar{y})(y_{t+k}-\bar{y})} {\sum\limits_{t=1}^{n} (y_t - \bar{y})^2}}$
# 
# This can be written in terms of the covariance constant $\gamma_k$ as:
# 
# ${\displaystyle \rho_k = \frac {\gamma_k n} {\gamma_0 n} = \frac {\gamma_k} {\sigma^2}}$
# 
# For example,<br>
# $\rho_4 = \frac {\gamma_4} {\sigma^2} = \frac{-0.6} {8} = -0.075$
# 
# Note that ACF values are bound by -1 and 1. That is, ${\displaystyle -1 \leq \rho_k \leq 1}$

# ## Partial Autocorrelation
# Partial autocorrelations measure the linear dependence of one variable after removing the effect of other variable(s) that affect both variables. That is, the partial autocorrelation at lag $k$ is the autocorrelation between $y_t$ and $y_{t+k}$ that is not accounted for by lags $1$ through $k−1$.
# 
# A common method employs the non-recursive <a href='https://en.wikipedia.org/wiki/Autoregressive_model#Calculation_of_the_AR_parameters'>Yule-Walker Equations</a>:
# 
# $\phi_0 = 1\\
# \phi_1 = \rho_1 = -0.50\\
# \phi_2 = \frac {\rho_2 - {\rho_1}^2} {1-{\rho_1}^2} = \frac {(-0.20) - {(-0.50)}^2} {1-{(-0.50)}^2}= \frac {-0.45} {0.75} = -0.60$

# As $k$ increases, we can solve for $\phi_k$ using matrix algebra and the <a href='https://en.wikipedia.org/wiki/Levinson_recursion'>Levinson–Durbin recursion</a> algorithm which maps the sample autocorrelations $\rho$ to a <a href='https://en.wikipedia.org/wiki/Toeplitz_matrix'>Toeplitz</a> diagonal-constant matrix. The full solution is beyond the scope of this course, but the setup is as follows:
# 
# 
# $\displaystyle \begin{pmatrix}\rho_0&\rho_1&\cdots &\rho_{k-1}\\
# \rho_1&\rho_0&\cdots &\rho_{k-2}\\
# \vdots &\vdots &\ddots &\vdots \\
# \rho_{k-1}&\rho_{k-2}&\cdots &\rho_0\\
# \end{pmatrix}\quad \begin{pmatrix}\phi_{k1}\\\phi_{k2}\\\vdots\\\phi_{kk}\end{pmatrix}
# \mathbf = \begin{pmatrix}\rho_1\\\rho_2\\\vdots\\\rho_k\end{pmatrix}$

# In[ ]:


arr = acovf(df['a'])


# In[ ]:


arr2 = acovf(df['a'],unbiased=True)


# In[ ]:


arr3 = acf(df['a'])


# In[ ]:


arr4 = pacf_yw(df['a'],nlags=4,method='mle')


# In[ ]:


arr5 = pacf_yw(df['a'],nlags=4,method='unbiased')


# In[ ]:


arr6 = pacf_ols(df['a'],nlags=4)


# ## ACF Plots
# Plotting the magnitude of the autocorrelations over the first few (20-40) lags can say a lot about a time series.
# 
# For example, consider the stationary <strong>Daily Total Female Births</strong> dataset:

# In[23]:


# 한국의 거시경제 통계자료 불러오기
data = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[24]:


# Let's look first at the ACF array. By default acf() returns 40 lags
acf(data['gdp'])


# In[25]:


# Now let's plot the autocorrelation at different lags
fig, ax = plt.subplots(figsize=(12,4))
title = 'Autocorrelation'
plot_acf(data['gdp'],title=title,lags=40, ax=ax);


# This is a typical ACF plot for stationary data, with lags on the horizontal axis and correlations on the vertical axis. The first value $y_0$ is always 1. A sharp dropoff indicates that there is no AR component in the ARIMA model.
# 
# Next we'll look at non-stationary data with the <strong>Airline Passengers</strong> dataset:

# ## PACF Plots
# Partial autocorrelations work best with stationary data. Let's look first at <strong>Daily Total Female Births</strong>:

# In[26]:


pacf(data['gdp'])


# In[27]:


fig, ax = plt.subplots(figsize=(12,4))
title='Partial Autocorrelation'
plot_pacf(data['gdp'],title=title,lags=40, ax=ax);


# In[28]:


data['D_gdp'] = diff(data[['gdp']], k_diff=4)
data['D_gdp'].plot(figsize=(12,4));


# In[29]:


fig, ax = plt.subplots(figsize=(12,4))
plot_pacf(data['D_gdp'].dropna(),title='PACF',lags=40,ax=ax); 


# In[30]:


fig, ax = plt.subplots(figsize=(12,4))
plot_acf(data['D_gdp'].dropna(),title='AC',lags=40, ax=ax);


# # Descriptive Statistics and Tests
# In upcoming sections we'll talk about different forecasting models like ARMA, ARIMA, Seasonal ARIMA and others. Each model addresses a different type of time series. For this reason, in order to select an appropriate model we need to know something about the data.
# 
# In this section we'll learn how to determine if a time series is <em>stationary</em>, if it's <em>independent</em>, and if two series demonstrate <em>correlation</em> and/or <em>causality</em>.

# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccovf.html'>stattools.ccovf</a></strong><font color=black>(x, y[, unbiased, demean])</font>&nbsp;&nbsp;crosscovariance for 1D<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html'>stattools.ccf</a></strong><font color=black>(x, y[, unbiased])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cross-correlation function for 1d<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.periodogram.html'>stattools.periodogram</a></strong><font color=black>(X)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns the periodogram for the natural frequency of X<br>
#     
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html'>stattools.adfuller</a></strong><font color=black>(x[, maxlag, regression, …])</font>&nbsp;&nbsp;Augmented Dickey-Fuller unit root test<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html'>stattools.kpss</a></strong><font color=black>(x[, regression, lags, store])</font>&nbsp;&nbsp;&nbsp;&nbsp;Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html'>stattools.coint</a></strong><font color=black>(y0, y1[, trend, method, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Test for no-cointegration of a univariate equation<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.bds.html'>stattools.bds</a></strong><font color=black>(x[, max_dim, epsilon, distance])</font>&nbsp;&nbsp;Calculate the BDS test statistic for independence of a time series<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.q_stat.html'>stattools.q_stat</a></strong><font color=black>(x, nobs[, type])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns Ljung-Box Q Statistic<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html'>stattools.grangercausalitytests</a></strong><font color=black>(x, maxlag[, …])</font>&nbsp;Four tests for granger non-causality of 2 timeseries<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.levinson_durbin.html'>stattools.levinson_durbin</a></strong><font color=black>(s[, nlags, isacov])</font>&nbsp;&nbsp;&nbsp;Levinson-Durbin recursion for autoregressive processes<br>
# 
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.mse.html'>stattools.eval_measures.mse</a></strong><font color=black>(x1, x2, axis=0)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mean squared error<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.rmse.html'>stattools.eval_measures.rmse</a></strong><font color=black>(x1, x2, axis=0)</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;root mean squared error<br>
# <strong><a href='https://www.statsmodels.org/stable/generated/statsmodels.tools.eval_measures.meanabs.html'>stattools.eval_measures.meanabs</a></strong><font color=black>(x1, x2, axis=0)</font>&nbsp;&nbsp;mean absolute error<br>
# </tt>
# 
# <h3>For Further Reading:</h3>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test'>Wikipedia:</a></strong>&nbsp;&nbsp;<font color=black>Augmented Dickey–Fuller test</font><br>
# <strong>
# <a href='https://en.wikipedia.org/wiki/KPSS_test'>Wikipedia:</a></strong>&nbsp;&nbsp;<font color=black>Kwiatkowski-Phillips-Schmidt-Shin test</font><br>
# <strong>
# <a href='https://en.wikipedia.org/wiki/Granger_causality'>Wikipedia:</a></strong>&nbsp;&nbsp;<font color=black>Granger causality test</font><br>
# <strong>
# <a href='https://otexts.com/fpp2/accuracy.html'>Forecasting: Principles and Practice:</a></strong>&nbsp;&nbsp;<font color=black>Evaluating forecast accuracy</font>
# 
# </div>
# 
# 

# ## Perform standard imports and load datasets

# In[31]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:\JupyterWorkingDirectory\MyStock\Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# # Tests for Stationarity
# A time series is <em>stationary</em> if the mean and variance are fixed between any two equidistant points. That is, no matter where you take your observations, the results should be the same. A times series that shows seasonality is <em>not</em> stationary.
# 
# A test for stationarity usually involves a <a href='https://en.wikipedia.org/wiki/Unit_root_test'>unit root</a> hypothesis test, where the null hypothesis $H_0$ is that the series is <em>nonstationary</em>, and contains a unit root. The alternate hypothesis $H_1$ supports stationarity. The <a href='https://en.wikipedia.org/wiki/Augmented_Dickey-Fuller_test'>augmented Dickey-Fuller</a> and <a href='https://en.wikipedia.org/wiki/KPSS_test'>Kwiatkowski-Phillips-Schmidt-Shin</a> tests are stationarity tests. 

# ## Augmented Dickey-Fuller Test
# To determine whether a series is stationary we can use the <a href='https://en.wikipedia.org/wiki/Augmented_Dickey-Fuller_test'>augmented Dickey-Fuller Test</a>. In this test the null hypothesis states that $\phi = 1$ (this is also called a unit test). The test returns several statistics we'll see in a moment. Our focus is on the p-value. A small p-value ($p<0.05$) indicates strong evidence against the null hypothesis.
# 
# To demonstrate, we'll use a dataset we know is <em>not</em> stationary, the airline_passenger dataset. First, let's plot the data along with a 12-month rolling mean and standard deviation:

# In[32]:


# 안정성 여부를 도표로 확인
data['gdp_ma'] = data['gdp'].rolling(window=4).mean()
data['gdp_ma_Std'] = data['gdp'].rolling(window=4).std()

data[['gdp','gdp_ma','gdp_ma_Std']].plot(figsize=(12,4));


# Not only is this dataset seasonal with a clear upward trend, the standard deviation increases over time as well.

# In[33]:


# Augmented Dickey-Fuller Test
dftest = adfuller(data['gdp'],autolag='AIC')
dftest


# To find out what these values represent we can run <tt>help(adfuller)</tt>. Then we can add our own labels:

# In[16]:


print('Augmented Dickey-Fuller Test')
dfout = pd.Series(dftest[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])

for key,val in dftest[4].items():
    dfout[f'critical value ({key})']=val

print(dfout)


# In[17]:


data['Gr_gdp'] = data['gdp'].pct_change(periods=4)
data['Gr_gdp'] = data['Gr_gdp']*100
datadna = data.dropna()
datadna


# In[18]:


# Augmented Dickey-Fuller Test for Gr_gdpo
dftest = adfuller(datadna['Gr_gdp'],autolag='AIC')
dftest


# In[19]:


print('Augmented Dickey-Fuller Test for Gr_gdp')
dfout = pd.Series(dftest[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])
for key,val in dftest[4].items():
    dfout[f'critical value ({key})']=val

print(dfout)


# In[38]:


#from arch.unitroot import Augmented Dickey-Fuller
for i in ['n','c','ct']:    
    adf_tt = ADF(data['gdp'], trend=i, lags=5)  
    print(adf_tt.summary().as_text()) 


# In[39]:


adf_ct = ADF(data['gdp'], trend="ct",lags=5)   
print(adf_ct.summary().as_text())

reg_res = adf_ct.regression
print(reg_res.summary().as_text())
reg_res.resid.plot(figsize=(16,5))


# ### Function for running the augmented Dickey-Fuller test
# Since we'll use it frequently in the upcoming forecasts, let's define a function we can copy into future notebooks for running the augmented Dickey-Fuller test. Remember that we'll still have to import <tt>adfuller</tt> at the top of our notebook.

# In[ ]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[65]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    시계열 이름과 부여하고 싶은 타이틀 지정
    """
    print(f'부가된 디키-풀러 테스트(Augmented Dickey-Fuller Test): {title}')
    result = adfuller(series.dropna(),autolag='AIC') 
    
    labels = ['검정통계량(ADF test statistic)','p-value','사용된 시차의 수(# of lags used)','관측치수(# of observations)']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string()) 
    
    if result[1] <= 0.05:
        print("- 귀무가설(단위근 존재, 불안정 시계열)의 강한 증거가 있다!")
        print("- 귀무가설을 기각!")
        print("- 이 시계열 자료는 단위근을 가지지 않아서 안정적이다!")
    else:
        print("- 귀무가설(단위근 존재, 불안정 시계열)의 약한 증거가 있다!")
        print("- 귀무가설을 기각의 기각 실패!")
        print("- 이 시계열 자료는 단위근을 가지고 있어서 불안정한 시계열이다!")


# In[66]:


adf_test(datadna['gdp'], 'GDP growth rate')


# In[67]:


adf_test(datadna['Gr_gdp'], 'GDP growth rate')


# In[68]:


# 인공적으로 만든 다른 데이터셋 사례
df = pd.read_csv('../Data/samples.csv',index_col=0,parse_dates=True)
df.index.freq = 'MS'
df[['a','d']].plot(figsize=(16,5));


# In[69]:


adf_test(df['a'], 'Sample data')


# ## PhillipsPerron Test

# In[32]:


#from arch.unitroot import PhillipsPerron 

for tt in ['n','c','ct']:    
    pp = PhillipsPerron(data['gdp'], trend=tt, test_type='tau')    
    print(pp.summary().as_text())


# ## KPSS(Kwiatkowski-Phillips-Schmidt-Shin) Test

# In[24]:


kpss_test = kpss(data['gdp'])
print(kpss_test)


# In[31]:


print('KPSS(Kwiatkowski-Phillips-Schmidt-Shin) Test for gdp')
dfout = pd.Series(kpss_test[0:4],index=['KPSS test statistic','p-value','# lags used','critical value'])
for key,val in kpss_test[3].items():
    dfout[f'critical value ({key})']=val

print(dfout)


# # Granger Causality Tests
# The <a href='https://en.wikipedia.org/wiki/Granger_causality'>Granger causality test</a> is a a hypothesis test to determine if one time series is useful in forecasting another. While it is fairly easy to measure correlations between series - when one goes up the other goes up, and vice versa - it's another thing to observe changes in one series correlated to changes in another after a consistent amount of time. This <em>may</em> indicate the presence of causality, that changes in the first series influenced the behavior of the second. However, it may also be that both series are affected by some third factor, just at different rates. Still, it can be useful if changes in one series can predict upcoming changes in another, whether there is causality or not. In this case we say that one series "Granger-causes" another.
# 
# In the case of two series, $y$ and $x$, the null hypothesis is that lagged values of $x$ do <em>not</em> explain variations in $y$.<br>
# In other words, it assumes that $x_t$ doesn’t Granger-cause $y_t$.
# 
# The stattools <tt><strong>grangercausalitytests</strong></tt> function offers four tests for granger non-causality of 2 timeseries
# 
# For this example we'll use the samples.csv file, where columns 'a' and 'd' are stationary datasets.

# In[82]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('../Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[84]:


data[['con','gdp']].plot(figsize=(16,5));


# In[88]:


data['gdp_L1'] = data['gdp'].shift(1) 


# In[90]:


grangercausalitytests(data[['con','gdp_L1']].dropna(),maxlag=4);


# In[96]:


data['Gr_gdp'] = 100 * data['gdp'].pct_change(periods=4)
data['Gr_con'] = 100 * data['con'].pct_change(periods=4)
data['Gr_gdp_L1'] = data['Gr_gdp'].shift(1) 
data


# In[97]:


grangercausalitytests(data[['Gr_con','Gr_gdp']].dropna(),maxlag=4);


# In[103]:


grangercausalitytests(data[['Gr_con','Gr_gdp_L1']].dropna(),maxlag=8);


# In[100]:


# 인위적으로 만든 자료에 적용
df3 = pd.read_csv('../Data/samples.csv',index_col=0,parse_dates=True)
df3.index.freq = 'MS'
df3[['a','d']].plot(figsize=(16,5));


# It's hard to tell from this overlay but <tt>df['d']</tt> almost perfectly predicts the behavior of <tt>df['a']</tt>.<br>
# To see this more clearly (spoiler alert!), we will shift <tt>df['d']</tt> two periods forward.

# In[101]:


df3['a'].iloc[2:].plot(figsize=(16,5),legend=True);
df3['d'].shift(2).plot(legend=True);


# ### Run the test
# The function takes in a 2D array [y,x] and a maximum number of lags to test on x. Here our y is column 'a' and x is column 'd'. We'll set maxlags to 3.

# In[102]:


# Add a semicolon at the end to avoid duplicate output
grangercausalitytests(df3[['a','d']],maxlag=3);


# Essentially we're looking for extremely low p-values, which we see at lag 2.<br>
# By comparison, let's compare two datasets that are not at all similar, 'b' and 'd'.

# In[77]:


# Add a semicolon at the end to avoid duplicate output
grangercausalitytests(df3[['b','d']],maxlag=3);


# That's it!

# # Evaluating forecast accuracy
# Two calculations related to linear regression are <a href='https://en.wikipedia.org/wiki/Mean_squared_error'><strong>mean squared error</strong></a> (MSE) and <a href='https://en.wikipedia.org/wiki/Root-mean-square_deviation'><strong>root mean squared error</strong></a> (RMSE)
# 
# The formula for the mean squared error is<br><br>
# &nbsp;&nbsp;&nbsp;&nbsp;$MSE = {\frac 1 L} \sum\limits_{l=1}^L (y_{T+l} - \hat y_{T+l})^2$<br><br>
# where $T$ is the last observation period and $l$ is the lag point up to $L$ number of test observations.
# 
# The formula for the root mean squared error is<br><br>
# &nbsp;&nbsp;&nbsp;&nbsp;$RMSE = \sqrt{MSE} = \sqrt{{\frac 1 L} \sum\limits_{l=1}^L (y_{T+l} - \hat y_{T+l})^2}$<br><br>
# 
# The advantage of the RMSE is that it is expressed in the same units as the data.<br><br>
# 
# A method similar to the RMSE is the <a href='https://en.wikipedia.org/wiki/Mean_absolute_error'><strong>mean absolute error</strong></a> (MAE) which is the mean of the magnitudes of the error, given as<br><br>
# 
# &nbsp;&nbsp;&nbsp;&nbsp;$MAE = {\frac 1 L} \sum\limits_{l=1}^L \mid{y_{T+l}} - \hat y_{T+l}\mid$<br><br>
# 
# A forecast method that minimizes the MAE will lead to forecasts of the median, while minimizing the RMSE will lead to forecasts of the mean.

# In[15]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)
df = pd.DataFrame(np.random.randint(20,30,(50,2)),columns=['test','predictions'])
df.plot(figsize=(12,4));


# In[11]:


MSE = mse(df['test'],df['predictions'])
RMSE = rmse(df['test'],df['predictions'])
MAE = meanabs(df['test'],df['predictions'])

print(f'Model  MSE: {MSE:.3f}')
print(f'Model RMSE: {RMSE:.3f}')
print(f'Model  MAE: {MAE:.3f}')


# ### AIC / BIC
# More sophisticated tests include the <a href='https://en.wikipedia.org/wiki/Akaike_information_criterion'><strong>Akaike information criterion</strong></a> (AIC) and the <a href='https://en.wikipedia.org/wiki/Bayesian_information_criterion'><strong>Bayesian information criterion</strong></a> (BIC).
# 
# The AIC evaluates a collection of models and estimates the quality of each model relative to the others. Penalties are provided for the number of parameters used in an effort to thwart overfitting. The lower the AIC and BIC, the better the model should be at forecasting.
# 
# These functions are available as
# 
# &nbsp;&nbsp;&nbsp;&nbsp;<tt>from from statsmodels.tools.eval_measures import aic, bic</tt>
# 
# but we seldom compute them alone as they are built into many of the statsmodels tools we use.

# # ARIMA모형에서 차수 자동선정(Choosing ARIMA Orders)
# 
# * Goals
#   * Understand PDQ terms for ARIMA (slides)
#   * Understand how to choose orders manually from ACF and PACF
#   * Understand how to use automatic order selection techniques using the functions below
#   
# Before we can apply an ARIMA forecasting model, we need to review the components of one.<br>
# ARIMA, or Autoregressive Independent Moving Average is actually a combination of 3 models:
# * <strong>AR(p)</strong> Autoregression - a regression model that utilizes the dependent relationship between a current observation and observations over a previous period.
# * <strong>I(d)</strong> Integration - uses differencing of observations (subtracting an observation from an observation at the previous time step) in order to make the time series stationary
# * <strong>MA(q)</strong> Moving Average - a model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# 
# <div class="alert alert-info"><h3>Related Functions:</h3>
# <tt>
# <strong>
# <a href='https://www.alkaline-ml.com/pmdarima/user_guide.html#user-guide'>pmdarima.auto_arima</a></strong><font color=black>(y[,start_p,d,start_q, …])</font>&nbsp;&nbsp;&nbsp;Returns the optimal order for an ARIMA model<br>
# 
# <h3>Optional Function (see note below):</h3>
# <strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.arma_order_select_ic.html'>stattools.arma_order_select_ic</a></strong><font color=black>(y[, max_ar, …])</font>&nbsp;&nbsp;Returns information criteria for many ARMA models<br><strong>
# <a href='https://www.statsmodels.org/stable/generated/statsmodels.tsa.x13.x13_arima_select_order.html'>x13.x13_arima_select_order</a></strong><font color=black>(endog[, …])</font>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Perform automatic seasonal ARIMA order identification using x12/x13 ARIMA</tt></div>

# ## Perform standard imports and load datasets

# In[21]:


# 시계열 분석을 위한 라이브러리 불러오기
exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[26]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index


# In[27]:


df = data[['gdp']]
df


# ## pmdarima Auto-ARIMA
# This is a third-party tool separate from statsmodels. It should already be installed if you're using our virtual environment. If not, then at a terminal run:<br>
# &nbsp;&nbsp;&nbsp;&nbsp;<tt>pip install pmdarima</tt>

# In[28]:


#!pip install pmdarima
from pmdarima import auto_arima


# In[109]:


help(auto_arima)


# In[29]:


auto_arima(data['gdp']).summary()


# <div class="alert alert-info"><strong>NOTE: </strong>Harmless warnings should have been suppressed, but if you see an error citing unusual behavior you can suppress this message by passing <font color=black><tt>error_action='ignore'</tt></font> into <tt>auto_arima()</tt>. Also, <font color=black><tt>auto_arima().summary()</tt></font> provides a nicely formatted summary table.</div>

# In[112]:


auto_arima(df['gdp'],error_action='ignore').summary()


# This shows a recommended (p,d,q) ARIMA Order of (1,1,1), with no seasonal_order component.
# 
# We can see how this was determined by looking at the stepwise results. The recommended order is the one with the lowest <a href='https://en.wikipedia.org/wiki/Akaike_information_criterion'>Akaike information criterion</a> or AIC score. Note that the recommended model may <em>not</em> be the one with the closest fit. The AIC score takes complexity into account, and tries to identify the best <em>forecasting</em> model.

# In[113]:


stepwise_fit = auto_arima(df['gdp'], start_p=0, start_q=0,
                          max_p=8, max_q=8, m=4,
                          seasonal=False,
                          d=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# ___
# Now let's look at the non-stationary, seasonal <strong>Airline Passengers</strong> dataset:

# In[116]:


stepwise_fit = auto_arima(df['gdp'], start_p=1, start_q=1,
                          max_p=3, max_q=3, m=4,
                          start_P=0, max_P=4, seasonal=True,
                          d=None, D=None, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

stepwise_fit.summary()


# ## OPTIONAL: statsmodels ARMA_Order_Select_IC
# Statsmodels has a selection tool to find orders for ARMA models on stationary data.

# In[118]:


from statsmodels.tsa.stattools import arma_order_select_ic


# In[119]:


help(arma_order_select_ic)


# In[120]:


arma_order_select_ic(df['gdp'])


# <div class="alert alert-success"><strong>A note about <tt>statsmodels.tsa.x13.x13_arima_select_order</tt></strong><br><br>
# This utility requires installation of <strong>X-13ARIMA-SEATS</strong>, a seasonal adjustment tool developed by the U.S. Census Bureau.<br>
# See <a href='https://www.census.gov/srd/www/x13as/'>https://www.census.gov/srd/www/x13as/</a> for details. Since the installation requires adding x13as to your PATH settings we've deemed it beyond the scope of this course.

# In[122]:


from statsmodels.tsa.x13 import x13_arima_analysis


# In[130]:


help(x13_arima_analysis)


# In[175]:


path = 'E:/JupyterWorkingDirectory//MyStock/x13as/'


# In[176]:


results_x13 = x13_arima_analysis(endog=data['gdp'], x12path=path, prefer_x13=True)


# In[177]:


results_x13.seasadj


# In[178]:


results_x13.trend


# In[179]:


results_x13.irregular


# In[181]:


results_x13.plot()
plt.show()


# In[6]:


# season-Trend decomposition using LOESS(Locally Estimated Scatterplot Smoothing)
from statsmodels.tsa.seasonal import STL


# In[7]:


help(STL)


# In[8]:


res = STL(data['gdp'], period=4).fit()


# In[9]:


res.plot()
plt.show()


# In[21]:


res.trend


# In[22]:


res.seasonal


# In[23]:


res.resid


# In[ ]:





# ### Structural Break

# In[3]:


# 시계열 분석을 위한 라이브러리 불러오기
exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[20]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('E:/JupyterWorkingDirectory/Udemy Time Series/Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index

y = data['con']
X = data['gdp']
X = sm.add_constant(X)

endog = pd.concat([y, X], axis=1)


# In[21]:


# OLS 추정
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# 잔차 그림
results.resid.plot(figsize=(10,4))

# 잔차에 대한 Cusum test
cusum_test = dg.breaks_cusumolsresid(results.resid)
print(cusum_test)

# 잔차에 대한 Hansen test
est_statistic = dg.breaks_hansen(results)
print(est_statistic)


# In[11]:


# 축차형 모형의 추정
results2 = sm.RecursiveLS(y,X).fit()
print(results2.summary())

results2.resid.plot(figsize=(10,4))


# In[19]:


results2.plot_cusum(figsize=(10,4))
results2.plot_cusum_squares(figsize=(10,4))
plt.show()


# In[14]:


X.shape


# In[18]:


for i in range(0, X.shape[1]):
    results2.plot_recursive_coefficient(i, alpha=0.05, figsize=(12,4))


# # Chow test

# In[1]:


# 시계열 분석을 위한 라이브러리 불러오기
exec(open('E:/JupyterWorkingDirectory/MyStock/Functions/Traditional_Econometrics_Lib.py').read())


# In[22]:


# 한국의 거시경제 통계자료 불러오기
# 통계 업데이트 : ecos.bok.or.kr/#/Short/af4c9c
data = pd.read_csv('Data/Korea_GDP.csv',index_col='Time', parse_dates=True)
new_index = pd.date_range(start='1961-03-31', periods=len(data), freq='Q')
data.index = pd.to_datetime(new_index)
data.index

y = data['con'].pct_change(4)*100
X = data['gdp'].pct_change(4)*100
X = sm.add_constant(X)

y = y.dropna(axis=0)
X = X.dropna(axis=0)
print(X)


# In[23]:


# Chow Test 
import scipy.stats as st
n = X.shape[0]
k = X.shape[1]

# 제약된 모형(모든 샘플)과 제약없는 모형(구조변화 반영)
model = sm.OLS(y, X).fit()
SSR = model.ssr
model1 = sm.OLS(y[:"1997-12-31"], X[:"1997-12-31"]).fit()
SSR1 = model1.ssr
model2 = sm.OLS(y["1998-03-31":], X["1998-03-31":]).fit()
SSR2 = model2.ssr

# Chow test를 위한 F-통계량
chow = ((SSR-(SSR1+SSR2))/ k) / ((SSR1+SSR2)/(n-2*k)) 
p_value = 1-st.f.cdf(chow, k, n-2*k)

print('F-statistics: ', np.round(chow, 4))
print('p_value: ', np.round(p_value, 4))
print(n)

# Chow test 전제조건 검정을 위한 F-통계량
n1 = X[:"1997-12-31"].shape[0]
n2 = X["1998-03-31":].shape[0]

homo = SSR1/SSR2 
p_value = 1-st.f.cdf(homo, n1-k, n2-k)
print('F-statistics: ', np.round(homo, 4))
print('p_value: ', np.round(p_value, 4))


# In[47]:


# Recursive Chow Test
chow = 1 
periods = 0

for i in range(k+1, n-(k+1)): 
    model = sm.OLS(y,X).fit()
    SCR = model.ssr
    model1 = sm.OLS(y[:i], X[:i]).fit()
    SCR1 = model1.ssr
    model2 = sm.OLS(y[i:], X[i:]).fit()
    SCR2 = model2.ssr
    chow_i = ((SCR - (SCR1+SCR2))/k) / ((SCR1+SCR2)/(n-2*k))

    if chow_i > chow: 
        periods = i 
        chow = chow_i
        p_value = 1-st.f.cdf(chow_i, k, n-2*k)

print('Break period: ', periods)
print('F-statistics: ', np.round(chow, 4)) 
print('p-value: ', np.round(p_value, 4))


# In[24]:


import statsmodels.api as sm
import numpy as np
import scipy.stats as st

# 날짜변수
dates = pd.date_range(start='1961-01-31', periods=len(y), freq='Q')  # Example of generating dates
print(dates)


# In[49]:


chow = 1 
periods = 0

for i in range(k+1, n-(k+1)): 
    model = sm.OLS(y, X).fit()
    SCR = model.ssr
    model1 = sm.OLS(y[:i], X[:i]).fit()
    SCR1 = model1.ssr
    model2 = sm.OLS(y[i:], X[i:]).fit()
    SCR2 = model2.ssr
    chow_i = ((SCR - (SCR1 + SCR2)) / k) / ((SCR1 + SCR2) / (n - 2 * k))

    if chow_i > chow: 
        periods = dates[i]  # Replace index with corresponding datetime
        chow = chow_i
        p_value = 1 - st.f.cdf(chow_i, k, n - 2 * k)
        print(periods, chow, p_value, i)
      


# In[50]:


print('Break period: ', periods.strftime('%Y-%m-%d'))  # Formatting datetime output
print('F-statistics: ', np.round(chow, 4)) 
print('p-value: ', np.round(p_value, 4))


# In[ ]:





# In[ ]:




