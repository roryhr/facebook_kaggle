import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats    
import scipy.special as sps

df = pd.read_csv('train.csv', index_col=0)

print(df.describe())

#%%
x = np.random.rand(100)*9 + 1
np.mean(x)
np.std(x)

#%%

print(df.place_id.drop_duplicates().count())

#%%

y = df.place_id.value_counts().hist(bins=200)
y.set(ylabel="Duplicate frequency", xlabel="Number of duplicates")


#%%

y = df.place_id.value_counts().plot.kde()
y.set_xlim(0, 2009)

#%%

h, bins = np.histogram(df.place_id.value_counts(), bins=200, normed=1)

#fit_alpha, fit_loc, fit_beta=stats.gamma.fit(h)
shape, loc, scale = stats.gamma.fit(h, floc=0, fscale=1)
#print(fit_alpha, fit_loc, fit_beta)

#%%

y = stats.gamma.pdf(h, a=shape, scale=scale, loc=loc)
plt.plot(bins[:-1], y)

#%%
xbars = []
for _ in range(10**4):
    xbars.append(df.place_id.sample(n=10**5, replace=True).mean())
    
n, bins, patches = plt.hist(xbars, bins=40)
#%%

df_sample = df[df.place_id == 4823777529]


#%%
xbars = []
for _ in range(10**4):
    xbars.append(df_sample.x.sample(frac=1, replace=True).mean())
    
n, bins, patches = plt.hist(xbars, bins=40)

#%%
xbars = []
for _ in range(10**4):
    xbars.append(df_sample.y.sample(frac=1, replace=True).mean())
    
n, bins, patches = plt.hist(xbars, bins=40)

#%%
xbars = []
for _ in range(10**4):
    xbars.append(df_sample.accuracy.sample(frac=1, replace=True).mean())
    
n, bins, patches = plt.hist(xbars, bins=40)
#%%
xbars = []
for _ in range(10**4):
    xbars.append(df_sample.x.sample(frac=1, replace=True).std())
    
n, bins, patches = plt.hist(xbars, bins=40)

#%%
x = df.place_id.sort_values().diff()

#%%
ax = df_sample.plot('x', 'y', kind='scatter')
ax.set_ylim([4,7])
ax.set_xlim([4,7])


#%%

shape, scale = 1.34, 199.38 # mean and dispersion
#shape, scale = .34, .001 # mean and dispersion

s = df.place_id.value_counts().values
#s = np.random.gamma(shape, scale, 1000)

#Display the histogram of the samples, along with the probability density function:
ax = df.place_id.value_counts().plot.kde()
ax.set_xlim(0, 2000)

count, bins, ignored = plt.hist(s, 100, normed=True)
#y = bins**(shape-1)*(np.exp(-bins/scale) /
#                      (sps.gamma(shape)*scale**shape))
#y = stats.gamma.pdf(bins, a=.6, loc=.999, scale=2.0549)
#rv = stats.maxwell(loc=-249.6547, scale=336.860199)
rv = stats.frechet_r(1.1, loc=0.89, scale=280)
y = rv.pdf(bins)
ax.plot(bins, y, linewidth=2, color='r')
#ax.ylim(0,.006)
#ax.show()


#%%
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df_sample, diagonal='kde', figsize=(11,11))

#%%
from pandas.tools.plotting import lag_plot
lag_plot(df_sample.time, lag=100)

#%%
from pandas.tools.plotting import autocorrelation_plot
ax = autocorrelation_plot(df.sample(n=1e4, replace=True).time)
ax.set_xlim([0,2000])
ax.set_ylim([-.1,.1])


#%%
from pandas.tools.plotting import bootstrap_plot
bootstrap_plot(df_sample.x)

#%%
from pandas.tools.plotting import radviz
df2 = df[(df.place_id == 8523065625) | 
         (df.place_id == 6567393236) | 
         (df.place_id == 1757726713) | 
         (df.place_id == 7440663949)]

radviz(df2, 'place_id')

#%%
df2.plot('x', 'y', kind='scatter')