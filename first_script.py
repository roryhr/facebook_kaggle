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

y = df.place_id.value_counts().hist(bins=500)
y.set(ylabel="Duplicate frequency", xlabel="Number of duplicates")

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
s = df.place_id.value_counts().values
#s = np.random.gamma(shape, scale, 1000)

#Display the histogram of the samples, along with the probability density function:


count, bins, ignored = plt.hist(s, 50, normed=True)
#y = bins**(shape-1)*(np.exp(-bins/scale) /
#                      (sps.gamma(shape)*scale**shape))
y = stats.gamma.pdf(bins, a=.0967, loc=.999, scale=2.0549)
 
plt.plot(bins, y, linewidth=2, color='r')
plt.ylim(0,.006)
plt.show()

