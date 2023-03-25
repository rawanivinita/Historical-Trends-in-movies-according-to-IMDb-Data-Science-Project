import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

df_movie = pd.read_csv("IMDB-Movie-Data.csv")

# we found that Revenue and MetaScore are the only columns with missing entries, so we removed rows with the missing  values

df_movie = df_movie.dropna(axis=0, how='any')
df_movie.info()

#Scatter Plot between Year and Revenue
plt.scatter( df_movie['Year'],df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Year')
plt.xlabel('Year')
plt.ylabel('Revenue (Millions)')
#plt.show()

#Scatter Plot between Runtime and Revenue
plt.scatter( df_movie['Runtime (Minutes)'],df_movie['Revenue (Millions)'])
plt.title('Revenue vs Runtime (Minutes)')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Revenue')
#plt.show()


#Scatter Plot between Year and Rating
plt.scatter( df_movie['Year'],df_movie['Rating'])
plt.title('Rating vs Year')
plt.xlabel('Year')
plt.ylabel('Rating')
#plt.show()

#Scatter Plot between Runtime and Rating
plt.scatter( df_movie['Runtime (Minutes)'],df_movie['Rating'])
plt.title('Runtime (Minutes) vs Rating')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Rating')
#plt.show()

#Scatter Plot between Revenue and Rating
plt.scatter( df_movie['Rating'],df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Rating')
plt.xlabel('Rating')
plt.ylabel('Revenue (Millions)')
#plt.show()


# print the correlation
corr_rev_run = df_movie['Runtime (Minutes)'].corr(df_movie['Revenue (Millions)'])
print("Correlation between Runtime and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Year'].corr(df_movie['Revenue (Millions)'])
print("Correlation between year and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Runtime (Minutes)'].corr(df_movie['Rating'])
print("Correlation between Runtime and Rating is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Year'].corr(df_movie['Rating'])
print("Correlation between Year and Rating is: ", round(corr_rev_run, 2))
