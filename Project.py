import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

df_movie = pd.read_csv("IMDB-Movie-Data.csv")

# we found that Revenue and MetaScore are the only columns with missing entries,so we removed rows with the missing values

df_movie = df_movie.dropna(axis=0, how='any')
#df_movie.info()


# Scatter Plot between Year and Revenue
plt.scatter(df_movie['Year'], df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Year')
plt.xlabel('Year')
plt.ylabel('Revenue (Millions)')

# line of best fit (Year Vs Revenue)
year_column = df_movie['Year']
revenue_column = df_movie['Revenue (Millions)']
p = np.polyfit(year_column, revenue_column, 1)
y_fit = p[0] * year_column + p[1]
plt.plot(year_column, y_fit, color='red')

#plt.show()



# Scatter Plot between Runtime and Revenue
plt.scatter(df_movie['Runtime (Minutes)'], df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Runtime (Minutes)')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Revenue (Millions)')

# line of best fit (Runtime Vs Revenue)
runtime_column = df_movie['Runtime (Minutes)']
p = np.polyfit(runtime_column, revenue_column, 1)
y_fit = p[0] * runtime_column + p[1]
plt.plot(runtime_column, y_fit, color='red')

#plt.show()



# Scatter Plot between Year and Rating
plt.scatter(df_movie['Year'], df_movie['Rating'])
plt.title('Rating vs Year')
plt.xlabel('Year')
plt.ylabel('Rating')

# line of best fit (Year Vs Rating)
rating_column = df_movie['Rating']
p = np.polyfit(year_column, rating_column, 1)
y_fit = p[0] * year_column + p[1]
plt.plot(year_column, y_fit, color='red')

#plt.show()



# Scatter Plot between Runtime and Rating
plt.scatter(df_movie['Runtime (Minutes)'], df_movie['Rating'])
plt.title('Runtime (Minutes) vs Rating')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Rating')

# line of best fit (Runtime Vs Rating)
p = np.polyfit(runtime_column, rating_column, 1)
y_fit = p[0] * runtime_column + p[1]
plt.plot(runtime_column, y_fit, color='red')

#plt.show()


# Scatter Plot between Revenue and Rating
plt.scatter(df_movie['Rating'], df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Rating')
plt.xlabel('Rating')
plt.ylabel('Revenue (Millions)')

# line of best fit (Revenue Vs Rating)
p = np.polyfit(rating_column, revenue_column, 1)
y_fit = p[0] * rating_column + p[1]
plt.plot(rating_column, y_fit, color='red')

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

#REVENUE
#Created column for each genre so converted categorical values of Genre to numerical values
#Dummies Encoding
genres = df_movie["Genre"].str.get_dummies(",")
df_movie_genre_revenue = pd.concat([df_movie['Revenue (Millions)'], genres], axis=1)
#We dropped columns with negative correlation in the correlation heatmap
df_movie_genre_revenue = df_movie_genre_revenue.drop(columns=['Biography', 'Comedy', 'Crime', 'Family', 'History', 'Western', 'Music', 'Musical', 'Mystery', 'Sport','Thriller', 'War'])

#HeatMap of correlations between revenue and the movie genres
plt.figure(figsize=(16,6))
heatmap = sns.heatmap(df_movie_genre_revenue.corr(), vmin = -1, vmax = 1, annot = True)
heatmap.set_title("Correlation Heatmap", fontdict = {"fontsize":12}, pad = 12);
#plt.show()

#RATING
df_movie_genre_rating = pd.concat([df_movie['Rating'], genres], axis=1)
#We dropped columns with negative correlation in the correlation heatmap
df_movie_genre_rating = df_movie_genre_rating.drop(columns=['Crime', 'Adventure', 'Comedy', 'Family', 'Fantasy', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'])

#HeatMap of correlations between revenue and the movie genres
plt.figure(figsize=(16,6))
heatmap = sns.heatmap(df_movie_genre_rating.corr(), vmin = -1, vmax = 1, annot = True)
heatmap.set_title("Correlation Heatmap", fontdict = {"fontsize":12}, pad = 12);
#plt.show()

print(df_movie['Director'].nunique())

#DIRECTORS
director_occurences = df_movie['Director'].value_counts().sort_values(ascending=False)
print(director_occurences.head(20))
#https://towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173




