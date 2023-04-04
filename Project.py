import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

df_movie = pd.read_csv("IMDB-Movie-Data.csv")

# we found that Revenue and MetaScore are the only columns with missing entries,so we removed rows with the missing values

df_movie = df_movie.dropna(axis=0, how='any')
# df_movie.info()


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

# plt.show()


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

# plt.show()


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

# plt.show()


# Scatter Plot between Runtime and Rating
plt.scatter(df_movie['Runtime (Minutes)'], df_movie['Rating'])
plt.title('Runtime (Minutes) vs Rating')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Rating')

# line of best fit (Runtime Vs Rating)
p = np.polyfit(runtime_column, rating_column, 1)
y_fit = p[0] * runtime_column + p[1]
plt.plot(runtime_column, y_fit, color='red')

# plt.show()


# Scatter Plot between Revenue and Rating
plt.scatter(df_movie['Rating'], df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Rating')
plt.xlabel('Rating')
plt.ylabel('Revenue (Millions)')

# line of best fit (Revenue Vs Rating)
p = np.polyfit(rating_column, revenue_column, 1)
y_fit = p[0] * rating_column + p[1]
plt.plot(rating_column, y_fit, color='red')

# plt.show()


# print the correlation
corr_rev_run = df_movie['Runtime (Minutes)'].corr(df_movie['Revenue (Millions)'])
print("Correlation between Runtime and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Year'].corr(df_movie['Revenue (Millions)'])
print("Correlation between year and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Runtime (Minutes)'].corr(df_movie['Rating'])
print("Correlation between Runtime and Rating is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Year'].corr(df_movie['Rating'])
print("Correlation between Year and Rating is: ", round(corr_rev_run, 2))

# -------------------Correlation HeatMaps for Categorical variables---------------------------------------------------
# Created columns for each unique element of Categorical variable by converting categorical values to numeric values
# Merged those values to the original data frame
# Machine learning technique - Dummies Encoding


# -----------REVENUE VS GENRE Correlation Heatmap-----------

# Dummies Encoding for Genre categorical variable
genres = df_movie["Genre"].str.get_dummies(",")
df_movie_genre_revenue = pd.concat([df_movie['Revenue (Millions)'], genres], axis=1)

# Dropped columns with negligible correlation in the Genre vs Revenue correlation heatmap
df_movie_genre_revenue = df_movie_genre_revenue.drop(
    columns=['Biography', 'Comedy', 'Crime', 'Family', 'History', 'Western', 'Music', 'Musical', 'Mystery', 'Sport',
             'Thriller', 'War'])

# Display Revenue vs Genre HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_genre_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("REVENUE VS GENRE Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
# plt.show()


# -----------RATING VS GENRE Correlation HeatMap-------------

df_movie_genre_rating = pd.concat([df_movie['Rating'], genres], axis=1)

# Dropped columns with negligible correlation in the Genre vs Rating correlation heatmap
df_movie_genre_rating = df_movie_genre_rating.drop(
    columns=['Crime', 'Adventure', 'Comedy', 'Family', 'Fantasy', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
             'Sport', 'Thriller', 'War', 'Western'])

# Display Rating vs Genre HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_genre_rating.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("RATING VS GENRE Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
# plt.show()


# -----------REVENUE VS DIRECTOR Correlation HeatMap----------------

# Found Top 20 Directors with highest occurences, sorted in descending order
director_occurences = df_movie.groupby('Director').size().reset_index(name='Occurence')
top_20_directors = director_occurences.sort_values('Occurence', ascending=False).head(20)

# Dummies Encoding for Top 20 Director categorical variable
df_movie_directors_revenue = pd.concat([df_movie['Revenue (Millions)'], df_movie['Director']], axis=1)

for director in top_20_directors['Director']:
    df_movie_directors_revenue[director] = df_movie_directors_revenue['Director'].apply(
        lambda x: 1 if director in x else 0)

# Dropped columns with negligible correlation in the Revenue vs Director correlation heatmap
df_movie_directors = df_movie_directors_revenue.drop('Director', axis=1)
print(sum(df_movie_directors['David Yates'] == 1))

# Display Revenue vs Director HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_directors_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
plt.show()
