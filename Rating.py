import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

df_movie = pd.read_csv("IMDB-Movie-Data.csv")

# we found that Revenue and MetaScore are the only columns with missing entries,so we removed rows with the missing values

df_movie = df_movie.dropna(axis=0, how='any')
# df_movie.info()


# Scatter Plot between Year and Rating
plt.scatter(df_movie['Year'], df_movie['Rating'])
plt.title('Rating vs Year')
plt.xlabel('Year')
plt.ylabel('Rating')

# line of best fit (Year Vs Rating)
year_column = df_movie['Year']
rating_column = df_movie['Rating']
runtime_column = df_movie['Runtime (Minutes)']

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



corr_rev_run = df_movie['Runtime (Minutes)'].corr(df_movie['Rating'])
print("Correlation between Runtime and Rating is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Year'].corr(df_movie['Rating'])
print("Correlation between Year and Rating is: ", round(corr_rev_run, 2))




# -------------------Correlation HeatMaps for Categorical variables---------------------------------------------------
# Created columns for each unique element of Categorical variable by converting categorical values to numeric values
# Merged those values to the original data frame
# Machine learning technique - Dummies Encoding

# -----------RATING VS GENRE Correlation HeatMap-------------
genres = df_movie["Genre"].str.get_dummies(",")

df_movie_genre_rating = pd.concat([df_movie['Rating'], genres], axis=1)

# Dropped columns with negligible correlation (<0.1) in the Genre vs Rating correlation heatmap

corr_matrix = df_movie_genre_rating.corr()
heatmap_firstrow_rating_genre = corr_matrix.iloc[0].abs()
cols_to_drop_rating_genre = heatmap_firstrow_rating_genre[heatmap_firstrow_rating_genre < 0.1].index
df_movie_genre_rating = df_movie_genre_rating.drop(cols_to_drop_rating_genre, axis=1)

""""
corr_matrix_rating_genre = df_movie_genre_rating.corr()
heatmap_firstrow_rating_genre = corr_matrix_rating_genre.iloc[0].abs()
cols_to_drop_rating_genre = heatmap_firstrow_rating_genre[heatmap_firstrow_rating_genre < 0.1].index
df_movie_genre_rating = df_movie_genre_rating.drop(cols_to_drop_rating_genre, axis=1)
"""

# Display Rating vs Genre HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_genre_rating.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("RATING VS GENRE Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
plt.show()



# -----------RATING VS DIRECTOR Correlation HeatMap----------------
# Found Top 20 Directors with highest occurences, sorted in descending order

director_occurences = df_movie.groupby('Director').size().reset_index(name='Occurence')
top_20_directors = director_occurences.sort_values('Occurence', ascending=False).head(20)

# Dummies Encoding for Top 20 Director categorical variable
df_movie_directors_rating = pd.concat([df_movie['Rating'], df_movie['Director']], axis=1)

for director in top_20_directors['Director']:
    df_movie_directors_rating[director] = df_movie_directors_rating['Director'].apply(
        lambda x: 1 if director in x else 0)

# Dropped columns with negligible correlation (<0.1) in the Rating vs Director correlation heatmap
df_movie_directors = df_movie_directors_rating.drop('Director', axis=1)

corr_matrix = df_movie_directors.corr()
heatmap_firstrow_rating_genre = corr_matrix.iloc[0].abs()
cols_to_drop_rating_genre = heatmap_firstrow_rating_genre[heatmap_firstrow_rating_genre < 0.1].index
df_movie_genre_rating = df_movie_genre_rating.drop(cols_to_drop_rating_genre, axis=1)

#corr_matrix_rating_director = df_movie_directors.corr()
#heatmap_firstrow_rating_genre = corr_matrix_rating_genre.iloc[0].abs()
#cols_to_drop_rating_genre = heatmap_firstrow_rating_genre[heatmap_firstrow_rating_genre < 0.1].index
#df_movie_genre_rating = df_movie_genre_rating.drop(cols_to_drop_rating_genre, axis=1)




# Display Rating vs Director HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_directors_rating.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("RATING VS DIRECTOR Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
plt.show()
