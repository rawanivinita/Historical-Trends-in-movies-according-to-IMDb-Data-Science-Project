from cmath import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

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
revenue_column = df_movie['Revenue (Millions)']
votes_column = df_movie['Votes']
metascore_column = df_movie['Metascore']

p = np.polyfit(year_column, rating_column, 1)
y_fit = p[0] * year_column + p[1]
plt.plot(year_column, y_fit, color='red')

plt.show()


# Scatter Plot between Runtime and Rating
plt.scatter(df_movie['Runtime (Minutes)'], df_movie['Rating'])
plt.title('Runtime (Minutes) vs Rating')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Rating')

# line of best fit (Runtime Vs Rating)
p = np.polyfit(runtime_column, rating_column, 1)
y_fit = p[0] * runtime_column + p[1]
plt.plot(runtime_column, y_fit, color='red')

plt.show()

# Scatter Plot between Revenue and Rating
plt.scatter(df_movie['Rating'], df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Rating')
plt.xlabel('Rating')
plt.ylabel('Revenue (Millions)')

# line of best fit (Revenue Vs Rating)
p = np.polyfit(rating_column, revenue_column, 1)
y_fit = p[0] * rating_column + p[1]
plt.plot(rating_column, y_fit, color='red')

plt.show()

# Scatter Plot between Votes and Rating
plt.scatter(df_movie['Votes'], df_movie['Rating'])
plt.title('Votes vs Rating')
plt.xlabel('Votes')
plt.ylabel('Rating')

# line of best fit (Votes Vs Rating)
p = np.polyfit(votes_column, rating_column, 1)
y_fit = p[0] * rating_column + p[1]
plt.plot(rating_column, y_fit, color='red')

#plt.show()

# Scatter Plot between Metascore and Rating
plt.scatter(df_movie['Metascore'], df_movie['Rating'])
plt.title('Rating vs Metascore')
plt.xlabel('Metascore')
plt.ylabel('Rating')

# line of best fit (Metascore Vs Rating)
rating_column = df_movie['Rating']
p = np.polyfit(metascore_column, rating_column, 1)
y_fit = p[0] * metascore_column + p[1]
plt.plot(metascore_column, y_fit, color='red')

plt.show()



corr_rev_run = df_movie['Runtime (Minutes)'].corr(df_movie['Rating'])
print("Correlation between Runtime and Rating is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Year'].corr(df_movie['Rating'])
print("Correlation between Year and Rating is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Votes'].corr(df_movie['Rating'])
print("Correlation between Votes and Rating is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Metascore'].corr(df_movie['Rating'])
print("Correlation between Metascore and Rating is: ", round(corr_rev_run, 2))


# -------------------Correlation HeatMaps for Categorical variables---------------------------------------------------
# Created columns for each unique element of Categorical variable by converting categorical values to numeric values
# Merged those values to the original data frame
# Machine learning technique - Dummies Encoding

# -----------RATING VS GENRE Correlation HeatMap-------------
genres = df_movie["Genre"].str.get_dummies(",")

df_movie_genre_rating = pd.concat([df_movie['Rating'], genres, df_movie['Rank']], axis=1)

# Dropped columns with negligible correlation (<0.1) in the Genre vs Rating correlation heatmap

corr_matrix_genre_rating = df_movie_genre_rating.corr()
heatmap_firstrow_genre_rating = corr_matrix_genre_rating.iloc[0].abs()
cols_to_drop = heatmap_firstrow_genre_rating[heatmap_firstrow_genre_rating < 0.1].index
df_movie_genre_rating = df_movie_genre_rating.drop(cols_to_drop, axis=1)

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
df_movie_directors_rating = pd.concat([df_movie['Rating'], df_movie['Director'], df_movie['Rank']], axis=1)

for director in top_20_directors['Director']:
    df_movie_directors_rating[director] = df_movie_directors_rating['Director'].apply(
        lambda x: 1 if director in x else 0)

# Dropped columns with negligible correlation (<0.1) in the Rating vs Director correlation heatmap
df_movie_directors_rating = df_movie_directors_rating.drop('Director', axis=1)

corr_matrix_directors_rating = df_movie_directors_rating.corr()
heatmap_firstrow_directors_rating = corr_matrix_directors_rating.iloc[0].abs()
cols_to_drop = heatmap_firstrow_directors_rating[heatmap_firstrow_directors_rating < 0.1].index
df_movie_directors_rating = df_movie_directors_rating.drop(cols_to_drop, axis=1)

#corr_matrix_rating_director = df_movie_directors.corr()
#heatmap_firstrow_rating_genre = corr_matrix_rating_genre.iloc[0].abs()
#cols_to_drop_rating_genre = heatmap_firstrow_rating_genre[heatmap_firstrow_rating_genre < 0.1].index
#df_movie_genre_rating = df_movie_genre_rating.drop(cols_to_drop_rating_genre, axis=1)

# Display Rating vs Director HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_directors_rating.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("RATING VS DIRECTOR Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
plt.show()


# -----------RATING VS ACTORS Correlation HeatMap----------------

# Found Top 20 Actors with highest occurences, sorted in descending order
actor_occurences = {}
df_movie_actor_rating = pd.concat([df_movie['Rating'], df_movie['Actors'], df_movie['Rank'], df_movie['Revenue (Millions)'],
                                    df_movie['Votes'], df_movie['Runtime (Minutes)'], df_movie['Metascore']], axis=1)

for index, row in df_movie_actor_rating.iterrows():
    actors = row['Actors'].split(', ')
    for actor in actors:
        if actor in actor_occurences:
            actor_occurences[actor] += 1
        else:
            actor_occurences[actor] = 1

top_actors = sorted(actor_occurences.items(), key=lambda x: x[1], reverse=True)
top_20_actors = [x[0] for x in top_actors[:20]]

# Dummies Encoding for Top 20 Actors categorical variable
for actor in top_20_actors:
    df_movie_actor_rating[actor] = df_movie_actor_rating['Actors'].apply(lambda x: 1 if actor in x else 0)

df_movie_actor_rating = df_movie_actor_rating.drop('Actors', axis=1)

# Dropped columns with negligible correlation (<0.1) in the Revenue vs Actors correlation heatmap
corr_matrix_actor_rating = df_movie_actor_rating.corr()
heatmap_firstrow_actor_rating = corr_matrix_actor_rating.iloc[0].abs()
cols_to_drop = heatmap_firstrow_actor_rating[heatmap_firstrow_actor_rating < 0.1].index
df_movie_actor_rating = df_movie_actor_rating.drop(cols_to_drop, axis=1)

# Display Revenue vs Actor HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_actor_rating.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("RATING VS ACTORS Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
plt.show()

# -------------------Random Forest Model---------------------------------------------------

predictors_rating = pd.merge(df_movie_genre_rating, df_movie_actor_rating)
predictors_rating = pd.merge(predictors_rating, df_movie_directors_rating)

# Dropped columns with weak correlation (<0.2)
corr_matrix_rating = predictors_rating.corr()
heatmap_firstrow_rating = corr_matrix_rating.iloc[0].abs()
cols_to_drop = heatmap_firstrow_rating[heatmap_firstrow_rating < 0.2].index
predictors_rating = predictors_rating.drop(cols_to_drop, axis=1)
predictors_rating = predictors_rating.drop('Rank', axis=1)
predictors_rating = predictors_rating.drop('Votes', axis=1) #To reduce overfitting

print(predictors_rating.columns)

# Split the data into independant variables and dependant variable
independant_variables_rating = predictors_rating.drop('Rating', axis=1)
dependant_variable_rating = predictors_rating['Rating']

# Split the data into training and test sets
independant_train_rating, independant_test_rating, dependant_train_rating, dependant_test_rating = train_test_split(independant_variables_rating, dependant_variable_rating, test_size=0.2)

randomForestModel = RandomForestRegressor()
randomForestModel.fit(independant_train_rating, dependant_train_rating)

predict_dependant_train_rating = randomForestModel.predict(independant_train_rating)
predict_dependant_test_rating = randomForestModel.predict(independant_test_rating)

mean_squared_error_rating = mean_squared_error(dependant_test_rating, predict_dependant_test_rating)
r_squared_train_rating = r2_score(dependant_train_rating, predict_dependant_train_rating)
r_squared_test_rating = r2_score(dependant_test_rating, predict_dependant_test_rating)
adj_r_squared_train_rating = 1 - ((1 - r_squared_train_rating) * (838 - 1) / (838 - len(independant_variables_rating.columns) - 1))
adj_r_squared_test_rating = 1 - ((1 - r_squared_test_rating) * (838 - 1) / (838 - len(independant_variables_rating.columns) - 1))

print("Root Mean Squared Error: ", sqrt(mean_squared_error_rating))
print("Train Adjusted R Squared", adj_r_squared_train_rating)
print("Test Adjusted R Squared", adj_r_squared_test_rating)

#RESIDUAL PLOT
residuals_rating = dependant_test_rating - predict_dependant_test_rating

# Scatter plot of the residuals against the predicted values
plt.scatter(predict_dependant_test_rating, residuals_rating)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.show()