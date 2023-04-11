from cmath import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


df_movie = pd.read_csv("IMDB-Movie-Data.csv")

#Revenue and MetaScore are the only columns with missing entries,so we removed rows with the missing values

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
rating_column = df_movie['Rating']
runtime_column = df_movie['Runtime (Minutes)']
votes_column = df_movie['Votes']
p = np.polyfit(year_column, revenue_column, 1)
y_fit = p[0] * year_column + p[1]
plt.plot(year_column, y_fit, color='red')
#plt.show()

logged_revenue = np.log(df_movie['Revenue (Millions)'])
logged_runtime = np.log(df_movie['Runtime (Minutes)'])

# Scatter Plot between Runtime and Revenue
plt.scatter(df_movie['Runtime (Minutes)'], df_movie['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Runtime (Minutes)')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Revenue (Millions)')

# line of best fit (Runtime Vs Revenue)
p = np.polyfit(runtime_column, revenue_column, 1)
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

# Scatter Plot between Votes and Revenue
plt.scatter(df_movie['Votes'], df_movie['Revenue (Millions)'])
plt.title('Votes vs Runtime (Minutes)')
plt.xlabel('Votes')
plt.ylabel('Revenue (Millions)')

# line of best fit (Votes Vs Revenue)
p = np.polyfit(votes_column, revenue_column, 1)
y_fit = p[0] * runtime_column + p[1]
plt.plot(runtime_column, y_fit, color='red')

#plt.show()

# print the correlation
corr_rev_run = df_movie['Runtime (Minutes)'].corr(df_movie['Revenue (Millions)'])
print("Correlation between Runtime and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Year'].corr(df_movie['Revenue (Millions)'])
print("Correlation between year and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Rating'].corr(df_movie['Revenue (Millions)'])
print("Correlation between Rating and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Votes'].corr(df_movie['Revenue (Millions)'])
print("Correlation between Votes and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie['Metascore'].corr(df_movie['Revenue (Millions)'])
print("Correlation between Metascore and Revenue is: ", round(corr_rev_run, 2))

# -------------------Correlation HeatMaps for Categorical variables---------------------------------------------------
# Created columns for each unique element of Categorical variable by converting categorical values to numeric values
# Merged those values to the original data frame
# Machine learning technique - Dummies Encoding




# -----------REVENUE VS GENRE Correlation Heatmap-----------

# Dummies Encoding for Genre categorical variable
genres = df_movie["Genre"].str.get_dummies(",")
df_movie_genre_revenue = pd.concat([df_movie['Revenue (Millions)'], genres, df_movie['Rank']], axis=1)

# Dropped columns with negligible correlation (<0.1) in the Revenue vs Genre correlation heatmap

corr_matrix = df_movie_genre_revenue.corr()
heatmap_firstrow = corr_matrix.iloc[0].abs()
cols_to_drop = heatmap_firstrow[heatmap_firstrow < 0.1].index
df_movie_genre_revenue = df_movie_genre_revenue.drop(cols_to_drop, axis=1)

# Display Revenue vs Genre HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_genre_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("REVENUE VS GENRE Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
#plt.show()

# -----------REVENUE VS DIRECTOR Correlation HeatMap----------------

# Found Top 20 Directors with highest occurences, sorted in descending order
director_occurences = df_movie.groupby('Director').size().reset_index(name='Occurence')
top_20_directors = director_occurences.sort_values('Occurence', ascending=False).head(20)

# Dummies Encoding for Top 20 Director categorical variable
df_movie_directors_revenue = pd.concat([df_movie['Revenue (Millions)'], df_movie['Director'], df_movie['Rank']], axis=1)

for director in top_20_directors['Director']:
    df_movie_directors_revenue[director] = df_movie_directors_revenue['Director'].apply(
        lambda x: 1 if director in x else 0)

df_movie_directors_revenue = df_movie_directors_revenue.drop('Director', axis=1)

# Dropped columns with negligible correlation (<0.1) in the Revenue vs Director correlation heatmap

corr_matrix = df_movie_directors_revenue.corr()
heatmap_firstrow = corr_matrix.iloc[0].abs()
cols_to_drop = heatmap_firstrow[heatmap_firstrow < 0.1].index
df_movie_directors_revenue = df_movie_directors_revenue.drop(cols_to_drop, axis=1)

# Display Revenue vs Director HeatMap
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_directors_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("REVENUE VS DIRECTORS Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
#plt.show()


# -----------REVENUE VS ACTORS Correlation HeatMap----------------

# Found Top 20 Actors with highest occurences, sorted in descending order
actor_occurences = {}
df_movie_actor_revenue = pd.concat([df_movie['Revenue (Millions)'], df_movie['Actors'], df_movie['Rank'], df_movie['Rating'],
                                    df_movie['Votes'], df_movie['Runtime (Minutes)']], axis=1)

for index, row in df_movie_actor_revenue.iterrows():
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
    df_movie_actor_revenue[actor] = df_movie_actor_revenue['Actors'].apply(lambda x: 1 if actor in x else 0)

df_movie_actor_revenue = df_movie_actor_revenue.drop('Actors', axis=1)

# Dropped columns with negligible correlation (<0.1) in the Revenue vs Actors correlation heatmap
corr_matrix = df_movie_actor_revenue.corr()
heatmap_firstrow = corr_matrix.iloc[0].abs()
cols_to_drop = heatmap_firstrow[heatmap_firstrow < 0.1].index
df_movie_actor_revenue = df_movie_actor_revenue.drop(cols_to_drop, axis=1)

# Display Revenue vs Actor HeatMap
#plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_movie_actor_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("REVENUE VS ACTORS Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
#plt.show()

# -------------------Scatter Plots for categorical variables---------------------------------------------------

# Scatter Plot between Genre Action and Revenue
plt.bar(df_movie_genre_revenue['Action'], df_movie_genre_revenue['Revenue (Millions)'])
plt.title('Genre Action vs Runtime (Minutes)')
plt.xlabel('Genre Action')
plt.ylabel('Revenue (Millions)')
#plt.show()

# -------------------Random Forest Model---------------------------------------------------

predictors_revenue = pd.merge(df_movie_genre_revenue, df_movie_actor_revenue)
predictors_revenue = pd.merge(predictors_revenue, df_movie_directors_revenue)

# Dropped columns with weak correlation (<0.2)
corr_matrix = predictors_revenue.corr()
heatmap_firstrow = corr_matrix.iloc[0].abs()
cols_to_drop = heatmap_firstrow[heatmap_firstrow < 0.2].index
predictors_revenue = predictors_revenue.drop(cols_to_drop, axis=1)
predictors_revenue = predictors_revenue.drop('Rank', axis=1)
#predictors_revenue = predictors_revenue.drop('Rating', axis=1) #To fix overfitting
#predictors = predictors.drop('Runtime (Minutes)', axis=1) #No removing this
#predictors = predictors.drop('Drama', axis=1) #To fix overfitting
#predictors = predictors.drop('Votes', axis=1) #To fix overfitting

heatmap = sns.heatmap(predictors_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title("Predictors Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
#plt.show()

# Split the data into independant variables and dependant variable
independant_variables = predictors_revenue.drop('Revenue (Millions)', axis=1)
dependant_variable = predictors_revenue['Revenue (Millions)']

# Split the data into training and test sets
independant_train, independant_test, dependant_train, dependant_test = train_test_split(independant_variables, dependant_variable, test_size=0.2)

randomForestModel = RandomForestRegressor()
randomForestModel.fit(independant_train, dependant_train)

predict_dependant_train = randomForestModel.predict(independant_train)
predict_dependant_test = randomForestModel.predict(independant_test)

mean_squared_error = mean_squared_error(dependant_test, predict_dependant_test)
r_squared_train = r2_score(dependant_train, predict_dependant_train)
r_squared_test = r2_score(dependant_test, predict_dependant_test)
adj_r_squared_train = 1 - ((1 - r_squared_train) * (838 - 1) / (838 - len(independant_variables.columns) - 1))
adj_r_squared_test = 1 - ((1 - r_squared_test) * (838 - 1) / (838 - len(independant_variables.columns) - 1))

print("Root Mean Squared Error: ", sqrt(mean_squared_error))
print("Train Adjusted R Squared", adj_r_squared_train)
print("Test Adjusted R Squared", adj_r_squared_test)

#RESIDUAL PLOT
residuals = dependant_test - predict_dependant_test

# Scatter plot of the residuals against the predicted values
plt.scatter(predict_dependant_test, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
#plt.show()