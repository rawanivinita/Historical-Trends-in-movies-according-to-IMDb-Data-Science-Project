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


df_movie_revenue = pd.read_csv("IMDB-Movie-Data.csv")

#Revenue and MetaScore are the only columns with missing entries,so we removed rows with the missing values

df_movie_revenue = df_movie_revenue.dropna(axis=0, how='any')
# df_movie.info()

# Scatter Plot between Year and Revenue
plt.scatter(df_movie_revenue['Year'], df_movie_revenue['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Year')
plt.xlabel('Year')
plt.ylabel('Revenue (Millions)')

# line of best fit (Year Vs Revenue)
year_column_revenue = df_movie_revenue['Year']
revenue_column_revenue = df_movie_revenue['Revenue (Millions)']
rating_column_revenue = df_movie_revenue['Rating']
runtime_column_revenue = df_movie_revenue['Runtime (Minutes)']
votes_column_revenue = df_movie_revenue['Votes']
p_revenue = np.polyfit(year_column_revenue, revenue_column_revenue, 1)
y_fit_revenue = p_revenue[0] * year_column_revenue + p_revenue[1]
plt.plot(year_column_revenue, y_fit_revenue, color='red')
#plt.show()

logged_revenue = np.log(df_movie_revenue['Revenue (Millions)'])
logged_runtime = np.log(df_movie_revenue['Runtime (Minutes)'])

# Scatter Plot between Runtime and Revenue
plt.scatter(df_movie_revenue['Runtime (Minutes)'], df_movie_revenue['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Runtime (Minutes)')
plt.xlabel('Runtime (Minutes)')
plt.ylabel('Revenue (Millions)')

# line of best fit (Runtime Vs Revenue)
p_revenue = np.polyfit(runtime_column_revenue, revenue_column_revenue, 1)
y_fit_revenue = p_revenue[0] * runtime_column_revenue + p_revenue[1]
plt.plot(runtime_column_revenue, y_fit_revenue, color='red')

#plt.show()

# Scatter Plot between Revenue and Rating
plt.scatter(df_movie_revenue['Rating'], df_movie_revenue['Revenue (Millions)'])
plt.title('Revenue (Millions) vs Rating')
plt.xlabel('Rating')
plt.ylabel('Revenue (Millions)')

# line of best fit (Revenue Vs Rating)
p_revenue = np.polyfit(rating_column_revenue, revenue_column_revenue, 1)
y_fit_revenue = p_revenue[0] * rating_column_revenue + p_revenue[1]
plt.plot(rating_column_revenue, y_fit_revenue, color='red')

#plt.show()

# Scatter Plot between Votes and Revenue
plt.scatter(df_movie_revenue['Votes'], df_movie_revenue['Revenue (Millions)'])
plt.title('Votes vs Runtime (Minutes)')
plt.xlabel('Votes')
plt.ylabel('Revenue (Millions)')

# line of best fit (Votes Vs Revenue)
p_revenue = np.polyfit(votes_column_revenue, revenue_column_revenue, 1)
y_fit_revenue = p_revenue[0] * runtime_column_revenue + p_revenue[1]
plt.plot(runtime_column_revenue, y_fit_revenue, color='red')

#plt.show()

# print the correlation
corr_rev_run = df_movie_revenue['Runtime (Minutes)'].corr(df_movie_revenue['Revenue (Millions)'])
#print("Correlation between Runtime and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie_revenue['Year'].corr(df_movie_revenue['Revenue (Millions)'])
#print("Correlation between year and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie_revenue['Rating'].corr(df_movie_revenue['Revenue (Millions)'])
#print("Correlation between Rating and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie_revenue['Votes'].corr(df_movie_revenue['Revenue (Millions)'])
#print("Correlation between Votes and Revenue is: ", round(corr_rev_run, 2))

corr_rev_run = df_movie_revenue['Metascore'].corr(df_movie_revenue['Revenue (Millions)'])
#print("Correlation between Metascore and Revenue is: ", round(corr_rev_run, 2))

# -------------------Correlation HeatMaps for Categorical variables---------------------------------------------------
# Created columns for each unique element of Categorical variable by converting categorical values to numeric values
# Merged those values to the original data frame
# Machine learning technique - Dummies Encoding




# -----------REVENUE VS GENRE Correlation Heatmap-----------

# Dummies Encoding for Genre categorical variable
genres_revenue = df_movie_revenue["Genre"].str.get_dummies(",")
df_movie_genre_revenue = pd.concat([df_movie_revenue['Revenue (Millions)'], genres_revenue, df_movie_revenue['Rank']], axis=1)

# Dropped columns with negligible correlation (<0.1) in the Revenue vs Genre correlation heatmap

corr_matrix_revenue = df_movie_genre_revenue.corr()
heatmap_firstrow_revenue = corr_matrix_revenue.iloc[0].abs()
cols_to_drop_revenue = heatmap_firstrow_revenue[heatmap_firstrow_revenue < 0.1].index
df_movie_genre_revenue = df_movie_genre_revenue.drop(cols_to_drop_revenue, axis=1)

# Display Revenue vs Genre HeatMap
plt.figure(figsize=(16, 6))
heatmap_revenue = sns.heatmap(df_movie_genre_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap_revenue.set_title("REVENUE VS GENRE Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
#plt.show()

# -----------REVENUE VS DIRECTOR Correlation HeatMap----------------

# Found Top 20 Directors with highest occurences, sorted in descending order
director_occurences_revenue = df_movie_revenue.groupby('Director').size().reset_index(name='Occurence')
top_20_directors_revenue = director_occurences_revenue.sort_values('Occurence', ascending=False).head(20)

# Dummies Encoding for Top 20 Director categorical variable
df_movie_directors_revenue = pd.concat([df_movie_revenue['Revenue (Millions)'], df_movie_revenue['Director'], df_movie_revenue['Rank']], axis=1)

for director in top_20_directors_revenue['Director']:
    df_movie_directors_revenue[director] = df_movie_directors_revenue['Director'].apply(
        lambda x: 1 if director in x else 0)

df_movie_directors_revenue = df_movie_directors_revenue.drop('Director', axis=1)

# Dropped columns with negligible correlation (<0.1) in the Revenue vs Director correlation heatmap

corr_matrix_revenue = df_movie_directors_revenue.corr()
heatmap_firstrow_revenue = corr_matrix_revenue.iloc[0].abs()
cols_to_drop_revenue = heatmap_firstrow_revenue[heatmap_firstrow_revenue < 0.1].index
df_movie_directors_revenue = df_movie_directors_revenue.drop(cols_to_drop_revenue, axis=1)

# Display Revenue vs Director HeatMap
plt.figure(figsize=(16, 6))
heatmap_revenue = sns.heatmap(df_movie_directors_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap_revenue.set_title("REVENUE VS DIRECTORS Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
#plt.show()


# -----------REVENUE VS ACTORS Correlation HeatMap----------------

# Found Top 20 Actors with highest occurences, sorted in descending order
actor_occurences_revenue = {}
df_movie_actor_revenue = pd.concat([df_movie_revenue['Revenue (Millions)'], df_movie_revenue['Actors'], df_movie_revenue['Rank'], df_movie_revenue['Rating'],
                                    df_movie_revenue['Votes'], df_movie_revenue['Runtime (Minutes)']], axis=1)

for index, row in df_movie_actor_revenue.iterrows():
    actors_revenue = row['Actors'].split(', ')
    for actor in actors_revenue:
        if actor in actor_occurences_revenue:
            actor_occurences_revenue[actor] += 1
        else:
            actor_occurences_revenue[actor] = 1

top_actors_revenue = sorted(actor_occurences_revenue.items(), key=lambda x: x[1], reverse=True)
top_20_actors_revenue = [x[0] for x in top_actors_revenue[:20]]

# Dummies Encoding for Top 20 Actors categorical variable
for actor in top_20_actors_revenue:
    df_movie_actor_revenue[actor] = df_movie_actor_revenue['Actors'].apply(lambda x: 1 if actor in x else 0)

df_movie_actor_revenue = df_movie_actor_revenue.drop('Actors', axis=1)

# Dropped columns with negligible correlation (<0.1) in the Revenue vs Actors correlation heatmap
corr_matrix_revenue = df_movie_actor_revenue.corr()
heatmap_firstrow_revenue = corr_matrix_revenue.iloc[0].abs()
cols_to_drop_revenue = heatmap_firstrow_revenue[heatmap_firstrow_revenue < 0.1].index
df_movie_actor_revenue = df_movie_actor_revenue.drop(cols_to_drop_revenue, axis=1)

# Display Revenue vs Actor HeatMap
#plt.figure(figsize=(16, 6))
heatmap_revenue = sns.heatmap(df_movie_actor_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap_revenue.set_title("REVENUE VS ACTORS Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
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
corr_matrix_revenue = predictors_revenue.corr()
heatmap_firstrow_revenue = corr_matrix_revenue.iloc[0].abs()
cols_to_drop_revenue = heatmap_firstrow_revenue[heatmap_firstrow_revenue < 0.2].index
predictors_revenue = predictors_revenue.drop(cols_to_drop_revenue, axis=1)
predictors_revenue = predictors_revenue.drop('Rank', axis=1)
predictors_revenue = predictors_revenue.drop('Runtime (Minutes)', axis=1) #To fix overfitting

heatmap_revenue = sns.heatmap(predictors_revenue.corr(), vmin=-1, vmax=1, annot=True)
heatmap_revenue.set_title("Predictors Correlation Heatmap", fontdict={"fontsize": 12}, pad=12);
#plt.show()

# Split the data into independant variables and dependant variable
independant_variables_revenue = predictors_revenue.drop('Revenue (Millions)', axis=1)
dependant_variable_revenue = predictors_revenue['Revenue (Millions)']

# Split the data into training and test sets
independant_train_revenue, independant_test_revenue, dependant_train_revenue, dependant_test_revenue = train_test_split(independant_variables_revenue, dependant_variable_revenue, test_size=0.2)

randomForestModel_revenue = RandomForestRegressor()
randomForestModel_revenue.fit(independant_train_revenue, dependant_train_revenue)

predict_dependant_train_revenue = randomForestModel_revenue.predict(independant_train_revenue)
predict_dependant_test_revenue = randomForestModel_revenue.predict(independant_test_revenue)

mean_squared_error_revenue = mean_squared_error(dependant_test_revenue, predict_dependant_test_revenue)
r_squared_train_revenue = r2_score(dependant_train_revenue, predict_dependant_train_revenue)
r_squared_test_revenue = r2_score(dependant_test_revenue, predict_dependant_test_revenue)


adj_r_squared_train_revenue = 1 - ((1 - r_squared_train_revenue) * (838 - 1) / (838 - len(independant_variables_revenue.columns) - 1))
adj_r_squared_test_revenue = 1 - ((1 - r_squared_test_revenue) * (838 - 1) / (838 - len(independant_variables_revenue.columns) - 1))

print("Mean Squared Error: ", mean_squared_error_revenue)
print("Root Mean Squared Error: ", sqrt(mean_squared_error_revenue))
print("Train Adjusted R Squared", adj_r_squared_train_revenue)
print("Test Adjusted R Squared", adj_r_squared_test_revenue)


#RESIDUAL PLOT
residuals_revenue = dependant_test_revenue - predict_dependant_test_revenue

# Scatter plot of the residuals against the predicted values
plt.scatter(predict_dependant_test_revenue, residuals_revenue)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
#plt.show()