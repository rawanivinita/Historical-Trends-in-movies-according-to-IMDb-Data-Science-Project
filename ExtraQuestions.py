from cmath import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

df_movie = pd.read_csv("IMDB-Movie-Data.csv")
df_movie = df_movie.dropna(axis=0, how='any')
X = df_movie['Rank']
Y = df_movie['Votes']

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the function to the data
popt, pcov = curve_fit(func, X, Y, maxfev=10000)

# Scatter Plot between Year and Revenue
plt.plot(X, func(X, *popt), 'r-', label='Fit')
plt.scatter(X,Y)
plt.title('Rank vs Number of Votes')
plt.xlabel('Rank')
plt.ylabel('Number of Votes')
#plt.show()

# Found Top 20 Actors with highest occurences, sorted in descending order
actor_occurences = {}

for index, row in df_movie.iterrows():
    actors = row['Actors'].split(', ')
    for actor in actors:
        if actor in actor_occurences:
            actor_occurences[actor] += 1
        else:
            actor_occurences[actor] = 1

top_actors = sorted(actor_occurences.items(), key=lambda x: x[1], reverse=True)
top_20_actors = [x[0] for x in top_actors[:20]]

#print(top_20_actors)
#Mark Wahlberg is first on this list while the first female (Emma Stone) is 11th
chosen_actors = ['Mark Wahlberg', 'Hugh Jackman', 'Brad Pitt',  'Cate Blanchett', 'Emma Stone', 'Anna Kendrick']

for actor in chosen_actors:
    df_movie[actor] = df_movie['Actors'].apply(lambda x: 1 if actor in x else 0)

num_of_movies = {}

for actor in chosen_actors:
    num_of_movies[actor] = num_of_movies.get(actor, 0) + sum(df_movie[actor] == 1)

print(num_of_movies)

genres = df_movie["Genre"].str.get_dummies(",")
df_movie_emma_mark = pd.concat([df_movie, genres], axis=1)

#print(df_movie_emma_mark.columns)

df_movie_emma_mark = df_movie_emma_mark[(df_movie_emma_mark['Mark Wahlberg'] == 1) | (df_movie_emma_mark['Hugh Jackman'] == 1)
                                        | (df_movie_emma_mark['Brad Pitt'] == 1) | (df_movie_emma_mark['Cate Blanchett'] == 1)
                                        | (df_movie_emma_mark['Emma Stone'] == 1) | (df_movie_emma_mark['Anna Kendrick'] == 1)]

mark_wahlberg_genres = {}
hugh_jackman_genres = {}
brad_pitt_genres = {}
cate_blanchett_genres = {}
emma_stone_genres = {}
anna_kendrick_genres = {}

genre_dicts = []
genre_dicts.append(mark_wahlberg_genres)
genre_dicts.append(hugh_jackman_genres)
genre_dicts.append(brad_pitt_genres)
genre_dicts.append(cate_blanchett_genres)
genre_dicts.append(emma_stone_genres)
genre_dicts.append(anna_kendrick_genres)

for genre in genres.columns:
    mark_wahlberg_genres[genre] = mark_wahlberg_genres.get(genre, 0) + sum((df_movie_emma_mark['Mark Wahlberg'] == 1) & (df_movie_emma_mark[genre] == 1))
    hugh_jackman_genres[genre] = hugh_jackman_genres.get(genre, 0) + sum((df_movie_emma_mark['Hugh Jackman'] == 1) & (df_movie_emma_mark[genre] == 1))
    brad_pitt_genres[genre] = brad_pitt_genres.get(genre, 0) + sum((df_movie_emma_mark['Brad Pitt'] == 1) & (df_movie_emma_mark[genre] == 1))
    cate_blanchett_genres[genre] = cate_blanchett_genres.get(genre, 0) + sum((df_movie_emma_mark['Cate Blanchett'] == 1) & (df_movie_emma_mark[genre] == 1))
    emma_stone_genres[genre] = emma_stone_genres.get(genre, 0) + sum((df_movie_emma_mark['Emma Stone'] == 1) & (df_movie_emma_mark[genre] == 1))
    anna_kendrick_genres[genre] = anna_kendrick_genres.get(genre, 0) + sum((df_movie_emma_mark['Anna Kendrick'] == 1) & (df_movie_emma_mark[genre] == 1))

mark_wahlberg_genres = dict(sorted(mark_wahlberg_genres.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------mark_wahlberg_genres:")
for genre, count in mark_wahlberg_genres.items():
    print(f"{genre}: {count/15}")

hugh_jackman_genres = dict(sorted(hugh_jackman_genres.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------hugh_jackman_genres:")
for genre, count in hugh_jackman_genres.items():
    print(f"{genre}: {count/14}")

brad_pitt_genres = dict(sorted(brad_pitt_genres.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------brad_pitt_genres:")
for genre, count in brad_pitt_genres.items():
    print(f"{genre}: {count/13}")

cate_blanchett_genres = dict(sorted(cate_blanchett_genres.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------cate_blanchett_genres:")
for genre, count in cate_blanchett_genres.items():
    print(f"{genre}: {count/11}")

emma_stone_genres = dict(sorted(emma_stone_genres.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------emma_stone_genres:")
for genre, count in emma_stone_genres.items():
    print(f"{genre}: {count/10}")

anna_kendrick_genres = dict(sorted(anna_kendrick_genres.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------anna_kendrick_genres:")
for genre, count in anna_kendrick_genres.items():
    print(f"{genre}: {count/10}")




