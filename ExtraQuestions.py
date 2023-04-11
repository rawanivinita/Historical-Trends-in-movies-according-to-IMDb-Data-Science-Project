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
plt.show()

# Found Top 20 Actors with highest occurences, sorted in descending order
actor_occurences_extra = {}

for index, row in df_movie.iterrows():
    actors = row['Actors'].split(', ')
    for actor in actors:
        if actor in actor_occurences_extra:
            actor_occurences_extra[actor] += 1
        else:
            actor_occurences_extra[actor] = 1

top_actors_extra = sorted(actor_occurences_extra.items(), key=lambda x: x[1], reverse=True)
top_20_actors_extra = [x[0] for x in top_actors_extra[:20]]

#print(top_20_actors)
#Mark Wahlberg is first on this list while the first female (Emma Stone) is 11th
chosen_actors_extra = ['Mark Wahlberg', 'Hugh Jackman', 'Brad Pitt',  'Cate Blanchett', 'Emma Stone', 'Anna Kendrick']

for actor in chosen_actors_extra:
    df_movie[actor] = df_movie['Actors'].apply(lambda x: 1 if actor in x else 0)

num_of_movies_extra = {}

for actor in chosen_actors_extra:
    num_of_movies_extra[actor] = num_of_movies_extra.get(actor, 0) + sum(df_movie[actor] == 1)

print(num_of_movies_extra)

genres_extra = df_movie["Genre"].str.get_dummies(",")
df_movie_emma_mark_extra = pd.concat([df_movie, genres_extra], axis=1)

#print(df_movie_emma_mark.columns)

df_movie_emma_mark_extra = df_movie_emma_mark_extra[(df_movie_emma_mark_extra['Mark Wahlberg'] == 1) | (df_movie_emma_mark_extra['Hugh Jackman'] == 1)
                                        | (df_movie_emma_mark_extra['Brad Pitt'] == 1) | (df_movie_emma_mark_extra['Cate Blanchett'] == 1)
                                        | (df_movie_emma_mark_extra['Emma Stone'] == 1) | (df_movie_emma_mark_extra['Anna Kendrick'] == 1)]

mark_wahlberg_genres_extra = {}
hugh_jackman_genres_extra = {}
brad_pitt_genres_extra = {}
cate_blanchett_genres_extra = {}
emma_stone_genres_extra = {}
anna_kendrick_genres_extra = {}

genre_dicts_extra = []
genre_dicts_extra.append(mark_wahlberg_genres_extra)
genre_dicts_extra.append(hugh_jackman_genres_extra)
genre_dicts_extra.append(brad_pitt_genres_extra)
genre_dicts_extra.append(cate_blanchett_genres_extra)
genre_dicts_extra.append(emma_stone_genres_extra)
genre_dicts_extra.append(anna_kendrick_genres_extra)

for genre in genres_extra.columns:
    mark_wahlberg_genres_extra[genre] = mark_wahlberg_genres_extra.get(genre, 0) + sum((df_movie_emma_mark_extra['Mark Wahlberg'] == 1) & (df_movie_emma_mark_extra[genre] == 1))
    hugh_jackman_genres_extra[genre] = hugh_jackman_genres_extra.get(genre, 0) + sum((df_movie_emma_mark_extra['Hugh Jackman'] == 1) & (df_movie_emma_mark_extra[genre] == 1))
    brad_pitt_genres_extra[genre] = brad_pitt_genres_extra.get(genre, 0) + sum((df_movie_emma_mark_extra['Brad Pitt'] == 1) & (df_movie_emma_mark_extra[genre] == 1))
    cate_blanchett_genres_extra[genre] = cate_blanchett_genres_extra.get(genre, 0) + sum((df_movie_emma_mark_extra['Cate Blanchett'] == 1) & (df_movie_emma_mark_extra[genre] == 1))
    emma_stone_genres_extra[genre] = emma_stone_genres_extra.get(genre, 0) + sum((df_movie_emma_mark_extra['Emma Stone'] == 1) & (df_movie_emma_mark_extra[genre] == 1))
    anna_kendrick_genres_extra[genre] = anna_kendrick_genres_extra.get(genre, 0) + sum((df_movie_emma_mark_extra['Anna Kendrick'] == 1) & (df_movie_emma_mark_extra[genre] == 1))

mark_wahlberg_genres_extra = dict(sorted(mark_wahlberg_genres_extra.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------mark_wahlberg_genres:")
for genre, count in mark_wahlberg_genres_extra.items():
    print(f"{genre}: {count/15}")

hugh_jackman_genres_extra = dict(sorted(hugh_jackman_genres_extra.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------hugh_jackman_genres:")
for genre, count in hugh_jackman_genres_extra.items():
    print(f"{genre}: {count/14}")

brad_pitt_genres_extra = dict(sorted(brad_pitt_genres_extra.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------brad_pitt_genres:")
for genre, count in brad_pitt_genres_extra.items():
    print(f"{genre}: {count/13}")

cate_blanchett_genres_extra = dict(sorted(cate_blanchett_genres_extra.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------cate_blanchett_genres:")
for genre, count in cate_blanchett_genres_extra.items():
    print(f"{genre}: {count/11}")

emma_stone_genres_extra = dict(sorted(emma_stone_genres_extra.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------emma_stone_genres:")
for genre, count in emma_stone_genres_extra.items():
    print(f"{genre}: {count/10}")

anna_kendrick_genres_extra = dict(sorted(anna_kendrick_genres_extra.items(), key=lambda item: item[1], reverse=True))
print("-----------------------------------------anna_kendrick_genres:")
for genre, count in anna_kendrick_genres_extra.items():
    print(f"{genre}: {count/10}")




