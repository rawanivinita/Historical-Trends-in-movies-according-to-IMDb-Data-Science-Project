import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

df_movie = pd.read_csv("IMDB-Movie-Data.csv")

# we found that Revenue and MetaScore are the only columns with missing entries, so we removed rows with the missing  values

df_movie = df_movie.dropna(axis=0, how='any')
df_movie.info()
