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



# transfer the genre column entries which are list of strings into list of lists of strings

#df_movie['Genre'] = df_movie['Genre'].apply(lambda x : x.split(','))
df_movie['Genre'].fillna('NA', inplace = True)
df_movie['Genre'] = df_movie['Genre'].str.split(',').tolist()
flat_genre = [item for sublist in df_movie['Genre'] for item in sublist]
set_genre = set(flat_genre)
unique_genre = list(set_genre)
df_movie = df_movie.reindex(df_movie.columns.tolist()+unique_genre,axis = 1, fill_value= 0)

for index,row in df_movie.iterrows():
    for val in row.columns.split(','):
        if val != 'NA':
            df_movie.loc[index,val] = 1


df_movie.drop('Genre', axis=1, inplace = True)
df_movie

#print((df_movie['Genre'][0]))


#dummies = df_movie['Genre'].str.get_dummies(sep=',')
#df_movie = pd.concat([df_movie, dummies], axis=1)

# Print the updated dataframe
#print(df_movie.columns)

#plt.figure(figsize=(16,6))
#heatmap = sns.heatmap(df_movie.corr(), vmin = -1, vmax = 1, annot = True)
#heatmap.set_title("Correlation Heatmap", fontdict = {"fontsize":12}, pad = 12);
#plt.show()

#https://towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173




