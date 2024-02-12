import pandas as pd
import numpy as np
from statistics import mode
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns

# Rotten Tomatoes Movie dataset
movies = pd.read_csv("rotten_tomatoes_movies.csv")
movies = movies[['movie_title', 'content_rating',
                 'runtime', 'tomatometer_rating', 'audience_rating']]
# IMDB Box Office Mojo dataset
mojo_box_office = pd.read_csv(
    "worldwide_and_domestic_lifetime_box_data_2022-01-27.csv")
second_week_box = pd.read_csv("second_weekend_drop_2022-01-27.csv")
second_week_box = second_week_box[['title', 'second_weekend_gross']]
movies_box_office = pd.merge(
    left=movies, right=mojo_box_office, left_on='movie_title', right_on='title')
movies_second_box = pd.merge(
    left=movies, right=second_week_box, left_on='movie_title', right_on='title')

runtime_box_office = movies_box_office[['ww_lifetime_gross', 'domestic_lifetime_gross', 'foreign_lifetime_gross', 'runtime', 'tomatometer_rating', 'audience_rating']][(
    movies_box_office.year > 1999)]  # Shorter time period to adjust for trend of change in runtime

audience_mean = []
critics_mean = []
grand_mean = []
gross_mean = []

# method obtained from Stackover flow user: Boern :- https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]

# Obtaining aggregate means of a variable based on/grouped by another variable
def aggregate_mean(criteria, column, measured_column, df, list):
    for i in range(len(criteria)):
        mean_rating = df[df[column] == criteria[i]][measured_column].mean()
        list.append(round(mean_rating, 2))
    
# cleaning datasets
movies_box_office = clean_dataset(movies_box_office)
movies = clean_dataset(movies)
movies_second_box = clean_dataset(movies_second_box)

# dictionary of each content rating and their corresponding frequencies in movies dataframe
crating_freq_dict = dict(Counter(list(movies['content_rating'])))
sorted_crating = sorted(
    crating_freq_dict, key=crating_freq_dict.get, reverse=True)
cont_rating = list(crating_freq_dict.keys())

# dictionary of each content rating and their corresponding frequencies in movies_box_office dataframe
cont_movies_box_dict = dict(Counter(list(movies_box_office['content_rating'])))
cont_mb_rating = list(cont_movies_box_dict.keys())

aggregate_mean(cont_rating, 'content_rating', 'audience_rating', movies, audience_mean) # Aggregate mean audience rating grouped by content rating
aggregate_mean(cont_rating, 'content_rating', 'tomatometer_rating', movies, critics_mean) # Aggregate mean tomatometer ratings grouped by content rating
aggregate_mean(cont_mb_rating, 'content_rating', 'ww_lifetime_gross', movies_box_office, gross_mean) # Aggregate mean lifetime gross grouped by content rating in movies_box_office dataframe

grand_mean = [round((aud + crit)/2, 2) for aud, crit in zip(audience_mean, critics_mean)]

# Printing results
print("\nIn Movies Dataframe : \n")

for i in range(len(cont_rating)):
    print("For {} {} rated movies, Mean Audience Rating: {}, Mean Critics Rating: {}, Overall Grand Mean {}.".format(
        crating_freq_dict[cont_rating[i]], cont_rating[i], audience_mean[i], critics_mean[i], grand_mean[i]))

print("\nIn Movies and Box Office Merged Dataframe: ")

for i in range(len(cont_mb_rating)):
    print("For {} {} rated movies, Mean World Wide Lifetime Gross Earning is {}.".format(
        cont_movies_box_dict[cont_mb_rating[i]], cont_mb_rating[i], gross_mean[i]))

print("\n")

mean_tomatometer_ratings = movies.groupby(
    'content_rating').tomatometer_rating.mean()
mean_content_gross = movies_box_office.groupby(
    'content_rating').ww_lifetime_gross.mean()

print(mean_tomatometer_ratings, '\n')
print(mean_content_gross, '\n')

# INSIGHT: 
# It is observed from the DataFrame mean_content_gross that G-rated movies had the most earnings, 
# followed by PG-13 rated movies. On the other hand, NR and NC17 movies had the worst earnings. 
# This may suggest that movies with content ratings that are open to the general public or that are intense 
# and dramatic tend to have higher earnings at the box office which makes sense since there are less restrictions. 
# It can also be noted that some of the best and most popular movies are PG-13, G or PG-rated like Avatar, Interstellar, 
# Batman The Dark Knight, The Lion King, and the Harry Potter sequel.

# Plotting Correlation between runtime, lifetime gross, tomatometer_rating, and audience_rating from 2000 - 2022
plt.close("all")
fig = plt.figure(figsize=(14, 11))
hmap = sns.heatmap(runtime_box_office.corr(), annot=True)
hmap.set(title="Correlation Matrix of Runtime, Ratings and World Wide Lifetime Gross Earnings for 2000-2022 Movies\n")
plt.savefig("runtime_boxoffice_correlation_heatMap.png")

# INSIGHT:
# The correlation matrix above shows that there is little correlation between BoxOfficeMojo’s dataset’s 
# life time gross earnings (lifetime_gross) and Rotten Tomatoes’ audience ratings (audience_rating) 
# and critics ratings (tomatometer_ratings). Each being 0.14 and 0.047 respectively. 
# This was surprising because one would expect these ratings would significantly influence the gross earnings of a movie.

# Plotting Correlation between runtime, lifetime gross, tomatometer_rating, audience_rating and Second Week Gross
plt.close("all")
fig = plt.figure(figsize=(13, 11))
hmap = sns.heatmap(movies_second_box.corr(), annot=True)
hmap.set(title="Correlation Matrix of Runtime, Ratings and Second Week Gross Earnings for All Movies\n")
plt.savefig("second_boxoffice_correlation_heatMap.png")

print("************** RESULTS COMPLETED *****************")
