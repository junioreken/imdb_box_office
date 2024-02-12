from audioop import reverse
import pandas as pd
import numpy as np
from statistics import mode
from collections import Counter
from matplotlib import pyplot as plt


movies = pd.read_csv("rotten_tomatoes_movies.csv")

task2_dimensions = movies[['genres', 'original_release_date', 'audience_rating', 'tomatometer_status']]
task2_dimensions.to_csv("214187611-216343097-215147499-T2.csv")

# print(movies.head())
# print(movies.info())
# print(task2_attributes.head(50))

# Nominal attribute = genres
# Interval attribute = original_release_date
# Ratio attribute = audience_rating
# Ordinal attribute = tomatometer_status
genre_list = list(movies['genres'].dropna()) # unprocessed list of genres
release_list = list(movies.original_release_date.dropna()) # unprocessed original release dates
audience_ratings = list(movies.audience_rating.dropna())


genres = [] # list of processed individual genres excluding "na" values
years_of_release = [] # list of years of original release date excluding "na" values

# parsing each individual genre per value in the genres dimension and adding them to the genres list.
for genre in genre_list:
    gen = str(genre).split(', ')
    for g in gen:
        if g != "nan":
            genres.append(g.strip())

# parsing the year of each date in the original_release_date dimension and adding them to the years_of_release list.
for date in release_list:
    val = str(date).split('-')[0]
    if val != "nan":
        years_of_release.append(int(val))


genre_freq_dict = dict(Counter(genres)) # dictionary of each genre and its corresponding frequency
sorted_genre = sorted(genre_freq_dict, key = genre_freq_dict.get, reverse=True) # sorting the genres in descending order based on frequency
year_freq_dict = dict(Counter(years_of_release)) # dictionary of each year of original release and its corresponding frequency
sorted_year = sorted(year_freq_dict, key = year_freq_dict.get, reverse=True) # sorting the years in descending order based on frequency

top20_years = sorted_year[:20] # list of Top 20 most common movie release years
top20_year_freq = sorted(year_freq_dict.values(), reverse=True)[:20] # Top 20 frequencies of release years

status_freq_dict = dict(Counter(list(movies.tomatometer_status.dropna()))) # dictionary of each status and its corresponding frequency
sorted_status = sorted(status_freq_dict, key = status_freq_dict.get, reverse=True) # sorting the statuses in descending order based on frequency


top15_genres = sorted_genre[:15] # list of top 15 most common movie genres
top15_genre_freq = sorted(genre_freq_dict.values(), reverse=True)[:15] # list of top 15 movie genres by frequency

# Histogram of Audience Ratings (Percentage of positive user ratings)
fig = plt.figure(figsize=(8, 6))
plt.hist(audience_ratings, bins = 20, edgecolor = "black")
plt.xlabel('Audience Rating (audience_rating)')
plt.ylabel('Frequency')
plt.title('Audience Ratings by Frequency')
plt.savefig("audience_ratings_hist.png")
# plt.show()


# Pie chart of Top 15 Most Common Movie Genres
plt.close('all')
fig = plt.figure(figsize=(11, 9))
plt.pie(top15_genre_freq, labels = top15_genres, autopct="%0.1f%%")
plt.axis('equal')
# plt.legend(genre_freq_dict
#.keys())
plt.title('Top 15 Most Common Movie Genres')
plt.savefig("top15_movie_genres_pie.png")
# plt.show()


# Pie chart of Top 20 Most Common Original Release Years of Movies
plt.close('all')
fig = plt.figure(figsize=(11, 9))
plt.pie(top20_year_freq, labels = top20_years, autopct="%0.1f%%")
plt.axis('equal')
plt.title('Top 20 Most Common Original Release Years of Movies')
plt.savefig("movies_release_year_pie.png")
# plt.show()


# Bar chart of Critics Tomatometer Status Rankings by Frequency
plt.close('all')
fig = plt.figure(figsize=(8, 6))
plt.bar(list(status_freq_dict.keys()), list(status_freq_dict.values()))
plt.xlabel('Tomatometer Status (tomatometer_status)')
plt.ylabel('Frequency')
plt.title('Critics Tomatometer Status Rankings by Frequency')
plt.savefig("tomatometer_status_freq_bar.png")
# plt.show()

# printing results
print("Mode of genres (nominal attribute):", mode(genres)) # Mode of Nominal Attribute: genres
print("Least occuring movie genre in genres (nominal attribute):", sorted_genre[-1]) # Least occuring movie genre in genres
print("Mean of audience_rating (ratio attribute):", round(
    movies.audience_rating.mean(), 2))  # Mean of ratio attribute: audience_rating
print("Standard Deviation of audience_rating (ratio attribute):", round(
    movies.audience_rating.std(), 2))  # Standard deviation of ratio attribute: audience_rating
print("Mode of years of release in original_release_date (interval attribute):",
      mode(years_of_release)) # Mode of years of release in original_release_date
print("50th percentile of the years of release in original_release_date (interval attribute):", np.percentile(years_of_release, 50)) # 50th percentile of the years of release in original_release_date
print("Mode of tomatometer_status (ordinal attribute):", list(movies.tomatometer_status.mode())[0]) # Mode of tomatometer_status
print("Least occuring status in tomatometer_status (ordinal attribute):", sorted_status[-1]) #classification of percentage of positive critic reviews
print("\n\n******************** OUTPUT COMPLETE *********************")

