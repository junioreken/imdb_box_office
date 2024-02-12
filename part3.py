import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from data_object_classes import *
from sklearn import preprocessing
import csv

def main():
    handler = data_object_classes.DataObjectsHandler(
    "rotten_tomatoes_movies.csv",
    "rotten_tomatoes_critic_reviews.csv",
    "worldwide_and_domestic_lifetime_box_data_2022-01-27.csv",
    "second_weekend_drop_2022-01-27.csv"
    )
    
    outfile = open('csvexport.csv', 'w')
    outcsv = csv.writer(outfile)
    outcsv.writerow(['runtime, frequency'])

    # Plot runtime frequencies
    run_time_freq_dict = {}
    for rt_movie in handler.rt_movies:
        if rt_movie.runtime != "":
            if int(rt_movie.runtime) in run_time_freq_dict:
                run_time_freq_dict[int(rt_movie.runtime)] += 1
            else:
                run_time_freq_dict[int(rt_movie.runtime)] = 1

    x = np.array(list(run_time_freq_dict.keys()))
    y = np.array(list(run_time_freq_dict.values()))
    np_2d_arr = np.column_stack((x, y))
    outcsv.writerows(np_2d_arr)

    plt.stem(x, y, markerfmt=" ")
    plt.suptitle('Movie Runtime Frequencies (Original)', fontsize=20)
    plt.xlabel('Runtime', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    outcsv.writerow(['runtime_zscore, frequency_zscore'])
    x_zscore = stats.zscore(np.array(x))
    y_zscore = stats.zscore(np.array(y))
    np_2d_arr_zscore = np.column_stack((x_zscore, y_zscore))
    outcsv.writerows(np_2d_arr_zscore)

    plt.suptitle('Movie Runtime Frequencies (Z-Score Standardized Runtimes)', fontsize=20)
    plt.xlabel('Runtime', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.stem(x_zscore, y, markerfmt=" ")
    plt.gcf().set_size_inches(11, 8.5)
    plt.show()

    plt.suptitle('Movie Runtime Frequencies (Z-Score Standardized Frequencies)', fontsize=20)
    plt.xlabel('Runtime', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.stem(x, y_zscore, markerfmt=" ")
    plt.gcf().set_size_inches(11, 8.5)
    plt.show()

    plt.suptitle('Movie Runtime Frequencies (Z-Score Standardized Runtimes and Frequencies)', fontsize=20)
    plt.xlabel('Runtime', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.stem(x_zscore, y_zscore, markerfmt=" ")
    plt.gcf().set_size_inches(11, 8.5)
    plt.show()

    outcsv.writerow(['runtime_minmax, frequency_minmax'])
    np_2d_arr = np.column_stack((x, y))
    np_2d_arr_scaled = preprocessing.MinMaxScaler().fit_transform(np_2d_arr)
    outcsv.writerows(np_2d_arr_scaled)

    x_scaled =[]
    y_scaled = []
    for i in range(len(np_2d_arr_scaled)):
        x_scaled.append(np_2d_arr_scaled[i][0])
        y_scaled.append(np_2d_arr_scaled[i][1])

    plt.stem(x_scaled, y, markerfmt=" ")
    plt.suptitle('Movie Runtime Frequencies (Min-Max Normalized Runtimes)', fontsize=20)
    plt.xlabel('Runtime', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    plt.stem(x, y_scaled, markerfmt=" ")
    plt.suptitle('Movie Runtime Frequencies (Min-Max Normalized Frequencies)', fontsize=20)
    plt.xlabel('Runtime', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    plt.stem(x_scaled, y_scaled, markerfmt=" ")
    plt.suptitle('Movie Runtime Frequencies (Min-Max Normalized Runtimes and Frequencies)', fontsize=20)
    plt.xlabel('Runtime', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    # Plot runtime frequencies
    release_year_freq_dict = {}

    for rt_movie in handler.rt_movies:
        if rt_movie.original_release_date != "":
            release_year = date.fromisoformat(rt_movie.original_release_date).year
            if release_year in release_year_freq_dict:
                release_year_freq_dict[release_year] += 1
            else:
                release_year_freq_dict[release_year] = 1

    x = np.array(list(release_year_freq_dict.keys()))
    y = np.array(list(release_year_freq_dict.values()))
    outcsv.writerow(['year, frequency'])
    np_2d_arr = np.column_stack((x, y))
    outcsv.writerows(np_2d_arr)

    plt.stem(x, y, markerfmt=" ")
    plt.suptitle('Number of Movies by Release Year (Original)', fontsize=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    x_zscore = stats.zscore(np.array(x))
    y_zscore = stats.zscore(np.array(y))
    outcsv.writerow(['year_zscore, frequency_zscore'])
    np_2d_arr_zscore = np.column_stack((x_zscore, y_zscore))
    outcsv.writerows(np_2d_arr_zscore)


    plt.suptitle('Number of Movies by Release Year (Z-Score Standardized Years)', fontsize=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.stem(x_zscore, y, markerfmt=" ")
    plt.gcf().set_size_inches(11, 8.5)
    plt.show()

    plt.suptitle('Number of Movies by Release Year (Z-Score Standardized Frequencies)', fontsize=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.stem(x, y_zscore, markerfmt=" ")
    plt.gcf().set_size_inches(11, 8.5)
    plt.show()

    plt.suptitle('Number of Movies by Release Year (Z-Score Standardized Runtimes and Frequencies)', fontsize=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.stem(x_zscore, y_zscore, markerfmt=" ")
    plt.gcf().set_size_inches(11, 8.5)
    plt.show()

    np_2d_arr = np.column_stack((x, y))
    np_2d_arr_scaled = preprocessing.MinMaxScaler().fit_transform(np_2d_arr)
    outcsv.writerow(['year_minmax, frequency_minmax'])
    outcsv.writerows(np_2d_arr_scaled)
    x_scaled =[]
    y_scaled = []
    for i in range(len(np_2d_arr_scaled)):
        x_scaled.append(np_2d_arr_scaled[i][0])
        y_scaled.append(np_2d_arr_scaled[i][1])

    plt.stem(x_scaled, y, markerfmt=" ")
    plt.suptitle('Number of Movies by Release Year (Min-Max Normalized Years)', fontsize=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    plt.stem(x, y_scaled, markerfmt=" ")
    plt.suptitle('Number of Movies by Release Year (Min-Max Normalized Frequencies)', fontsize=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    plt.stem(x_scaled, y_scaled, markerfmt=" ")
    plt.suptitle('Number of Movies by Release Year (Min-Max Normalized Years and Frequencies)', fontsize=20)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

    outfile.close()


if __name__ == "__main__":
    main()
