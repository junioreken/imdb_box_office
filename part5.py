import csv
import numpy as np
import itertools
import scipy.spatial
import random

import data_object_classes

# runtime
# tomatometer_rating
# tomatometer_count
# audience_rating
# audience_count

# rotten_tomatoes_link,movie_title,movie_info,critics_consensus,content_rating,genres,directors,authors,actors,original_release_date,streaming_release_date,runtime,production_company,tomatometer_status,tomatometer_rating,tomatometer_count,audience_status,audience_rating,audience_count,tomatometer_top_critics_count,tomatometer_fresh_critics_count,tomatometer_rotten_critics_count

FILENAME_PREFIX = ""

#class Distance:
#    __slots__ = ("u", "v", "distance")
#
#    def __init__(self, u, v, distance):
#        self.u = u
#        self.v = v
#        self.distance = distance

chosen_5_attributes = ("runtime", "tomatometer_rating", "tomatometer_count", "audience_rating", "audience_count")

EUCLIDEAN = 0
COSINE = 1
MAHALANOBIS = 2

class RtMovieXY:
    __slots__ = ("x", "y", "xy")

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xy = x + y

def calculate_distance(filtered_rt_movies, distance_func, csv_output_filename, report_output_filename, **kwargs):
    len_filtered_rt_movies = len(filtered_rt_movies)
    distances = []

    for i, rt_movie_x in enumerate(filtered_rt_movies):
        if i % 100 == 0:
            print(f"{csv_output_filename}: {i}")
    
        for j, rt_movie_y in enumerate(filtered_rt_movies):
            if i == j:
                continue

            cur_distance = distance_func(rt_movie_x, rt_movie_y, **kwargs)
            #if distance_type == EUCLIDEAN:
            #    cur_distance = np.linalg.norm(rt_movie_x - rt_movie_y)
            #elif distance_type == COSINE:
            #    cur_distance = np.dot(rt_movie_x, rt_movie_y)/(np.linalg.norm(rt_movie_x) * np.linalg.norm(rt_movie_y))
            #else:
            #    delta = rt_movie_x - rt_movie_y
            #    cur_distance = np.sqrt(np.einsum("nj,jk,nk->n", delta, VI, delta))
            #try:
            #    cur_distance = distance_func(rt_movie_x, rt_movie_y, **kwargs)
            #except Exception as e:
            #    raise RuntimeError(f"rt_movie_x: {rt_movie_x}, rt_movie_y: {rt_movie_y}") from e
    
            #distances.append((rt_movie_x, rt_movie_y, cur_distance))
            #distance_intermediate = np.concatenate((rt_movie_x, rt_movie_y))
            #np.append(distance_intermediate, cur_distance)
            distances.append(rt_movie_x + rt_movie_y + [cur_distance])

    print(f"distances[0]: {distances[0]}")
    random_10_distances = random.sample(distances, k=10)

    csv_header = []

    for prefix in ("x", "y"):
        for attribute_name in chosen_5_attributes:
            csv_header.append(f"{prefix}_{attribute_name}")

    csv_header.append("distance")
    csv_header_as_str = ','.join(csv_header)

    with open(f"{FILENAME_PREFIX}-{csv_output_filename}", "w+") as f:
        f.write(f"{csv_header_as_str}\n")

        writer = csv.writer(f, delimiter=",", quotechar='"')

        for distance in distances:
            writer.writerow(distance)

    output = ""
    csv_header_as_table_header = "\t".join(csv_header)
    output += f"{csv_header_as_table_header}\n"

    for distance in random_10_distances:
        output += "\t".join(str(x) for x in distance[:-1]) + f"\t{distance[-1]:.10f}" + "\n"

    with open(report_output_filename, "w+") as f:
        f.write(output)

def int_or_float(x):
    try:
        return int(x)
    except ValueError:
        return float(x)

def main():
    random.seed(567)

    handler = data_object_classes.DataObjectsHandler(
        "rotten_tomatoes_movies.csv",
        None,#"rotten_tomatoes_critic_reviews.csv",
        None,#"worldwide_and_domestic_lifetime_box_data_2022-01-27.csv",
        None,#"second_weekend_drop_2022-01-27.csv"
    )

    rt_movies, rt_reviews, lifetime_grosses, second_week_drops = handler.get_all_data_objects()
    #rt_movies = random.sample(rt_movies_full, k=1000)
    filtered_rt_movies = []

    for rt_movie in rt_movies:
        try:
            filtered_rt_movie = [int_or_float(getattr(rt_movie, attribute_name)) for attribute_name in chosen_5_attributes]
        except Exception as e:
            print(f"rt_movie: {rt_movie.movie_title}, {', '.join(getattr(rt_movie, attribute_name) for attribute_name in chosen_5_attributes)}")
            continue

        filtered_rt_movies.append(filtered_rt_movie)

    with open(f"{FILENAME_PREFIX}-T5Data.csv", "w+") as f:
        f.write(",".join(chosen_5_attributes) + "\n")

        writer = csv.writer(f, delimiter=",", quotechar='"')

        for filtered_rt_movie in filtered_rt_movies:
            writer.writerow(filtered_rt_movie)

    filtered_rt_movies_sample = random.sample(filtered_rt_movies, k=1000)

    # euclidean distance
    calculate_distance(filtered_rt_movies_sample, scipy.spatial.distance.euclidean, "T5EU.csv", "random_10_euclidean_distances.txt")

    # cosine distance
    calculate_distance(filtered_rt_movies_sample, scipy.spatial.distance.cosine, "T5CO.csv", "random_10_cosine_distances.txt")

    # mahalanobis distance
    V = np.cov(np.array(filtered_rt_movies_sample).T)
    VI = np.linalg.inv(V)

    calculate_distance(filtered_rt_movies_sample, scipy.spatial.distance.mahalanobis, "T5MA.csv", "random_10_mahalonobis_distances.txt", VI=VI)

if __name__ == "__main__":
    main()
