import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

# method obtained from Stackover flow user: Boern :- https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

movies = pd.read_csv("rotten_tomatoes_movies.csv")
selected_dimension = movies[['original_release_date', 'runtime', 'tomatometer_status', 'tomatometer_rating', 'tomatometer_count', 'audience_rating', 
                     'audience_count', 'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count']]

ratio_dimension = movies[['runtime', 'tomatometer_rating', 'tomatometer_count', 'audience_rating', 'audience_count', 
                          'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count']]

ratio_dimension = clean_dataset(ratio_dimension)
# print(ratio_dimension.info())

# selected_dimension = clean_dataset(selected_dimension) # can't use dimensions containing string variables. 

other_dimension = movies[['original_release_date', 'tomatometer_status']]

# normalizing data before applying fit method

# considering only ratio attributes because categorical/ordinal/interval cannot be normalized. 
min_max_scaler = preprocessing.MinMaxScaler()
ratio_scaled = min_max_scaler.fit_transform(ratio_dimension)
ratio_normalized = pd.DataFrame(ratio_scaled)

# ratio_normalized=(selected_dimension - selected_dimension.mean()) / selected_dimension.std()
pca = PCA(n_components=2)
pca.fit(ratio_normalized)
components = pd.DataFrame(pca.components_, columns=ratio_dimension.columns)
components.to_csv("214187611-216343097-215147499-T4.csv")
# print(components)

plt.plot(pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.title('Explained Variance vs Components')
# plt.show()
