#import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import sklearn.model_selection #import train_test_split
import sklearn.tree #import DecisionTreeClassifier
import sklearn.metrics #import accuracy_score

import wittgenstein as lw
import pandas as pd

import subprocess
import shutil

from rulekit import RuleKit
from rulekit.classification import RuleClassifier
from rulekit.params import Measures

##from sklearn import tree
##import graphviz
#from sklearn.tree import export_text

import csv
import itertools
import collections

FILENAME_PREFIX = ""

chosen_5_attributes = ("runtime", "tomatometer_rating", "tomatometer_count", "audience_rating", "audience_count")
chosen_5_attributes_as_set = set(chosen_5_attributes)

class RtMovieNumeric:
    __slots__ = chosen_5_attributes + ("avg_distance",)

    def __init__(self, rt_movie_mahalanobis_distance, avg_distance):
        for attribute_name in chosen_5_attributes:
            setattr(self, attribute_name, rt_movie_mahalanobis_distance[f"x_{attribute_name}"])

        self.avg_distance = avg_distance

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)

CLASS_LT_2 = 0
CLASS_2_x_2p5 = 1
CLASS_2p5_x_3 = 2
CLASS_GT_3 = 3

def main():
    all_rt_movie_mahalanobis_distances = []

    with open(f"{FILENAME_PREFIX}-T5MA.csv", "r") as f:
        reader = csv.DictReader(f, delimiter=",", quotechar='"')

        for row in reader:
            all_rt_movie_mahalanobis_distances.append(row)

    rt_movie_average_mahalanobis_distances = []

    for rt_movie_mahalanobis_distances in grouper(all_rt_movie_mahalanobis_distances, 999):
        total_rt_movie_distance = 0

        for rt_movie_mahalanobis_distance in rt_movie_mahalanobis_distances:
            total_rt_movie_distance += float(rt_movie_mahalanobis_distance["distance"])

        avg_rt_movie_distance = total_rt_movie_distance/len(rt_movie_mahalanobis_distances)

        rt_movie_mahalanobis_distance_0 = rt_movie_mahalanobis_distances[0]
        rt_movie_chosen_attributes_dict = {attribute_name: rt_movie_mahalanobis_distance_0[f"x_{attribute_name}"] for attribute_name in chosen_5_attributes}
        rt_movie_chosen_attributes_dict["avg_distance"] = avg_rt_movie_distance

        if avg_rt_movie_distance < 2:
            class_label = CLASS_LT_2
        elif 2 <= avg_rt_movie_distance < 2.5:
            class_label = CLASS_2_x_2p5
        elif 2.5 <= avg_rt_movie_distance < 3:
            class_label = CLASS_2p5_x_3
        elif 3 <= avg_rt_movie_distance:
            class_label = CLASS_GT_3

        rt_movie_chosen_attributes_dict["class_label"] = class_label
        rt_movie_average_mahalanobis_distances.append(rt_movie_chosen_attributes_dict)

    with open(f"{FILENAME_PREFIX}-T6Data.csv", "w+") as f:
        csv_header_as_str = f"{','.join(chosen_5_attributes)},avg_distance,class_label\n"
        f.write(csv_header_as_str)

        writer = csv.DictWriter(f, fieldnames=chosen_5_attributes + ("avg_distance", "class_label"), delimiter=",", quotechar='"')
        for rt_movie in rt_movie_average_mahalanobis_distances:
            writer.writerow(rt_movie)

    #classes_frequency = collections.Counter()
    #
    #for rt_movie in rt_movie_average_mahalanobis_distances:
    #    avg_distance = rt_movie["avg_distance"]
    #    if avg_distance < 1:
    #        classes_frequency["x < 1"] += 1
    #    elif 1 <= avg_distance < 2:
    #        classes_frequency["1 <= x < 2"] += 1
    #    elif 2 <= avg_distance < 2.5:
    #        classes_frequency["2 <= x < 2.5"] += 1
    #    elif 2.5 <= avg_distance < 3:
    #        classes_frequency["2.5 <= x < 3"] += 1
    #    elif 3 <= avg_distance:
    #        classes_frequency["x >= 3"] += 1
    #
    #output = "".join(f"{k}: {v}\n" for k, v in classes_frequency.items())
    #print(output)

    #col = [ 'Class Name','Left weight','Left distance','Right weight','Right distance']
    #df = pd.read_csv('balance-scale.data',names=col,sep=',')
    #df.head()
    #
    #X = df.drop('Class Name',axis=1)
    #y = df[['Class Name']]

    print("Creating decision tree classifier!")
    #full_data = pd.DataFrame.from_records([{k: v for k, v in rt_movie.items() if k in chosen_5_attributes} for rt_movie in rt_movie_average_mahalanobis_distances])
    full_data = [tuple(v for k, v in rt_movie.items() if k in chosen_5_attributes) for rt_movie in rt_movie_average_mahalanobis_distances]
    full_class_labels = [rt_movie["class_label"] for rt_movie in rt_movie_average_mahalanobis_distances]

    training_data, test_data, training_class_labels, test_class_labels = sklearn.model_selection.train_test_split(
        full_data, full_class_labels, test_size=0.2, random_state=123123)

    #print(f"training_data: {training_data}\ntraining_class_labels: {training_class_labels}\ntest_data: {test_data}\ntest_class_labels: {test_class_labels}")
    classification_model = sklearn.tree.DecisionTreeClassifier(criterion="gini", random_state=777777)
    classification_model.fit(training_data, training_class_labels)

    test_data_predicted_class_labels = classification_model.predict(test_data)
    data_accuracy_score = sklearn.metrics.accuracy_score(test_class_labels, test_data_predicted_class_labels)
    decision_tree_confusion_matrix = sklearn.metrics.confusion_matrix(test_class_labels, test_data_predicted_class_labels)
    print(f"decision_tree_confusion_matrix: {decision_tree_confusion_matrix}")

    print(f"data_accuracy_score: {data_accuracy_score}")
    decision_tree_text = sklearn.tree.export_text(classification_model, feature_names=chosen_5_attributes)
    print(decision_tree_text)

    RuleKit.init()
    
    clf = RuleClassifier(
        induction_measure=Measures.C2,
        pruning_measure=Measures.C2,
        voting_measure=Measures.C2,
    )
    clf.fit(training_data, training_class_labels)
    predicted_labels = clf.predict(test_data)
    rule_score = sklearn.metrics.accuracy_score(test_class_labels, predicted_labels)
    rule_confusion_matrix = sklearn.metrics.confusion_matrix(test_class_labels, predicted_labels)

    print(f"rule_score: {rule_score}")
    print(f"rule_confusion_matrix: {rule_confusion_matrix}")
    print(f"model: {clf.model}")
    #movies_train = []
    #
    #for rt_movie, class_label in zip(training_data, training_class_labels):
    #    movies_train.append(f"{' '.join(rt_movie)} {class_label}\n")
    #
    #with open("movies-train.txt", "w+") as f:
    #    f.write("".join(movies_train))
    #
    #python_command = "python2" if shutil.which("python2") is not None else "python"
    #
    #subprocess.run((python_command, "ripperk.py", "-e", "learn", "-a", "movies-attr.txt", "-c", "class_label", "-t", "movies-train.txt", "-m", "movies-model.dat", "-o", "movies-model.txt"))

    #ripper_classification_model = lw.RIPPER(random_state=12)
    #ripper_classification_model.fit(training_data, y=training_class_labels, feature_names=chosen_5_attributes, pos_class=CLASS_2_x_2p5)
    #print(ripper_classification_model.out_model())
    #
    #ripper_accuracy = ripper_classification_model.score(test_data, test_class_labels)
    #print(f"ripper_accuracy: {ripper_accuracy}")
    

    #dot_data = sklearn.tree.export_graphviz(classification_model,
    #                    out_file=None,
    #                    feature_names=chosen_5_attributes,
    #                    class_names=("Less than 2", "2 <= x < 2.5", "2.5 <= x < 3", ">= 3"),  
    #                    filled=True, rounded=True,
    #                    special_characters=True)
    #graph = graphviz.Source(dot_data)
    #graph.render(filename=None, outfile="task6graph.png")
    #graph.save("task6graph.jpg")

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)   
    #clf_model.fit(X_train,y_train)
    #y_predict = clf_model.predict(X_test)
    #accuracy_score(y_test,y_predict)
    #
    ##target = list(df['Class Name'].unique())
    ##feature_names = list(X.columns)
    ##dot_data = tree.export_graphviz(clf_model,
    ##                                out_file=None, 
    ##                    feature_names=feature_names,  
    ##                    class_names=target,  
    ##                    filled=True, rounded=True,  
    ##                    special_characters=True)  
    ##graph = graphviz.Source(dot_data)
    #
    #r = export_text(clf_model, feature_names=feature_names)
    #print(r)

if __name__ == "__main__":
    main()
