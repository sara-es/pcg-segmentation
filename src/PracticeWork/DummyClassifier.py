import sys, os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from DataManipulation.DataPresentation import DataPresentation
from PracticeWork.FirstModel import PatientData
from PracticeWork.FirstModel import split_data


STRATEGIES = ["most_frequent", "stratified", "uniform"]


dataset = PatientData("training_data.csv")
accuracy = [[], []] 

for i in range(len(STRATEGIES)):

    train_data, test_data = split_data(dataset, 0.7)

    feature_matrix = dataset.convert_patient_df_to_feature_matrix()
    murmur_classes = dataset.murmurs_to_numerical_classes()
    outcome_classes = dataset.outcomes_to_numerical_classes()

    train_X = feature_matrix[train_data.indices]
    train_murmurs = murmur_classes[train_data.indices]
    train_outcomes = outcome_classes[train_data.indices]

    test_X = feature_matrix[test_data.indices]
    test_murmurs = murmur_classes[test_data.indices]
    test_outcomes = outcome_classes[test_data.indices]

    dummy_clf = DummyClassifier(strategy=STRATEGIES[i])
    dummy_clf.fit(train_X, train_murmurs)
    dummy_clf.predict(test_X)
    accuracy[0].append(dummy_clf.score(test_X, test_murmurs))

    dummy_clf.fit(train_X, train_outcomes)
    dummy_clf.predict(test_X)
    accuracy[1].append(dummy_clf.score(test_X, test_outcomes))

# plt.title('Dummy Classifier Performance on Feature Matrix')

# x_axis = np.arange(3)

# plt.bar(x_axis-0.2, accuracy[0], width=0.4, label = 'Murmur Presence')
# plt.bar(x_axis+0.2, accuracy[1], width=0.4, label = 'Clinical Outcome')

# plt.xticks(x_axis, STRATEGIES)
# plt.legend()


# plt.savefig("DummyClassifierAccuracy")

data_presentation = DataPresentation()
data_presentation.plot_multi_bar_chart(x_labels=STRATEGIES, 
                                       x_label_title="SkLearn Dummy Classifier",
                                       y_label_title="Classifier Accuracy",
                                       data=accuracy, 
                                       bar_labels=["Murmur Presence", "Clinical Outcome"],
                                       title='Dummy Classifier Performance on Feature Matrix')