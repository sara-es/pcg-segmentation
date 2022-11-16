from DataPresentation import DataPresentation
from FirstModel import PatientData
from constants import * 
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np 

data_presentation = DataPresentation()
dataset = PatientData("training_data.csv")

# Totals confusion Matrix 
confusion_matrix = []
for m in range(len(MURMUR_PRESENCE)):
    confusion_matrix.append([])
    for o in range(len(OUTCOMES)):
        confusion_matrix[m].append(len(dataset.patient_frame[(dataset.murmurs == MURMUR_PRESENCE[m]) & (dataset.outcomes == OUTCOMES[o])]))
        
# data_presentation.plot_confusion_matrix(confusion_matrix, 
#                                         "Visualising the Relationship between Murmur Presence and Clinical Outcome", 
#                                         "Outcome",
#                                         "Murmur",
#                                         OUTCOMES, MURMUR_PRESENCE)
print(confusion_matrix)
stat, p, dof, expected = chi2_contingency(np.array(confusion_matrix))
prob = 0.99
print(expected)
critical = chi2.ppf(prob, dof)

if abs(stat) >= critical:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')
