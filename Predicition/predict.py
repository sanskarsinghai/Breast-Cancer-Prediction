import pickle
import warnings

warnings.filterwarnings('ignore')

def breast_cancer_prediction_logistic(features):
    pickled_model = pickle.load(open('model\\breast_cancer_detection_Logistic.pkl', 'rb'))
    breast_cancer = str(round(list(pickled_model.predict([features]))[0]))
    a=''
    if breast_cancer=='0':
        a='B'
    else:
        a='M'

    return str("Diagnosis: " + breast_cancer+" or "+a)

def breast_cancer_prediction_decision(features):
    pickled_model = pickle.load(open('model\\breast_cancer_detection_Decision.pkl', 'rb'))
    breast_cancer = str(round(list(pickled_model.predict([features]))[0]))
    a=''
    if breast_cancer=='0':
        a='B'
    else:
        a='M'

    return str("Diagnosis: " + breast_cancer+" or "+a)

def breast_cancer_prediction_random(features):
    pickled_model = pickle.load(open('model\\breast_cancer_detection_Random.pkl', 'rb'))
    breast_cancer = str(round(list(pickled_model.predict([features]))[0]))
    a=''
    if breast_cancer=='0':
        a='B'
    else:
        a='M'

    return str("Diagnosis: " + breast_cancer+" or "+a)

def breast_cancer_prediction_kn(features):
    pickled_model = pickle.load(open('model\\breast_cancer_detection_KNearest.pkl', 'rb'))
    breast_cancer = str(round(list(pickled_model.predict([features]))[0]))
    a=''
    if breast_cancer=='0':
        a='B'
    else:
        a='M'

    return str("Diagnosis: " + breast_cancer+" or "+a)
