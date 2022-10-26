import json
from flask import Flask,jsonify
from Predicition import predict
app = Flask(__name__)

@app.route('/')
def hello_world():
    
    test_features =[4.352011,4.616276,11.117554,33.615473,0.30015,0.32078,0.328634,0.281975,0.397744,0.888144,2.342221,9.80051,0.150632,0.117047,4.98598,5.15558,12.880217,43.197222,0.345398,0.483322,0.518363,0.422966,0.505074,0.25669]
    test_features2=[3.531289,4.312772,9.004999,21.952221,0.315674,0.325269,0.282931,0.195474,0.438748,0.629365,1.58019,5.503635,0.164347,0.101833,3.869108,4.963869,9.80051,26.036513,0.377624,0.487647,0.516817,0.318591,0.548999,0.295804]
    ln1=predict.breast_cancer_prediction_logistic(test_features)
    ln2=predict.breast_cancer_prediction_logistic(test_features2)
    
    dn1=predict.breast_cancer_prediction_decision(test_features)
    dn2=predict.breast_cancer_prediction_decision(test_features2)
    
    rn1=predict.breast_cancer_prediction_random(test_features)
    rn2=predict.breast_cancer_prediction_random(test_features2)


    return jsonify({'Logistic Tree Result':{'First Feature':{'Features':test_features,'Result':ln1},'Second Feature':{'Features':test_features2,'Result':ln2}},
    'Decision Tree Result':{'First Feature':{'Features':test_features,'Result':dn1},'Second Feature':{'Features':test_features2,'Result':dn2}},
    'Random Forest Tree Result':{'First Feature':{'Features':test_features,'Result':rn1},'Second Feature':{'Features':test_features2,'Result':rn2}}
    })


if __name__=="__main__":
    app.run(debug=True,port=8000)