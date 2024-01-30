# acc computed following the formula in the paper
# predictor obtained from https://github.com/raruidol/ArgumentRelationMining
from collections import defaultdict
from simpletransformers.classification import ClassificationModel
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from itertools import product

# process the answer
def fact_check(input_text):
    claim, *a = [line.strip() for line in input_text.split('\n') if line.strip()]

    verdict = (claim.strip('.').lower() == "true")

    a_squared = list(product(a, repeat=2))

    fact_check_result = (verdict, a, a_squared)

    return fact_check_result

def acc(A, Rsup, RAtt):
    def delta (aj):
        return int(_, aj in RAtt)

    def inner_sum(ai):
        Atts = [aj for aj, ax in RAtt if ai == ax]
        return 1 / len(Atts) * sum(map(delta, Atts))

    total_sum = 1 / len(A) * sum(inner_sum(ai) for ai, _ in A)

    return total_sum


predictor = ClassificationModel('albert', 'trained_models/albertxxlv2-model-trained', cuda_device=0, args={'silent':True})

def relation(a1, a2):
    predictions, raw_outputs = predictor.predict([[a1, a2]])
    #print(nb1, nb2)
    if predictions[0] == 0 or 2:
        print('Inference')
        return "Support"
    elif predictions[0] == 1:
        print('Conflict')
        return "Attacks"
    elif predictions[0] == 2:
        print('Rephrase')
        return "Support"
    elif predictions[0] == 3:
        print('No Relation')
        return "Neither"
    
def process(triplet):
    (verdict, A, a_squared) = triplet
    Rsup = []
    Ratt = []

    for a1, a2 in a_squared:
        tuple = (a1, a2)
        prediction = relation(a1, a2)
        if prediction == "Support":
            Rsup.append(tuple)
        if prediction == "Attacks":
            Ratt.append(tuple)

    return (A, Rsup, Ratt)
