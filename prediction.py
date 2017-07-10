#from classifier import GNB
from gnb_AZ import GNB_AZ
import json


def main():
    gnb = GNB_AZ()
    with open('train.json', encoding='utf-8') as f:
        j = json.load(f)
    print(j.keys())
    X = j['states']
    Y = j['labels']
    gnb.train(X, Y)

    with open('test.json', encoding='utf-8') as f:
        j = json.load(f)

    X = j['states']
    Y = j['labels']
    score = 0
    for coords, label in zip(X, Y):
        predicted = gnb.predict(coords)
        if predicted == label:
            score += 1
    fraction_correct = float(score) / len(X)
    print("You got {} percent correct".format(100 * fraction_correct))

main()
