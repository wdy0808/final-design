from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

label_file_name = "./data/cora/cora.content"
embedding_file_name = "data/cora/output.embeddings"
labels = []
datas = [[] for i in range(2708)]

with open(label_file_name) as f:
    index = 0
    for line in f:
        data = line.strip().split()[-1]
        labels.append(
            {
                'Case_Based': 0,
                'Genetic_Algorithms': 1,
                'Neural_Networks': 2,
                'Probabilistic_Methods': 3,
                'Reinforcement_Learning': 4,
                'Rule_Learning': 5,
                'Theory': 6
            }.get(data, 0)*1.0
        )

with open(embedding_file_name) as f:
    index = 0
    for line in f:
        data = line.strip().split()
        for d in data:
            datas[index].append(float(d))
        index = index + 1

percent = [0.9, 0.7, 0.5]
scores = [[]]

for train_perc in percent:

    Data_train, Data_test, label_train, label_test = train_test_split(
        datas, labels, test_size=train_perc, random_state=42)

    classifier = LogisticRegression(penalty='l2')
    scorer_micro = make_scorer(f1_score, average='micro')
    scorer_macro = make_scorer(f1_score, average='macro')
    scores_micro = cross_val_score(classifier, Data_train, label_train, cv=5, scoring=scorer_micro)
    scores_macro = cross_val_score(classifier, Data_train, label_train, cv=5, scoring=scorer_macro)
    #classifier.fit(Data_train, label_train)
    #x = classifier.predict(Data_test) 
    scores.append(scores_micro)
    scores.append(scores_macro)

print(scores)