# https://archive.ics.uci.edu/ml/datasets/Sepsis+survival+minimal+clinical+records#
from sklearn.ensemble import RandomForestClassifier
import csv
import time

X = [
  [72, 0, 1],
  [63, 0, 1],
  [89, 1, 2],
  [80, 0, 3],
  [62, 1, 3],
  [56, 1, 3],
  [63, 0, 1],
  [61, 0, 1],
  [66, 1, 1],
  [70, 1, 3],
  [49, 1, 1],
  [33, 1, 1],
  [33, 1, 2],
  [68, 0, 1],
  [68, 0, 2],
  [48, 0, 1],
  [49, 0, 2],
  [77, 0, 1],
  [39, 0, 1],
  [58, 1, 1],
]

Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

X_full = []
Y_full = []

file500 = open('sepsis_survival_primary_cohort_500_entries.csv')
file1000 = open('sepsis_survival_primary_cohort_1000_entries.csv')

# 20 entries only
t20Beginning = time.time()
clf = RandomForestClassifier(max_features=1.0, bootstrap=True, n_estimators=50, oob_score=True)
clf.fit(X, Y)
prediction = clf.predict(X)
correct = 0

for i in range(0,len(Y)):
    if Y[i] == prediction[i]:
        correct = correct + 1

t20End = time.time()
print("Score for 20 entries using python: " + str(correct/len(Y)))

# 500 entries
t500Beginning = time.time()
csvreader = csv.reader(file500)
next(csvreader)

for row in csvreader:
    row = [int(s) for s in row]
    X_full.append(row[0:3])
    Y_full.append(row[3])
        
file500.close()

clf_full = RandomForestClassifier(max_features=1.0, bootstrap=True, n_estimators=50, oob_score=True)
clf_full.fit(X_full, Y_full)
prediction_full = clf_full.predict(X_full)
correct = 0

for i in range(0,len(Y_full)):
    if Y_full[i] == prediction_full[i]:
        correct = correct + 1

t500End = time.time()   
print("Score for 500 entries using python: " + str(correct/len(Y_full)))

# 1000 entries 
t1000Beginning = time.time()   
csvreader = csv.reader(file1000)
next(csvreader)

for row in csvreader:
    row = [int(s) for s in row]
    X_full.append(row[0:3])
    Y_full.append(row[3])
        
file1000.close()

clf_full = RandomForestClassifier(max_features=1.0, bootstrap=True, n_estimators=50, oob_score=True)
clf_full.fit(X_full, Y_full)
prediction_full = clf_full.predict(X_full)
correct = 0

for i in range(0,len(Y_full)):
    if Y_full[i] == prediction_full[i]:
        correct = correct + 1
        
t1000End = time.time()      
print("Score for 1000 entries using python: " + str(correct/len(Y_full)))

print("Time for training 20 entries : " + str(t20End - t20Beginning) + " seconds")
print("Time for training 500 entries : " + str(t500End - t500Beginning) + " seconds")
print("Time for training 1000 entries : " + str(t1000End - t1000Beginning) + " seconds")