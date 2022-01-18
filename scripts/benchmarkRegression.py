# https://archive.ics.uci.edu/ml/datasets/Sepsis+survival+minimal+clinical+records#

from sklearn.ensemble import RandomForestRegressor
import csv
import time

def approx(val, expected, eps):
  return val - eps < expected and expected < val + eps

X = [
  [6.559, 73.8, 0.083, 0.051, 0.119],
  [6.414, 74.5, 0.083, 0.07, 0.085],
  [6.313, 74.5, 0.08, 0.062, 0.1],
  [6.121, 75.0, 0.083, 0.091, 0.096],
  [5.921, 75.7, 0.081, 0.048, 0.085],
  [5.853, 76.9, 0.081, 0.059, 0.108],
  [5.641, 77.7, 0.08, 0.048, 0.096],
  [5.496, 78.2, 0.085, 0.055, 0.093],
  [5.678, 78.1, 0.081, 0.066, 0.141],
  [5.491, 77.3, 0.082, 0.062, 0.111],
  [5.516, 77.5, 0.081, 0.051, 0.108],
  [5.471, 76.7, 0.083, 0.059, 0.126],
  [5.059, 78.6, 0.081, 0.07, 0.096],
  [4.968, 78.8, 0.084, 0.07, 0.134],
  [4.975, 78.9, 0.083, 0.055, 0.152],
  [4.897, 79.1, 0.083, 0.07, 0.096],
  [5.02, 79.7, 0.081, 0.051, 0.134],
  [5.407, 78.5, 0.082, 0.062, 0.163],
  [5.169, 77.9, 0.083, 0.066, 0.108],
  [5.081, 77.7, 0.084, 0.051, 0.13],
]

Y = [
  34055.6962, 29814.6835, 29128.1012, 28228.8607, 27335.6962, 26624.8101,
  25998.9873, 25446.0759, 24777.7215, 24279.4936, 23896.7088, 23544.3038,
  23003.5443, 22329.1139, 22092.1519, 21903.7974, 21685.0632, 21484.5569,
  21107.8481, 20998.481,
]

X_full = []
Y_full = []

file500 = open('tetuan_city_power_consumption_500_entries.csv')
file1000 = open('tetuan_city_power_consumption_1000_entries.csv')

# 20 entries only
t20Beginning = time.time()

clf = RandomForestRegressor(max_features=1.0, bootstrap=True, n_estimators=50, oob_score=True)
clf.fit(X, Y)
prediction = clf.predict(X)
correct = 0

for i in range(0,len(Y)):
    if approx(Y[i], prediction[i], 1000):
        correct = correct + 1

t20End = time.time()
#print("Prediction for 20 entries using python: " + str(prediction))
print("Score for 20 entries using python: " + str(correct/len(Y)))

# 500 entries
t500Beginning = time.time()
csvreader = csv.reader(file500)
next(csvreader)

for row in csvreader:
    row = [s for s in row]
    row.pop(0)
    row = [float(s) for s in row]
    X_full.append(row[1:6])
    Y_full.append(row[6])
        
file500.close()

clf_full = RandomForestRegressor(max_features=1.0, bootstrap=True, n_estimators=50, oob_score=True)
clf_full.fit(X_full, Y_full)
prediction_full = clf_full.predict(X_full)
correct = 0

for i in range(0,len(Y_full)):
    if approx(Y_full[i], prediction_full[i], 1000):
        correct = correct + 1

t500End = time.time()
# print("Prediction for 500 entries using python: " + str(prediction_full))
print("Score for 500 entries using python: " + str(correct/len(Y_full)))

# 1000 entries 
t1000Beginning = time.time()   
csvreader = csv.reader(file1000)
next(csvreader)

for row in csvreader:
    row = [s for s in row]
    row.pop(0)
    row = [float(s) for s in row]
    X_full.append(row[1:6])
    Y_full.append(row[6])
        
file1000.close()

clf_full = RandomForestRegressor(max_features=1.0, bootstrap=True, n_estimators=50, oob_score=True)
clf_full.fit(X_full, Y_full)
prediction_full = clf_full.predict(X_full)
correct = 0

for i in range(0,len(Y_full)):
    if approx(Y_full[i], prediction_full[i], 1000):
        correct = correct + 1
        
t1000End = time.time()
# print("Prediction for 500 entries using python: " + str(prediction_full))
print("Score for 1000 entries using python: " + str(correct/len(Y_full)))

print("Time for training 20 entries : " + str(t20End - t20Beginning) + " seconds")
print("Time for training 500 entries : " + str(t500End - t500Beginning) + " seconds")
print("Time for training 1000 entries : " + str(t1000End - t1000Beginning) + " seconds")
