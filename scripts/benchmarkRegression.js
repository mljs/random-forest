const RF = require('../random-forest.js');
const fs = require('fs');

function approx(val, expected, eps) {
  return val - eps < expected && expected < val + eps;
}

let X = [
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
];

let Y = [
  34055.6962, 29814.6835, 29128.1012, 28228.8607, 27335.6962, 26624.8101,
  25998.9873, 25446.0759, 24777.7215, 24279.4936, 23896.7088, 23544.3038,
  23003.5443, 22329.1139, 22092.1519, 21903.7974, 21685.0632, 21484.5569,
  21107.8481, 20998.481,
];

let options = {
  seed: 3,
  maxFeatures: 1.0,
  replacement: true,
  nEstimators: 50,
  treeOptions: undefined,
  useSampleBagging: false,
};

const allEqual = (arr) => arr.every((val) => val === arr[0]);

// FULL DATASET
const pathFull = 'scripts/tetuan_city_power_consumption.csv';

// DATASET with 500 entries
const path500 = 'scripts/tetuan_city_power_consumption_500_entries.csv';

// DATASET with 1000 entries
const path1000 = 'scripts/tetuan_city_power_consumption_1000_entries.csv';

// DATASET with 5000 entries
const path5000 = 'scripts/tetuan_city_power_consumption_5000_entries.csv';

let regressor = new RF.RandomForestRegression(options);
let xFull = [];
let yFull = [];

// regression20Entries();
// regression500Entries();
regression1000Entries(); // sometimes gives null values

// Takes Too Long
// regression5000Entries();
// regressionAllEntries();

function regression20Entries() {
  regressor.train(X, Y);
  // regressor.printTrees();
  let result = regressor.predict(X);

  const correct = result.reduce((prev, value, index) => {
    return approx(value, Y[index], 1000) ? prev + 1 : prev;
  }, 0);

  let score = correct / result.length;

  console.log('Score for the dataset with ', 20, ' entries: ', score);
  console.log('The predictions for ', 20, ' entries: ', result);
  console.log('Are all values equal for ', 20, ' entries :', allEqual(result));
}

function regression500Entries() {
  regression(path500, 500);
}

function regression1000Entries() {
  regression(path1000, 1000);
}

function regression5000Entries() {
  regression(path5000, 5000);
}

function regressionAllEntries() {
  regression(pathFull, 'all');
}

function callback(numberEntries, score, result) {
  try {
    console.log(
      'Score for the dataset with ',
      numberEntries,
      ' entries: ',
      score,
    );
    console.log('The predictions for ', numberEntries, ' entries: ', result);
    console.log(
      'Are all values equal for ',
      numberEntries,
      ' entries :',
      allEqual(result),
    );
  } catch (error) {
    console.log(error);
  }
}

function regression(path, numberEntries) {
  fs.readFile(path, 'utf8', function (err, data) {
    let dataFinal = data.split(/\r?\n/);

    for (let i = 0; i < dataFinal.length; i++) {
      let intermediate = dataFinal[i].split(',');
      let arrayToInt = intermediate.map(function (x) {
        return parseInt(x, 10);
      });
      let xPart = arrayToInt.slice(1, 6);
      let yPart = arrayToInt[6];
      xFull.push(xPart);
      yFull.push(yPart);
    }

    xFull.shift();
    xFull.pop();
    yFull.shift();
    yFull.pop();

    regressor.train(xFull, yFull);
    // regressor.printTrees();
    let result = regressor.predict(xFull);

    const correct = result.reduce((prev, value, index) => {
      return approx(value, yFull[index], 1000) ? prev + 1 : prev;
    }, 0);

    let score = correct / result.length;
    callback(numberEntries, score, result);
  });
}
