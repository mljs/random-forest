const RF = require('../random-forest.js');
const fs = require('fs');

function approx(val, expected, eps) {
  return val - eps < expected && expected < val + eps;
}

let X = [
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
];

let Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

let options = {
  seed: 3,
  maxFeatures: 1.0,
  replacement: true,
  nEstimators: 50,
  treeOptions: undefined,
  useSampleBagging: true,
};

// DATASET with 500 entries
const path500 = 'scripts/sepsis_survival_primary_cohort_500_entries.csv';

// DATASET with 10000 entries
const path1000 = 'scripts/sepsis_survival_primary_cohort_1000_entries.csv';

// DATASET with 5000 entries
const path5000 = 'scripts/sepsis_survival_primary_cohort_5000_entries.csv';

// DATASET with 1000 entries
const path10000 = 'scripts/sepsis_survival_primary_cohort_10000_entries.csv';

// FULL DATASET
const pathFull = 'scripts/sepsis_survival_primary_cohort.csv';

let regressor = new RF.RandomForestRegression(options);
let xFull = [];
let yFull = [];

regression20Entries();
regression500Entries();
regression1000Entries();
regression5000Entries();

// Takes Too Long
// regression10000Entries();
// regressionAllEntries();

function regression20Entries() {
  regressor.train(X, Y);
  let result = regressor.predict(X);

  const correct = result.reduce((prev, value, index) => {
    return approx(value, Y[index], 5) ? prev + 1 : prev; // from 10 changed to 5
  }, 0);

  let score = correct / result.length;
  console.log('Score for 20 entries: ', score);
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

function regression10000Entries() {
  regression(path10000, 10000);
}

function regressionAllEntries() {
  regression(pathFull, 'all');
}

function callback(numberEntries, score) {
  try {
    console.log(
      'Score for the dataset with ',
      numberEntries,
      ' entries: ',
      score,
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
      let xPart = arrayToInt.slice(0, 3);
      let yPart = arrayToInt[3];
      xFull.push(xPart);
      yFull.push(yPart);
    }

    xFull.shift();
    xFull.pop();
    yFull.shift();
    yFull.pop();

    regressor.train(xFull, yFull);
    let result = regressor.predict(xFull);
    const correct = result.reduce((prev, value, index) => {
      return approx(value, yFull[index], 5) ? prev + 1 : prev; // from 10 changed to 5
    }, 0);

    let score = correct / result.length;
    callback(numberEntries, score);
  });
}
