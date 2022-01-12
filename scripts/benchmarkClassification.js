const RF = require('../random-forest.js');
const fs = require('fs');

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

const allEqual = (arr) => arr.every((val) => val === arr[0]);

// FULL DATASET
const pathFull = 'scripts/sepsis_survival_primary_cohort.csv';

// DATASET with 500 entries
const path500 = 'scripts/sepsis_survival_primary_cohort_500_entries.csv';

// DATASET with 1000 entries
const path1000 = 'scripts/sepsis_survival_primary_cohort_1000_entries.csv';

// DATASET with 5000 entries
const path5000 = 'scripts/sepsis_survival_primary_cohort_5000_entries.csv';

let classifier = new RF.RandomForestClassifier(options);
let xFull = [];
let yFull = [];

classification20Entries();
classification500Entries();
classification1000Entries();

// Takes Too Long
// classification5000Entries();
// classificationAllEntries();

function classification20Entries() {
  classifier.train(X, Y);
  // classifier.printTrees();
  let result = classifier.predict(X);
  let correct = result.reduce((previous, result, index) => {
    return result === Y[index] ? previous + 1 : previous;
  }, 0);

  let score = correct / result.length;

  console.log('Score for the dataset with ', 20, ' entries: ', score);
  console.log('The predictions for ', 20, ' entries: ', result);
  console.log('Are all values equal for ', 20, ' entries :', allEqual(result));
}

function classification500Entries() {
  classification(path500, 500);
}

function classification1000Entries() {
  classification(path1000, 1000);
}

function classification5000Entries() {
  classification(path5000, 5000);
}

function classificationAllEntries() {
  classification(pathFull, 'all');
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

function classification(path, numberEntries) {
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

    classifier.train(xFull, yFull);
    // classifier.printTrees();
    let result = classifier.predict(xFull);

    const correct = result.reduce((previous, result, index) => {
      return result === yFull[index] ? previous + 1 : previous;
    }, 0);

    let score = correct / result.length;
    callback(numberEntries, score, result);
  });
}
