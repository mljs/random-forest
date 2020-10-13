# ml-random-forest

[![NPM version][npm-image]][npm-url]
[![build status][travis-image]][travis-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

Random forest for classification and regression.

## Installation

`npm install --save ml-random-forest`

## [API Documentation](https://mljs.github.io/random-forest/)

## Usage

### As classifier

```js
import IrisDataset from 'ml-dataset-iris';
import { RandomForestClassifier as RFClassifier } from 'ml-random-forest';

var trainingSet = IrisDataset.getNumbers();
var predictions = IrisDataset.getClasses().map((elem) =>
  IrisDataset.getDistinctClasses().indexOf(elem)
);

var options = {
  seed: 3,
  maxFeatures: 0.8,
  replacement: true,
  nEstimators: 25
};

var classifier = new RFClassifier(options);
classifier.train(trainingSet, predictions);
var result = classifier.predict(trainingSet);
var oobResult = classifier.predictOOB();
var confusionMatrix = classifier.getConfusionMatrix();
```

### As regression

```js
import { RandomForestRegression as RFRegression } from 'ml-random-forest';

var dataset = [
  [73, 80, 75, 152],
  [93, 88, 93, 185],
  [89, 91, 90, 180],
  [96, 98, 100, 196],
  [73, 66, 70, 142],
  [53, 46, 55, 101],
  [69, 74, 77, 149],
  [47, 56, 60, 115],
  [87, 79, 90, 175],
  [79, 70, 88, 164],
  [69, 70, 73, 141],
  [70, 65, 74, 141],
  [93, 95, 91, 184],
  [79, 80, 73, 152],
  [70, 73, 78, 148],
  [93, 89, 96, 192],
  [78, 75, 68, 147],
  [81, 90, 93, 183],
  [88, 92, 86, 177],
  [78, 83, 77, 159],
  [82, 86, 90, 177],
  [86, 82, 89, 175],
  [78, 83, 85, 175],
  [76, 83, 71, 149],
  [96, 93, 95, 192]
];

var trainingSet = new Array(dataset.length);
var predictions = new Array(dataset.length);

for (var i = 0; i < dataset.length; ++i) {
  trainingSet[i] = dataset[i].slice(0, 3);
  predictions[i] = dataset[i][3];
}

var options = {
  seed: 3,
  maxFeatures: 2,
  replacement: false,
  nEstimators: 200
};

var regression = new RFRegression(options);
regression.train(trainingSet, predictions);
var result = regression.predict(trainingSet);
```

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-random-forest.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-random-forest
[travis-image]: https://img.shields.io/travis/mljs/random-forest/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/random-forest
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/random-forest.svg?style=flat-square
[codecov-url]: https://codecov.io/github/mljs/random-forest
[download-image]: https://img.shields.io/npm/dm/ml-random-forest.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-random-forest
