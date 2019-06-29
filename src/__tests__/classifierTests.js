import IrisDataset from 'ml-dataset-iris';
import Matrix from 'ml-matrix';

import { RandomForestClassifier as RFClassifier } from '..';


describe('Random Forest Classifier', function () {
  var trainingSet = IrisDataset.getNumbers();
  var predictions = IrisDataset.getClasses().map((elem) =>
    IrisDataset.getDistinctClasses().indexOf(elem)
  );

  var options = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined, // default options for the decision tree
    useSampleBagging: true
  };

  var classifier = new RFClassifier(options);
  classifier.train(trainingSet, predictions);
  var result = classifier.predict(trainingSet);

  it('Random Forest Classifier with iris dataset', function () {
    var correct = 0;
    for (var i = 0; i < result.length; ++i) {
      if (result[i] === predictions[i]) correct++;
    }

    var score = correct / result.length;
    expect(score).toBeGreaterThanOrEqual(0.7); // above or equal
  });

  it('Export and import for random forest classifier', () => {
    var model = JSON.parse(JSON.stringify(classifier));

    var newClassifier = RFClassifier.load(model);
    var newResult = newClassifier.predict(trainingSet);

    for (var i = 0; i < result.length; ++i) {
      expect(newResult[i]).toBe(result[i]);
    }
  });

  it('Test with a 2 features dataset', function () {
    var X = new Matrix([
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ]);
    var Y = [
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0
    ];

    // the test set (Xtest, Ytest)
    var Xtest = new Matrix([
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ]);
    var Ytest = [1, 1, 1, 0, 1, 1, 1, 0];

    // we will train our model
    var rf = new RFClassifier({ nEstimators: 50 });
    rf.train(X, Y);

    // we try to predict the test set
    var finalResults = rf.predict(Xtest);
    for (var i = 0; i < Ytest.rows; ++i) {
      expect(finalResults[i]).toBe(Ytest[i][0]);
    }
  });

  it('Test with full features dataset', function () {
    var X = new Matrix([
      [0, -1],
      [1, 0],
      [1, 1],
      [1, -1],
      [2, 0],
      [2, 1],
      [2, -1],
      [3, 2],
      [0, 4],
      [1, 3],
      [1, 4],
      [1, 5],
      [2, 3],
      [2, 4],
      [2, 5],
      [3, 4],
      [1, 10],
      [1, 12],
      [2, 10],
      [2, 11],
      [2, 14],
      [3, 11]
    ]);
    var Y = [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      2,
      2,
      2,
      2,
      2,
      2
    ];

    // the test set (Xtest, Ytest)
    var Xtest = new Matrix([
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 2.5],
      [2, 3.5],
      [1.5, 4],
      [1, 10.5],
      [2.5, 10.5],
      [2, 11.5]
    ]);
    var Ytest = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    // we will train our model
    var rf = new RFClassifier({ nEstimators: 50 });
    rf.train(X, Y);

    // we try to predict the test set
    var finalResults = rf.predict(Xtest);
    for (var i = 0; i < Ytest.rows; ++i) {
      expect(finalResults[i]).toBe(Ytest[i][0]);
    }
  });
});
