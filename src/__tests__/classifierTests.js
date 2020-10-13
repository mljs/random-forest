import IrisDataset from 'ml-dataset-iris';
import Matrix from 'ml-matrix';

import { RandomForestClassifier as RFClassifier } from '..';

describe('Random Forest Classifier', function() {
  let trainingSet = IrisDataset.getNumbers();
  let predictions = IrisDataset.getClasses().map((elem) =>
    IrisDataset.getDistinctClasses().indexOf(elem),
  );

  let options = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined, // default options for the decision tree
    useSampleBagging: true,
  };

  let classifier = new RFClassifier(options);
  classifier.train(trainingSet, predictions);
  let result = classifier.predict(trainingSet);

  it('Random Forest Classifier with iris dataset', function() {
    let correct = 0;
    for (let i = 0; i < result.length; ++i) {
      if (result[i] === predictions[i]) correct++;
    }

    let score = correct / result.length;
    expect(score).toBeGreaterThanOrEqual(0.7); // above or equal
  });

  it('Export and import for random forest classifier', () => {
    let model = JSON.parse(JSON.stringify(classifier));

    let newClassifier = RFClassifier.load(model);
    let newResult = newClassifier.predict(trainingSet);

    for (let i = 0; i < result.length; ++i) {
      expect(newResult[i]).toBe(result[i]);
    }
  });

  it('Test with a 2 features dataset', function() {
    let X = new Matrix([
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
      [0, 0],
    ]);
    let Y = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0];

    // the test set (Xtest, Ytest)
    let Xtest = new Matrix([
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0],
    ]);
    let Ytest = [1, 1, 1, 0, 1, 1, 1, 0];

    // we will train our model
    let rf = new RFClassifier({ nEstimators: 50 });
    rf.train(X, Y);

    // we try to predict the test set
    let finalResults = rf.predict(Xtest);
    for (let i = 0; i < Ytest.rows; ++i) {
      expect(finalResults[i]).toBe(Ytest[i][0]);
    }
  });

  it('Test with full features dataset', function() {
    let X = new Matrix([
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
      [3, 11],
    ]);
    let Y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];

    // the test set (Xtest, Ytest)
    let Xtest = new Matrix([
      [0, -2],
      [1, 0.5],
      [1.5, -1],
      [1, 2.5],
      [2, 3.5],
      [1.5, 4],
      [1, 10.5],
      [2.5, 10.5],
      [2, 11.5],
    ]);
    let Ytest = [0, 0, 0, 1, 1, 1, 2, 2, 2];

    // we will train our model
    let rf = new RFClassifier({ nEstimators: 50 });
    rf.train(X, Y);

    // we try to predict the test set
    let finalResults = rf.predict(Xtest);
    for (let i = 0; i < Ytest.rows; ++i) {
      expect(finalResults[i]).toBe(Ytest[i][0]);
    }
  });
  it('Random Forest Classifier with iris dataset - probability', function() {
    let opts = {
      seed: 17,
      nEstimators: 100,
      treeOptions: undefined, // default options for the decision tree
      useSampleBagging: true,
    };
    let classifierProb = new RFClassifier(opts);

    const n = 147;
    const toPredict = trainingSet.slice(n);
    const toTrain = trainingSet.slice(0, n);
    const trainLabel = predictions.slice(0, n);

    classifierProb.train(toTrain, trainLabel);

    const probabilities = classifierProb.predictProbability(toPredict, 2);
    expect(
      probabilities.reduce((p, v) => Math.min(p, v), 1),
    ).toBeGreaterThanOrEqual(0.7);
    })
    //expect(score).toBeGreaterThanOrEqual(0.7); // above or equal
  it('Test Out-Of-Bag estimates', () => {
    let opts = {
      seed: 17,
      replacement: false,
      nEstimators: 100,
      treeOptions: { minNumSamples: 1 }, // default options for the decision tree
      useSampleBagging: true,
    };

    let OOBclassifier = new RFClassifier(opts);
    OOBclassifier.train(trainingSet, predictions);
    const confusionMatrix = OOBclassifier.getConfusionMatrix();
    const correctVsTotal = confusionMatrix.reduce(
      (p, v, i) => {
        p.correct += v[i];
        p.total += v.reduce((q, w) => q + w, 0);
        return p;
      },
      { correct: 0, total: 0 },
    );
    expect(
      (100 * correctVsTotal.correct) / correctVsTotal.total,
    ).toBeGreaterThanOrEqual(95.0);
  });
})
