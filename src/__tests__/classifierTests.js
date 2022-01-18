import { getClasses, getDistinctClasses, getNumbers } from 'ml-dataset-iris';
import { Matrix } from 'ml-matrix';

import { RandomForestClassifier as RFClassifier } from '..';

describe('Random Forest Classifier', () => {
  let trainingSet = getNumbers();
  let predictions = getClasses().map((elem) =>
    getDistinctClasses().indexOf(elem),
  );

  let options130Max = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined,
    useSampleBagging: true,
    maxSamples: 130,
  };

  let classifier130Max = new RFClassifier(options130Max);
  classifier130Max.train(trainingSet, predictions);
  let result130Max = classifier130Max.predict(trainingSet);

  it('Random Forest Classifier with iris dataset and 130 maximum samples', () => {
    const correct = result130Max.reduce((previous, result, index) => {
      return result === predictions[index] ? previous + 1 : previous;
    }, 0);

    let score = correct / result130Max.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  let options200Max = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined,
    useSampleBagging: true,
    maxSamples: 200,
  };

  it('Should throw error: maxSamples bigger than 150', () => {
    const t = () => {
      let classifier200Max = new RFClassifier(options200Max);
      classifier200Max.train(trainingSet, predictions);
    };
    expect(t).toThrow(RangeError);
  });

  let optionsMaxFraction = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined,
    useSampleBagging: true,
    maxSamples: 0.9,
  };

  let classifierMaxFraction = new RFClassifier(optionsMaxFraction);
  classifierMaxFraction.train(trainingSet, predictions);
  let resultMaxFraction = classifierMaxFraction.predict(trainingSet);

  it('Random Forest Classifier with iris dataset and 0.9 fraction of total dataset for maximum samples', () => {
    const correct = resultMaxFraction.reduce((previous, result, index) => {
      return result === predictions[index] ? previous + 1 : previous;
    }, 0);

    let score = correct / resultMaxFraction.length;
    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  let optionsMaxFractionError = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined,
    useSampleBagging: true,
    maxSamples: 1.8,
  };

  it('Should throw error: if maxSamples is a float, it should be between 0 and 1.0', () => {
    const t = () => {
      let classifierMaxFractionError = new RFClassifier(
        optionsMaxFractionError,
      );
      classifierMaxFractionError.train(trainingSet, predictions);
    };
    expect(t).toThrow(RangeError);
  });

  let optionsMaxNegativeError = {
    seed: 3,
    maxFeatures: 0.8,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined,
    useSampleBagging: true,
    maxSamples: -2,
  };

  it('Should throw error: maxSamples should be positive', () => {
    const t = () => {
      let classifierMaxNegativeError = new RFClassifier(
        optionsMaxNegativeError,
      );
      classifierMaxNegativeError.train(trainingSet, predictions);
    };
    expect(t).toThrow(RangeError);
  });

  let optionsMaxFeatureError = {
    seed: 3,
    maxFeatures: 1.5,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined,
    useSampleBagging: true,
  };

  it('Should throw error: maxFeatures should be an integer or a float between 0 and 1.0', () => {
    const t = () => {
      let classifierMaxFeatureError = new RFClassifier(optionsMaxFeatureError);
      classifierMaxFeatureError.train(trainingSet, predictions);
    };
    expect(t).toThrow(RangeError);
  });

  let optionsMaxFeatureColumnError = {
    seed: 3,
    maxFeatures: 200,
    replacement: true,
    nEstimators: 25,
    treeOptions: undefined,
    useSampleBagging: true,
  };

  it('Should throw error: maxFeatures should be smaller than the number of columns of the training set', () => {
    const t = () => {
      let classifierMaxFeatureColumnError = new RFClassifier(
        optionsMaxFeatureColumnError,
      );
      classifierMaxFeatureColumnError.train(trainingSet, predictions);
    };
    expect(t).toThrow(RangeError);
  });

  it('Feature importances should sum to 1', () => {
    let featureImportances = classifierMaxFraction.featureImportance();
    let sum = featureImportances.reduce((a, b) => {
      return a + b;
    }, 0);
    expect(sum).toBeGreaterThanOrEqual(1);
  });

  it('Export and import for random forest classifier', () => {
    let model = JSON.parse(JSON.stringify(classifier130Max));

    let newClassifier = RFClassifier.load(model);
    let newResult = newClassifier.predict(trainingSet);

    for (let i = 0; i < result130Max.length; ++i) {
      expect(newResult[i]).toBe(result130Max[i]);
    }
  });

  it('Test with a 2 features dataset', () => {
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

    let rf = new RFClassifier({ nEstimators: 50 });
    rf.train(X, Y);

    let finalResults = rf.predict(Xtest);
    for (let i = 0; i < Ytest.rows; ++i) {
      expect(finalResults[i]).toBe(Ytest[i][0]);
    }
  });

  it('Test with full features dataset', () => {
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

    let rf = new RFClassifier({ nEstimators: 50 });
    rf.train(X, Y);

    let finalResults = rf.predict(Xtest);
    for (let i = 0; i < Ytest.rows; ++i) {
      expect(finalResults[i]).toBe(Ytest[i][0]);
    }
  });

  it('Random Forest Classifier with iris dataset - probability', () => {
    let opts = {
      seed: 17,
      nEstimators: 100,
      treeOptions: undefined,
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
  });

  it('Test Out-Of-Bag estimates', () => {
    let opts = {
      seed: 17,
      replacement: false,
      nEstimators: 100,
      treeOptions: { minNumSamples: 1 },
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
});
