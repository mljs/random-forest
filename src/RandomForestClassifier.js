import arrayMode from 'ml-array-mode';

import { RandomForestBase } from './RandomForestBase';

const defaultOptions = {
  maxFeatures: 1.0,
  replacement: true,
  nEstimators: 50,
  seed: 42,
  useSampleBagging: true,
  noOOB: false,
};

/**
 * @class RandomForestClassifier
 * @augments RandomForestBase
 */
export class RandomForestClassifier extends RandomForestBase {
  /**
   * Create a new base random forest for a classifier or regression model.
   * @constructor
   * @param {object} options
   * @param {number} [options.maxFeatures=1.0] - the number of features used on each estimator.
   *        * if is an integer it selects maxFeatures elements over the sample features.
   *        * if is a float between (0, 1), it takes the percentage of features.
   * @param {boolean} [options.replacement=true] - use replacement over the sample features.
   * @param {number} [options.seed=42] - seed for feature and samples selection, must be a 32-bit integer.
   * @param {number} [options.nEstimators=50] - number of estimator to use.
   * @param {object} [options.treeOptions={}] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
   * @param {boolean} [options.useSampleBagging=true] - use bagging over training samples.
   * @param {number} [options.maxSamples=null] - if null, then draw X.shape[0] samples. If int, then draw maxSamples samples. If float, then draw maxSamples * X.shape[0] samples. Thus, maxSamples should be in the interval (0.0, 1.0].
   * @param {object} model - for load purposes.
   */
  constructor(options, model) {
    if (options === true) {
      super(true, model.baseModel);
    } else {
      options = Object.assign({}, defaultOptions, options);
      options.isClassifier = true;
      super(options);
    }
  }

  /**
   * retrieve the prediction given the selection method.
   * @param {Array} values - predictions of the estimators.
   * @return {number} prediction
   */
  selection(values) {
    return arrayMode(values);
  }

  /**
   * Export the current model to JSON.
   * @return {object} - Current model.
   */
  toJSON() {
    let baseModel = super.toJSON();
    return {
      baseModel: baseModel,
      name: 'RFClassifier',
    };
  }

  /**
   * Returns the confusion matrix
   * Make sure to run train first.
   * @return {object} - Current model.
   */
  getConfusionMatrix() {
    if (!this.oobResults) {
      throw new Error('No Out-Of-Bag results available.');
    }

    const labels = new Set();
    const matrix = this.oobResults.reduce((p, v) => {
      labels.add(v.true);
      labels.add(v.predicted);
      const x = p[v.predicted] || {};
      x[v.true] = (x[v.true] || 0) + 1;
      p[v.predicted] = x;
      return p;
    }, {});
    const sortedLabels = [...labels].sort();

    return sortedLabels.map((v) =>
      sortedLabels.map((w) => (matrix[v] || {})[w] || 0),
    );
  }

  /**
   * Load a Decision tree classifier with the given model.
   * @param {object} model
   * @return {RandomForestClassifier}
   */
  static load(model) {
    if (model.name !== 'RFClassifier') {
      throw new RangeError(`Invalid model: ${model.name}`);
    }

    return new RandomForestClassifier(true, model);
  }

  /**
   * Predicts the probability of a label given the matrix to predict.
   * @param {Matrix|Array} toPredict
   * @param {number} label
   * @return {Array} predictions
   */
  predictProbability(toPredict, label) {
    const predictionValues = this.predictionValues(toPredict);
    let predictions = new Array(predictionValues.rows);
    for (let i = 0; i < predictionValues.rows; ++i) {
      const pvs = predictionValues.getRow(i);
      const l = pvs.length;
      const roundFactor = Math.pow(10, 6);
      predictions[i] =
        Math.round(
          pvs.reduce((p, v) => {
            if (v === label) {
              p += roundFactor / l;
            }
            return p;
          }),
        ) / roundFactor;
    }

    return predictions;
  }
}
