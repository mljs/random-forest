import arrayMean from 'ml-array-mean';
import arrayMedian from 'ml-array-median';

import { RandomForestBase } from './RandomForestBase';

const selectionMethods = {
  mean: arrayMean,
  median: arrayMedian,
};

const defaultOptions = {
  maxFeatures: 1.0,
  replacement: false,
  nEstimators: 50,
  treeOptions: {},
  selectionMethod: 'mean',
  seed: 42,
  useSampleBagging: true,
  noOOB: false,
};

/**
 * @class RandomForestRegression
 * @augments RandomForestBase
 */
export class RandomForestRegression extends RandomForestBase {
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
   * @param {string} [options.selectionMethod="mean"] - the way to calculate the prediction from estimators, "mean" and "median" are supported.
   * @param {boolean} [options.useSampleBagging=true] - use bagging over training samples.
   * @param {number} [options.maxSamples=null] - if null, then draw X.shape[0] samples. If int, then draw maxSamples samples. If float, then draw maxSamples * X.shape[0] samples. Thus, maxSamples should be in the interval (0.0, 1.0].
   * @param {object} model - for load purposes.
   */
  constructor(options, model) {
    if (options === true) {
      super(true, model.baseModel);
      this.selectionMethod = model.selectionMethod;
    } else {
      options = Object.assign({}, defaultOptions, options);
      if (
        !(
          options.selectionMethod === 'mean' ||
          options.selectionMethod === 'median'
        )
      ) {
        throw new RangeError(
          `Unsupported selection method ${options.selectionMethod}`,
        );
      }

      options.isClassifier = false;

      super(options);
      this.selectionMethod = options.selectionMethod;
    }
  }

  /**
   * retrieve the prediction given the selection method.
   * @param {Array} values - predictions of the estimators.
   * @return {number} prediction
   */
  selection(values) {
    return selectionMethods[this.selectionMethod](values);
  }

  /**
   * Export the current model to JSON.
   * @return {object} - Current model.
   */
  toJSON() {
    let baseModel = super.toJSON();
    return {
      baseModel: baseModel,
      selectionMethod: this.selectionMethod,
      name: 'RFRegression',
    };
  }

  /**
   * Load a Decision tree classifier with the given model.
   * @param {object} model
   * @return {RandomForestRegression}
   */
  static load(model) {
    if (model.name !== 'RFRegression') {
      throw new RangeError(`Invalid model: ${model.name}`);
    }

    return new RandomForestRegression(true, model);
  }
}
