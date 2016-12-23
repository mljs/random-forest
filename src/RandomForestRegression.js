'use strict';

var RandomForestBase = require('./RandomForestBase');
var Stats = require('ml-stat');

var selectionMethods = {
    mean: Stats.array.mean,
    median: Stats.array.median
};

/**
 * @class RandomForestRegression
 * @augments RandomForestBase
 */
class RandomForestRegression extends RandomForestBase {

    /**
     * Create a new base random forest for a classifier or regression model.
     * @constructor
     * @param {object} options
     * @param {number} [options.maxFeatures] - the number of features used on each estimator.
     *        * if is an integer it selects maxFeatures elements over the sample features.
     *        * if is a float between (0, 1), it takes the percentage of features.
     * @param {boolean} [options.replacement] - use replacement over the sample features.
     * @param {number} [options.seed] - seed for feature and samples selection, must be a 32-bit integer.
     * @param {number} [options.nEstimators] - number of estimator to use.
     * @param {object} [options.treeOptions] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
     * @param {string} [options.selectionMethod] - the way to calculate the prediction from estimators, "mean" and "median" are supported.
     * @param {object} model - for load purposes.
     */
    constructor(options, model) {
        if (options === true) {
            super(true, model.baseModel);
        } else {
            if (options === undefined) options = {};
            if (options.selectionMethod === undefined) options.selectionMethod = 'mean';

            if (!(options.selectionMethod === 'mean' || options.selectionMethod === 'median')) {
                throw new RangeError('Unsupported selection method ' + options.selectionMethod);
            }

            options.classifier = false;
            super(options);
        }
    }

    /**
     * retrieve the prediction given the selection method.
     * @param {Array} values - predictions of the estimators.
     * @return {number} prediction
     */
    selection(values) {
        return selectionMethods[this.options.selectionMethod](values);
    }

    /**
     * Export the current model to JSON.
     * @return {object} - Current model.
     */
    toJSON() {
        var baseModel = super.toJSON();
        return {
            baseModel: baseModel,
            name: 'RFRegression'
        };
    }

    /**
     * Load a Decision tree classifier with the given model.
     * @param {object} model
     * @return {RandomForestRegression}
     */
    static load(model) {
        if (model.name !== 'RFRegression') {
            throw new RangeError('Invalid model: ' + model.name);
        }

        return new RandomForestRegression(true, model);
    }
}

module.exports = RandomForestRegression;
