'use strict';

var RandomForestBase = require('./RandomForestBase');

var selectionMethods = {
    mean: mean,
    median: median
};

/**
 * @class RandomForestRegression
 * @augments RandomForestBase
 */
class RandomForestRegression extends RandomForestBase {

    /**
     * Create a new base random forest for a classifier or regression model.
     * @constructor
     * @param {Object} options
     * @param {Number|String} [options.maxFeatures] - the number of features used on each estimator.
     *        * if is a String it support two methods to get the max features, "sqrt" or "log2" over all
     *          sample features.
     *        * if is an integer it selects maxFeatures elements over the sample features.
     *        * if is a float between (0, 1), it takes the percentage of features.
     * @param {Boolean} [options.replacement] - use replacement over the sample features.
     * @param {Number} [options.seed] - seed for feature and samples selection, must be a 32-bit integer.
     * @param {Number} [options.nEstimators] - number of estimator to use.
     * @param {Object} [options.treeOptions] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
     * @param {String} [options.selectionMethod] - the way to calculate the prediction from estimators, "mean" and "median" are supported.
     */
    constructor(options, model) {
        if (options === true) {
            super(true, model.baseModel);
        } else {
            if (options === undefined) options = {};
            if (options.selectionMethod === undefined) options.selectionMethod = 'mean';

            if (!['mean', 'median'].includes(options.selectionMethod)) {
                throw new RangeError('Unsupported selection method ' + options.selectionMethod);
            }

            options.classifier = false;
            super(options);
        }
    }

    /**
     * retrieve the prediction given the selection method.
     * @param {Array} values - predictions of the estimators.
     * @returns {Number} prediction
     */
    selection(values) {
        return selectionMethods[this.options.selectionMethod](values);
    }

    /**
     * Export the current model to JSON.
     * @returns {Object} - Current model.
     */
    export() {
        var baseModel = super.export();
        return {
            baseModel: baseModel,
            name: 'RFRegression'
        };
    }

    /**
     * Load a Decision tree classifier with the given model.
     * @param {Object} model
     * @returns {RandomForestRegression}
     */
    static load(model) {
        if (model.name !== 'RFRegression') {
            throw new RangeError('Invalid model: ' + model.name);
        }

        return new RandomForestRegression(true, model);
    }
}


/**
 * Return the mean of the given array.
 * @param {array} values
 * @return {number} mean
 */
function mean(values) {
    var sum = 0;
    for (var i = 0; i < values.length; ++i) {
        sum += values[i];
    }

    return sum / values.length;
}

/**
 * Return the median of the given array.
 * @param {array} values
 * @return {number} median
 */
function median(values) {
    values.sort(function (a, b) {
        return a - b;
    });
    var half = Math.floor(values.length / 2);
    if (values.length % 2)
        return values[half];
    else
        return (values[half - 1] + values[half]) / 2.0;
}

module.exports = RandomForestRegression;
