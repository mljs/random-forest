'use strict';

var RandomForestBase = require('./RandomForestBase');

/**
 * @class RandomForestClassifier
 * @augments RandomForestBase
 */
class RandomForestClassifier extends RandomForestBase {

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
     */
    constructor(options, model) {
        if (options === true) {
            super(true, model.baseModel);
        } else {
            if (options === undefined) options = {};
            options.classifier = true;
            super(options);
        }
    }

    /**
     * retrieve the prediction given the selection method.
     * @param {Array} values - predictions of the estimators.
     * @returns {Number} prediction
     */
    selection(values) {
        return mode(values);
    }

    /**
     * Export the current model to JSON.
     * @returns {Object} - Current model.
     */
    export() {
        var baseModel = super.export();
        return {
            baseModel: baseModel,
            name: 'RFClassifier'
        };
    }

    /**
     * Load a Decision tree classifier with the given model.
     * @param {Object} model
     * @returns {RandomForestClassifier}
     */
    static load(model) {
        if (model.name !== 'RFClassifier') {
            throw new RangeError('Invalid model: ' + model.name);
        }

        return new RandomForestClassifier(true, model);
    }
}

/**
 * Return the most repeated element on the array.
 * @param {Array} arr
 * @return {Number} mode
 */
function mode(arr) {
    return arr.sort((a, b) =>
        arr.filter(v => v === a).length
        - arr.filter(v => v === b).length
    ).pop();
}

module.exports = RandomForestClassifier;
