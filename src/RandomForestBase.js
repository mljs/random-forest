import {
    DecisionTreeClassifier as DTClassfier,
    DecisionTreeRegression as DTRegression
} from 'ml-cart';
import * as Utils from './utils';
import Matrix from 'ml-matrix';

/**
 * @class RandomForestBase
 */
export default class RandomForestBase {

    /**
     * Create a new base random forest for a classifier or regression model.
     * @constructor
     * @param {object} options
     * @param {number|String} [options.maxFeatures] - the number of features used on each estimator.
     *        * if is an integer it selects maxFeatures elements over the sample features.
     *        * if is a float between (0, 1), it takes the percentage of features.
     * @param {boolean} [options.replacement] - use replacement over the sample features.
     * @param {number} [options.seed] - seed for feature and samples selection, must be a 32-bit integer.
     * @param {number} [options.nEstimators] - number of estimator to use.
     * @param {object} [options.treeOptions] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
     * @param {boolean} [options.isClassifier] - boolean to check if is a classifier or regression model (used by subclasses).
     * @param {object} model - for load purposes.
     */
    constructor(options, model) {
        if (options === true) {
            this.replacement = model.replacement;
            this.maxFeatures = model.maxFeatures;
            this.nEstimators = model.nEstimators;
            this.treeOptions = model.treeOptions;
            this.isClassifier = model.isClassifier;
            this.seed = model.seed;
            this.n = model.n;
            this.indexes = model.indexes;

            var Estimator = this.isClassifier ? DTClassfier : DTRegression;
            this.estimators = model.estimators.map(est => Estimator.load(est));
        } else {
            this.replacement = options.replacement;
            this.maxFeatures = options.maxFeatures;
            this.nEstimators = options.nEstimators;
            this.treeOptions = options.treeOptions;
            this.isClassifier = options.isClassifier;
            this.seed = options.seed;
        }
    }

    /**
     * Train the decision tree with the given training set and labels.
     * @param {Matrix|Array} trainingSet
     * @param {Array} trainingValues
     */
    train(trainingSet, trainingValues) {
        trainingSet = Matrix.checkMatrix(trainingSet);

        this.maxFeatures = this.maxFeatures || trainingSet.columns;

        if (Number.isInteger(this.maxFeatures)) {
            if (this.maxFeatures > trainingSet.columns) {
                throw new RangeError('The maxFeatures parameter should be less than ' + trainingSet.columns);
            } else {
                this.n = this.maxFeatures;
            }
        } else if (Utils.checkFloat(this.maxFeatures)) {
            this.n = Math.floor(trainingSet.columns * this.maxFeatures);
        } else {
            throw new RangeError('Cannot process the maxFeatures parameter ' + this.maxFeatures);
        }


        if (this.isClassifier) {
            var Estimator = DTClassfier;
        } else {
            Estimator = DTRegression;
        }

        this.estimators = new Array(this.nEstimators);
        this.indexes = new Array(this.nEstimators);

        for (var i = 0; i < this.nEstimators; ++i) {
            var res = Utils.examplesBaggingWithReplacement(trainingSet, trainingValues, this.seed);
            var X = res.X;
            var y = res.y;

            res = Utils.featureBagging(X, this.n, this.replacement, this.seed);
            X = res.X;

            this.indexes[i] = res.usedIndex;
            this.estimators[i] = new Estimator(this.treeOptions);
            this.estimators[i].train(X, y);
        }
    }

    /**
     * Method that returns the way the algorithm generates the predictions, for example, in classification
     * you can return the mode of all predictions retrieved by the trees, or in case of regression you can
     * use the mean or the median.
     * @abstract
     * @param {Array} values - predictions of the estimators.
     * @return {number} prediction.
     */
    // eslint-disable-next-line no-unused-vars
    selection(values) {
        throw new Error('Abstract method \'selection\' not implemented!');
    }

    /**
     * Predicts the output given the matrix to predict.
     * @param {Matrix|Array} toPredict
     * @return {Array} predictions
     */
    predict(toPredict) {
        var predictionValues = new Array(this.nEstimators);
        for (var i = 0; i < this.nEstimators; ++i) {
            var X = Utils.retrieveFeatures(new Matrix(toPredict), this.indexes[i]);
            predictionValues[i] = this.estimators[i].predict(X);
        }

        predictionValues = new Matrix(predictionValues).transpose();
        var predictions = new Array(predictionValues.length);
        for (i = 0; i < predictionValues.length; ++i) {
            predictions[i] = this.selection(predictionValues[i]);
        }

        return predictions;
    }

    /**
     * Export the current model to JSON.
     * @return {object} - Current model.
     */
    toJSON() {
        return {
            indexes: this.indexes,
            n: this.n,
            replacement: this.replacement,
            maxFeatures: this.maxFeatures,
            nEstimators: this.nEstimators,
            treeOptions: this.treeOptions,
            isClassifier: this.isClassifier,
            seed: this.seed,
            estimators: this.estimators
        };
    }
}
