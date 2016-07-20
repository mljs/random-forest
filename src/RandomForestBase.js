'use strict';

var DTClassfier = require('ml-cart').DecisionTreeClassifier;
var DTRegression = require('ml-cart').DecisionTreeRegression;
var Utils = require('./Utils');
var Matrix = require('ml-matrix');

var functions = {
    'sqrt': Math.sqrt,
    'log2': Math.log2
};

class RandomForestBase {
    constructor(options, model) {
        if (options === true) {
            this.options = model.options;
            this.n = model.n;
            this.indexes = model.indexes;

            var Estimator = this.options.classifier ? DTClassfier : DTRegression;
            this.estimators = model.estimators.map(est => Estimator.load(est));
        } else {
            if (options === undefined) options = {};
            if (options.classifier === undefined) options.classifier = true;
            if (options.nEstimators === undefined) options.nEstimators = 10;
            if (options.replacement === undefined) options.replacement = true;
            if (options.maxFeatures === undefined) options.maxFeatures = 'sqrt';

            if (Utils.isString(options.maxFeatures) && functions[options.maxFeatures] === undefined) {
                throw new RangeError('Not supported operation: ' + options.maxFeatures);
            }

            this.options = options;
        }
    }

    train(trainingSet, trainingValues) {
        if (!Matrix.isMatrix(trainingSet)) {
            trainingSet = new Matrix(trainingSet);
        }

        if (Utils.isString(this.options.maxFeatures)) {
            this.n = Math.floor(functions[this.options.maxFeatures](trainingSet.columns));
        } else if (Utils.isInt(this.options.maxFeatures)) {
            if (this.options.maxFeatures > trainingSet.columns) {
                throw new RangeError('The maxFeatures parameter should be lesser than ' + trainingSet.columns);
            } else {
                this.n = this.options.maxFeatures;
            }
        } else if (Utils.checkFloat(this.options.maxFeatures)) {
            this.n = Math.floor(trainingSet.columns * this.options.maxFeatures);
        } else {
            throw new RangeError('Cannot process the maxFeatures parameter ' + this.options.maxFeatures);
        }


        if (this.options.classifier) {
            var Estimator = DTClassfier;
        } else {
            Estimator = DTRegression;
        }

        this.estimators = new Array(this.options.nEstimators);
        this.indexes = new Array(this.options.nEstimators);

        for (var i = 0; i < this.options.nEstimators; ++i) {
            var res = Utils.examplesBaggingWithReplacement(trainingSet, trainingValues, this.options.seed);
            var X = res.X;
            var y = res.y;

            res = Utils.featureBagging(X, this.n, this.options.replacement, this.options.seed);
            X = res.X;

            this.indexes[i] = res.usedIndex;
            this.estimators[i] = new Estimator(this.options.treeOptions);
            this.estimators[i].train(X, y);
        }
    }

    selection() {
        throw new Error('Abstract method \'selection\' not implemented!');
    }

    predict(toPredict) {
        var predictionValues = new Array(this.options.nEstimators);
        for (var i = 0; i < this.options.nEstimators; ++i) {
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

    export() {
        return {
            indexes: this.indexes,
            n: this.n,
            options: this.options,
            estimators: this.estimators.map(est => est.export())
        };
    }

    static load(model) {
        return new RandomForestBase(true, model);
    }
}

module.exports = RandomForestBase;
