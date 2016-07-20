'use strict';

var Random = require('random-js');
var Matrix = require('ml-matrix');


function isInt(n) {
    return Number(n) === n && n % 1 === 0;
}

function isString(s) {
    return typeof s === 'string' || s instanceof String;
}

function checkFloat(n) {
    return n > 0.0 && n < 1.0;
}

/**
 * Select n with replacement elements on the training set and values, where n is the size of the training set.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {Array} trainingValue
 * @param {Number} seed - seed for the random selection, must be a 32-bit integer.
 * @return {{X: Matrix, y: Array}}
 */
function examplesBaggingWithReplacement(trainingSet, trainingValue, seed) {
    var engine = Random.engines.mt19937();
    var distribution = Random.integer(0, trainingSet.rows - 1);
    if (seed === undefined) {
        engine = engine.autoSeed();
    } else if (isInt(seed)) {
        engine = engine.seed(seed);
    } else {
        throw new RangeError('Expected seed must be undefined or integer not ' + seed);
    }

    var Xr = new Array(trainingSet.rows);
    var yr = new Array(trainingSet.rows);

    for (var i = 0; i < trainingSet.rows; ++i) {
        var index = distribution(engine);
        Xr[i] = trainingSet[index];
        yr[i] = trainingValue[index];
    }

    return {
        X: new Matrix(Xr),
        y: yr
    };
}

/**
 * selects n features from the training set with or without replacement, returns the new training set and the indexes used.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {Number} n - features.
 * @param {Boolean} replacement
 * @param {Number} seed - seed for the random selection, must be a 32-bit integer.
 * @return {{X: *, usedIndex: Array}}
 */
function featureBagging(trainingSet, n, replacement, seed) {
    if (trainingSet.columns < n) {
        throw new RangeError('N should be lesser or equal to the number of columns of X');
    }

    var distribution = Random.integer(0, trainingSet.columns - 1);
    var engine = Random.engines.mt19937();
    if (seed === undefined) {
        engine = engine.autoSeed();
    } else if (isInt(seed)) {
        engine = engine.seed(seed);
    } else {
        throw new RangeError('Expected seed must be undefined or integer not ' + seed);
    }

    var toRet = new Matrix(trainingSet.rows, n);

    if (replacement) {
        var usedIndex = new Array(n);
        for (var i = 0; i < n; ++i) {
            var index = distribution(engine);
            usedIndex[i] = index;
            toRet.setColumn(i, trainingSet.getColumn(index));
        }
    } else {
        usedIndex = new Set();
        index = distribution(engine);
        for (i = 0; i < n; ++i) {
            while (usedIndex.has(index)) {
                index = distribution(engine);
            }
            toRet.setColumn(i, trainingSet.getColumn(index));
            usedIndex.add(index);
        }
        usedIndex = Array.from(usedIndex);
    }

    return {
        X: toRet,
        usedIndex: usedIndex
    };
}

/**
 * retrieve a new X matrix containing the column elements at the given indexes.
 * @ignore
 * @param {Matrix} X
 * @param {Array} indexes
 * @return {Matrix} toRet - the new X matrix.
 */
function retrieveFeatures(X, indexes) {
    var toRet = new Matrix(X.rows, indexes.length);
    for (var i = 0; i < indexes.length; ++i) {
        toRet.setColumn(i, X.getColumn(indexes[i]));
    }

    return toRet;
}

module.exports = {
    examplesBaggingWithReplacement: examplesBaggingWithReplacement,
    featureBagging: featureBagging,
    retrieveFeatures: retrieveFeatures,
    isInt: isInt,
    isString: isString,
    checkFloat: checkFloat
};
