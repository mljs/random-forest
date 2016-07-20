'use strict';

var Random = require('random-js');
var Matrix = require('ml-matrix');

function isInt(n) {
    return Number(n) === n && n % 1 === 0;
}

function examplesBaggingWithReplacement(X, y, seed) {
    var engine = Random.engines.mt19937();
    var distribution = Random.integer(0, X.rows - 1);
    if (seed === undefined) {
        engine = engine.autoSeed();
    } else if (isInt(seed)) {
        engine = engine.seed(seed);
    } else {
        throw new RangeError('Expected seed must be undefined or integer not ' + seed);
    }

    var Xr = new Array(X.rows);
    var yr = new Array(X.rows);

    for (var i = 0; i < X.rows; ++i) {
        var index = distribution(engine);
        Xr[i] = X[index];
        yr[i] = y[index];
    }

    return {
        X: new Matrix(Xr),
        y: yr
    };
}

function featureBagging(X, n, replacement, seed) {
    if (X.columns < n) {
        throw new RangeError('N should be lesser or equal to the number of columns of X');
    }

    var distribution = Random.integer(0, X.columns - 1);
    var engine = Random.engines.mt19937();
    if (seed === undefined) {
        engine = engine.autoSeed();
    } else if (isInt(seed)) {
        engine = engine.seed(seed);
    } else {
        throw new RangeError('Expected seed must be undefined or integer not ' + seed);
    }

    var toRet = new Matrix(X.rows, n);

    if (replacement) {
        var usedIndex = new Array(n);
        for (var i = 0; i < n; ++i) {
            var index = distribution(engine);
            usedIndex[i] = index;
            toRet.setColumn(i, X.getColumn(index));
        }
    } else {
        usedIndex = new Set();
        index = distribution(engine);
        for (i = 0; i < n; ++i) {
            while (usedIndex.has(index)) {
                index = distribution(engine);
            }
            toRet.setColumn(i, X.getColumn(index));
            usedIndex.add(index);
        }
        usedIndex = Array.from(usedIndex);
    }

    return {
        X: toRet,
        usedIndex: usedIndex
    };
}

function retrieveFeatures(X, indexes) {
    var toRet = new Matrix(X.rows, indexes.length);
    for (var i = 0; i < indexes.length; ++i) {
        toRet.setColumn(i, X.getColumn(indexes[i]));
    }

    return toRet;
}

function isString(s) {
    return typeof s === 'string' || s instanceof String;
}

function checkFloat(n) {
    return n > 0.0 && n < 1.0;
}

module.exports = {
    examplesBaggingWithReplacement: examplesBaggingWithReplacement,
    featureBagging: featureBagging,
    retrieveFeatures: retrieveFeatures,
    isInt: isInt,
    isString: isString,
    checkFloat: checkFloat
};
