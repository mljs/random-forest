import Random from 'random-js';
import Matrix from 'ml-matrix';

export function checkFloat(n) {
    return n > 0.0 && n < 1.0;
}

/**
 * Select n with replacement elements on the training set and values, where n is the size of the training set.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {Array} trainingValue
 * @param {number} seed - seed for the random selection, must be a 32-bit integer.
 * @return {object} with new X and y.
 */
export function examplesBaggingWithReplacement(trainingSet, trainingValue, seed) {
    var engine = Random.engines.mt19937();
    var distribution = Random.integer(0, trainingSet.rows - 1);
    if (seed === undefined) {
        engine = engine.autoSeed();
    } else if (Number.isInteger(seed)) {
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
 * @param {number} n - features.
 * @param {boolean} replacement
 * @param {number} seed - seed for the random selection, must be a 32-bit integer.
 * @return {object}
 */
export function featureBagging(trainingSet, n, replacement, seed) {
    if (trainingSet.columns < n) {
        throw new RangeError('N should be less or equal to the number of columns of X');
    }

    var distribution = Random.integer(0, trainingSet.columns - 1);
    var engine = Random.engines.mt19937();
    if (seed === undefined) {
        engine = engine.autoSeed();
    } else if (Number.isInteger(seed)) {
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
 * @return {MatrixTransposeView} toRet - the new X matrix.
 */
export function retrieveFeatures(X, indexes) {
    var toRet = new Matrix(indexes.length, X.rows);
    for (var i = 0; i < indexes.length; ++i) {
        toRet[i] = X.getColumn(indexes[i]);
    }
    return toRet.transposeView();
}
