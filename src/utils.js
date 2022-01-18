import { Matrix } from 'ml-matrix';
import * as Random from 'random-js';

export function checkFloat(n) {
  return n > 0.0 && n <= 1.0;
}

export function isFloat(n) {
  return Number(n) === n && n % 1 !== 0;
}

/**
 * Select n with replacement elements on the training set and values, where n is the size of the training set.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {Array} trainingValue
 * @param {number} seed - seed for the random selection, must be a 32-bit integer.
 * @return {object} with new X and y.
 */
export function examplesBaggingWithReplacement(
  trainingSet,
  trainingValue,
  seed,
) {
  let engine;
  let distribution = Random.integer(0, trainingSet.rows - 1);
  if (seed === undefined) {
    engine = Random.MersenneTwister19937.autoSeed();
  } else if (Number.isInteger(seed)) {
    engine = Random.MersenneTwister19937.seed(seed);
  } else {
    throw new RangeError(
      `Expected seed must be undefined or integer not ${seed}`,
    );
  }

  let Xr = new Array(trainingSet.rows);
  let yr = new Array(trainingSet.rows);

  let oob = new Array(trainingSet.rows).fill(0);
  let oobN = trainingSet.rows;

  for (let i = 0; i < trainingSet.rows; ++i) {
    let index = distribution(engine);
    Xr[i] = trainingSet.getRow(index);
    yr[i] = trainingValue[index];

    if (oob[index]++ === 0) {
      oobN--;
    }
  }

  let Xoob = new Array(oobN);
  let ioob = new Array(oobN);

  // run backwards to have ioob filled in increasing order
  for (let i = trainingSet.rows - 1; i >= 0 && oobN > 0; --i) {
    if (oob[i] === 0) {
      Xoob[--oobN] = trainingSet.getRow(i);
      ioob[oobN] = i;
    }
  }

  return {
    X: new Matrix(Xr),
    y: yr,
    Xoob: new Matrix(Xoob),
    ioob,
    seed: engine.next(),
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
    throw new RangeError(
      'N should be less or equal to the number of columns of X',
    );
  }

  let distribution = Random.integer(0, trainingSet.columns - 1);
  let engine;
  if (seed === undefined) {
    engine = Random.MersenneTwister19937.autoSeed();
  } else if (Number.isInteger(seed)) {
    engine = Random.MersenneTwister19937.seed(seed);
  } else {
    throw new RangeError(
      `Expected seed must be undefined or integer not ${seed}`,
    );
  }

  let toRet = new Matrix(trainingSet.rows, n);

  let usedIndex;
  let index;
  if (replacement) {
    usedIndex = new Array(n);
    for (let i = 0; i < n; ++i) {
      index = distribution(engine);
      usedIndex[i] = index;
      toRet.setColumn(i, trainingSet.getColumn(index));
    }
  } else {
    usedIndex = new Set();
    index = distribution(engine);
    for (let i = 0; i < n; ++i) {
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
    usedIndex: usedIndex,
    seed: engine.next(),
  };
}

/**
 * collects and combines the individual results from the tree predictions on Out-Of-Bag data
 * @ignore
 * @param {{index: {Array},predicted: {Array}}[]} oob: array of individual tree predictions
 * @param {array} y: true labels
 * @param {(predictions:{Array})=>{number}} aggregate: aggregation function
 * @return {Array}
 */
export const collectOOB = (oob, y, aggregate) => {
  const res = Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const all = [];
    for (let j = 0; j < oob.length; j++) {
      const o = oob[j];
      if (o.index[0] === i) {
        all.push(o.predicted[0]);
        o.index = o.index.slice(1);
        o.predicted = o.predicted.slice(1);
      }
    }
    res[i] = { true: y[i], all: all, predicted: aggregate(all) };
  }
  return res;
};
