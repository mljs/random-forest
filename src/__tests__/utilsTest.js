import { Matrix, MatrixColumnSelectionView } from 'ml-matrix';

import * as Utils from '../utils';

var rows = 40;
var cols = 25;
var testX = Matrix.rand(rows, cols);
var testY = new Array(rows).fill(1);

describe('Utils', () => {
  it('Retrieve features', () => {
    var data = new Matrix([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]);
    var indexes = [0, 4];

    var newData = new MatrixColumnSelectionView(new Matrix(data), indexes);
    for (var i = 0; i < newData.column; ++i) {
      expect(newData[i]).toBe([1, 5]);
    }
  });

  it('Examples bagging', () => {
    var data = Utils.examplesBaggingWithReplacement(testX, testY);

    expect(data.X).toBeInstanceOf(Matrix);
  });

  it('Feature bagging with replacement', () => {
    var data = Utils.featureBagging(testX, cols - 5, true, 7);
    expect(new Set(data.usedIndex).size).toBeLessThan(data.usedIndex.length);
    expect(data.X.columns).toBe(20);
  });

  it('Feature bagging without replacement', () => {
    var data = Utils.featureBagging(testX, cols - 5, false, 7);
    expect(new Set(data.usedIndex).size).toBe(cols - 5);
    expect(data.X.columns).toBe(20);
  });
});
