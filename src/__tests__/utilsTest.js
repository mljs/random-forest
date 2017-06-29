import * as Utils from '../utils';
import Matrix from 'ml-matrix';

describe('Utils', function () {
    test('Retrieve features', function () {
        var data = new Matrix([[1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]]);
        var indexes = [0, 4];

        var newData = Utils.retrieveFeatures(data, indexes);
        for (var i = 0; i < newData.column; ++i) {
            expect(newData[i]).toBe([1, 5]);
        }
    });

    var rows = 40;
    var cols = 25;
    var testX = Matrix.rand(rows, cols);
    var testY = new Array(rows).fill(1);

    test('Examples bagging', function () {
        var data = Utils.examplesBaggingWithReplacement(testX, testY);

        expect(data.X).toBeInstanceOf(Matrix);
    });

    test('Feature bagging with replacement', function () {
        var data = Utils.featureBagging(testX, cols - 5, true, 7);
        expect(new Set(data.usedIndex).size).toBeLessThan(data.usedIndex.length);
        expect(data.X.columns).toBe(20);
    });

    test('Feature bagging without replacement', function () {
        var data = Utils.featureBagging(testX, cols - 5, false, 7);
        expect(new Set(data.usedIndex).size).toBe(cols - 5);
        expect(data.X.columns).toBe(20);
    });
});
