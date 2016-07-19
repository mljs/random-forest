'use strict';

var Utils = require("../src/Utils");
var Matrix = require("ml-matrix");

describe('Utils', function () {
    it('Retrieve features', function () {
        var data = new Matrix([[1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5]]);
        var indexes = [0, 4];

        var newData = Utils.retrieveFeatures(data, indexes);
        for(var i = 0; i < newData.column; ++i) {
            newData[i].should.be.deepEqual([1, 5]);
        }
    });

    var rows = 40, cols = 25;
    var testX = Matrix.rand(rows, cols);
    var testY = new Array(rows).fill(1);

    it('Examples bagging', function () {
        var data = Utils.examplesBaggingWithReplacement(testX, testY);

        data.X.should.be.instanceOf(Matrix);
    });

    it('Feature bagging with replacement', function () {
        var data = Utils.featureBagging(testX, cols - 5, true, 7);
        new Set(data.usedIndex).size.should.be.lessThan(data.usedIndex.length);
        data.X.columns.should.be.equal(20);
    });

    it('Feature bagging without replacement', function () {
        var data = Utils.featureBagging(testX, cols - 5, false, 7);
        new Set(data.usedIndex).size.should.be.equal(cols - 5);
        data.X.columns.should.be.equal(20);
    });
});
