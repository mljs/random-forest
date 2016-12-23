'use strict';

var Utils = require('../src/utils');
var Matrix = require('ml-matrix');
var irisDataset = require('ml-dataset-iris');
var RFClassifier = require('..').RandomForestClassifier;
var RFRegression = require('..').RandomForestRegression;

describe('Basic functionality', function () {
    describe('Random Forest Classifier', function () {
        var trainingSet = irisDataset.getNumbers();
        var predictions = irisDataset.getClasses().map(elem => irisDataset.getDistinctClasses().indexOf(elem));

        var options = {
            seed: 3,
            maxFeatures: 4,
            replacement: false,
            nEstimators: 50,
            treeOptions: undefined // default options for the decision tree
        };

        var classifier = new RFClassifier(options);
        classifier.train(trainingSet, predictions);
        var result = classifier.predict(trainingSet);

        it('Random Forest Classifier with iris dataset', function () {
            var correct = 0;
            for (var i = 0; i < result.length; ++i) {
                if (result[i] === predictions[i]) correct++;
            }

            var score = correct / result.length;
            score.should.be.aboveOrEqual(0.7);
        });

        it('Export and import for random forest classifier', function () {
            var model = JSON.parse(JSON.stringify(classifier));

            var newClassifier = RFClassifier.load(model);
            var newResult = newClassifier.predict(trainingSet);

            for (var i = 0; i < result.length; ++i) {
                newResult[i].should.be.equal(result[i]);
            }
        });
    });

    describe('Random Forest Regression', function () {
        var dataset = [[73, 80, 75, 152],
                        [93, 88, 93, 185],
                        [89, 91, 90, 180],
                        [96, 98, 100, 196],
                        [73, 66, 70, 142],
                        [53, 46, 55, 101],
                        [69, 74, 77, 149],
                        [47, 56, 60, 115],
                        [87, 79, 90, 175],
                        [79, 70, 88, 164],
                        [69, 70, 73, 141],
                        [70, 65, 74, 141],
                        [93, 95, 91, 184],
                        [79, 80, 73, 152],
                        [70, 73, 78, 148],
                        [93, 89, 96, 192],
                        [78, 75, 68, 147],
                        [81, 90, 93, 183],
                        [88, 92, 86, 177],
                        [78, 83, 77, 159],
                        [82, 86, 90, 177],
                        [86, 82, 89, 175],
                        [78, 83, 85, 175],
                        [76, 83, 71, 149],
                        [96, 93, 95, 192]];

        var trainingSet = new Array(dataset);
        var predictions = new Array(dataset);

        for (var i = 0; i < dataset.length; ++i) {
            trainingSet[i] = dataset[i].slice(0, 3);
            predictions[i] = dataset[i][3];
        }

        var options = {
            seed: 3,
            maxFeatures: 2,
            replacement: false,
            nEstimators: 200,
            treeOptions: undefined // default options for the decision tree
        };

        var regression = new RFRegression(options);
        regression.train(trainingSet, predictions);
        var result = regression.predict(trainingSet);

        it('Random Forest regression with scores psychology from Houghton Mifflin', function () {
            var correct = 0;
            for (var i = 0; i < result.length; ++i) {
                if (approx(result[i], predictions[i], 10)) correct++;
            }

            var score = correct / result.length;
            score.should.be.aboveOrEqual(0.7);
        });

        it('Export and import for random forest regression', function () {
            var model = JSON.parse(JSON.stringify(regression));

            var newClassifier = RFRegression.load(model);
            var newResult = newClassifier.predict(trainingSet);

            for (var i = 0; i < result.length; ++i) {
                newResult[i].should.be.approximately(result[i], 0.01);
            }
        });
    });
});

function approx(val, expected, eps) {
    return val - eps < expected && expected < val + eps;
}

describe('Utils', function () {
    it('Retrieve features', function () {
        var data = new Matrix([[1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5]]);
        var indexes = [0, 4];

        var newData = Utils.retrieveFeatures(data, indexes);
        for (var i = 0; i < newData.column; ++i) {
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
