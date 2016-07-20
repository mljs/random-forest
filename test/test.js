'use strict';

var Utils = require("../src/Utils");
var Matrix = require("ml-matrix");
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
            for(var i = 0 ; i < result.length; ++i) {
                if(result[i] == predictions[i]) correct++;
            }

            var score = correct / result.length;
            console.log(score);
            score.should.be.aboveOrEqual(0.7);
        })

    });
});

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
