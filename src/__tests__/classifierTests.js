import IrisDataset from 'ml-dataset-iris';
import {RandomForestClassifier as RFClassifier} from '..';
import Matrix from 'ml-matrix';

describe('Random Forest Classifier', function () {

    var trainingSet = IrisDataset.getNumbers();
    var predictions = IrisDataset.getClasses().map(elem => IrisDataset.getDistinctClasses().indexOf(elem));

    var options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 25,
        treeOptions: undefined, // default options for the decision tree
        useSampleBagging: true
    };

    var classifier = new RFClassifier(options);
    classifier.train(trainingSet, predictions);
    var result = classifier.predict(trainingSet);


    test('Random Forest Classifier with iris dataset', function () {
        var correct = 0;
        for (var i = 0; i < result.length; ++i) {
            if (result[i] === predictions[i]) correct++;
        }

        var score = correct / result.length;
        expect(score).toBeGreaterThanOrEqual(0.7); // above or equal
    });

    test('Export and import for random forest classifier', function () {
        var model = JSON.parse(JSON.stringify(classifier));

        var newClassifier = RFClassifier.load(model);
        var newResult = newClassifier.predict(trainingSet);

        for (var i = 0; i < result.length; ++i) {
            expect(newResult[i]).toBe(result[i]);
        }
    });

    test('Accuracy test', function () {
        var X = new Matrix([[0, -1], [1, 0], [1, 1], [1, -1], [2, 0], [2, 1], [2, -1], [3, 2], [0, 4], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 4], [1, 10], [1, 12], [2, 10], [2, 11], [2, 14], [3, 11]]);
        var Y = Matrix.columnVector([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);

        // the test set (Xtest, Ytest)
        var Xtest = new Matrix([[0, -2], [1, 0.5], [1.5, -1], [1, 2.5], [2, 3.5], [1.5, 4], [1, 10.5], [2.5, 10.5], [2, 11.5]]);
        var Ytest = Matrix.columnVector([0, 0, 0, 1, 1, 1, 2, 2, 2]);

        // we will train our model
        var rf = new RFClassifier({nEstimators: 50});
        rf.train(X, Y);

        // we try to predict the test set
        var finalResults = rf.predict(Xtest);
        console.log(finalResults);
    });
});
