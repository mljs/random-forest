import IrisDataset from 'ml-dataset-iris';
import {RandomForestClassifier as RFClassifier} from '..';

describe('Random Forest Classifier', function () {

    var trainingSet = IrisDataset.getNumbers();
    var predictions = IrisDataset.getClasses().map(elem => IrisDataset.getDistinctClasses().indexOf(elem));

    var options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 25,
        treeOptions: undefined // default options for the decision tree
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
});
