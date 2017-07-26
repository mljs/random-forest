import {RandomForestRegression as RFRegression} from '../..';

describe('Random Forest Regression', function () {
    function approx(val, expected, eps) {
        return val - eps < expected && expected < val + eps;
    }

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

    var trainingSet = new Array(dataset.length);
    var predictions = new Array(dataset.length);

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

    /**
     * test dataset found here:
     * link: http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html
     * Name: Test Scores for General Psychology
     */
    test('Random Forest regression with scores psychology from Houghton Mifflin', function () {
        var correct = 0;
        for (var i = 0; i < result.length; ++i) {
            if (approx(result[i], predictions[i], 10)) correct++;
        }

        var score = correct / result.length;
        expect(score).toBeGreaterThanOrEqual(0.7);
    });

    test('Export and import for random forest regression', function () {
        var model = JSON.parse(JSON.stringify(regression));

        var newClassifier = RFRegression.load(model);
        var newResult = newClassifier.predict(trainingSet);

        for (var i = 0; i < result.length; ++i) {
            expect(newResult[i]).toBeCloseTo(result[i], 0.01);
        }
    });
});
