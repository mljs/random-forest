import { RandomForestRegression as RFRegression } from '..';

function approx(val, expected, eps) {
  return val - eps < expected && expected < val + eps;
}

let dataset = [
  [73, 80, 75, 152],
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
  [96, 93, 95, 192],
];

let trainingSet = new Array(dataset.length);
let predictions = new Array(dataset.length);

for (let i = 0; i < dataset.length; ++i) {
  trainingSet[i] = dataset[i].slice(0, 3);
  predictions[i] = dataset[i][3];
}

let options = {
  seed: 3,
  maxFeatures: 2,
  replacement: false,
  nEstimators: 200,
  treeOptions: undefined,
  useSampleBagging: true,
};

let regression = new RFRegression(options);
regression.train(trainingSet, predictions);
let result = regression.predict(trainingSet);

/**
 * link: http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html
 * Name: Test Scores for General Psychology
 */
describe('Random Forest Regression', () => {
  it('Random Forest regression with scores psychology from Houghton Mifflin', () => {
    const correct = result.reduce((prev, value, index) => {
      return approx(value, predictions[index], 10) ? prev + 1 : prev;
    }, 0);

    let score = correct / result.length;

    expect(score).toBeGreaterThanOrEqual(0.7);
  });

  it('Export and import for random forest regression', () => {
    let model = JSON.parse(JSON.stringify(regression));

    let newClassifier = RFRegression.load(model);
    let newResult = newClassifier.predict(trainingSet);

    for (let i = 0; i < result.length; ++i) {
      expect(newResult[i]).toBeCloseTo(result[i], 0.01);
    }
  });
});
