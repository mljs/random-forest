const RF = require('../random-forest.js');
const IrisDataset = require('ml-dataset-iris');

// All estimators are no longer identical to each other in the RF ensemble.

/*
feature_importances_ : ndarray of shape (n_features,)
    The impurity-based feature importances.
    The higher, the more important the feature.
    The importance of a feature is computed as the (normalized)
    total reduction of the criterion brought by that feature.  It is also
    known as the Gini importance.
    Warning: impurity-based feature importances can be misleading for
    high cardinality features (many unique values). See
    :func:`sklearn.inspection.permutation_importance` as an alternative.
*/

// 4 different features
const trainingSet = IrisDataset.getNumbers();
const predictions = IrisDataset.getClasses().map((elem) =>
  IrisDataset.getDistinctClasses().indexOf(elem),
);

function getAccuracy(predictions, target) {
  const nSamples = predictions.length;
  let nCorrect = 0;
  predictions.forEach((val, idx) => {
    if (val == target[idx]) {
      nCorrect++;
    }
  });
  return nCorrect / nSamples;
}

(() => {
  console.log('Random Forest Model');
  const options = {
    seed: 5,
    maxFeatures: 1.0,
    replacement: true,
    nEstimators: 25,
  };

  const classifier = new RF.RandomForestClassifier(options);
  classifier.train(trainingSet, predictions);
  let featureImportances = classifier.featureImportance();
  const result = classifier.predict(trainingSet);

  var trees = JSON.parse(JSON.stringify(classifier.estimators));

  function print_node(node, depth = 0, label = 'root') {
    if (!node) return;
    console.log(
      '\t'.repeat(depth),
      '[' + label + ']',
      'splitColumn' in node ? node.splitColumn : '-',
      node.splitValue || '-',
      Math.round(node.gain * 1000) / 1000 || '-',
      node.samples || '-',
      '->',
    );
    if (!!node.left) print_node(node.left, depth + 1, 'left');
    if (!!node.right) print_node(node.right, depth + 1, 'right');
  }

  for (var i = 0; i < trees.length; i++) {
    console.log('--------------------');
    console.log('TREE', i);
    console.log('--------------------');
    console.log('label, feature, split, gini, samples');
    print_node(trees[i].root);
    console.log('--------------------');
    console.log('--------------------');
  }

  console.log('Feature importances: ' + featureImportances);
  console.log('Accuracy: ' + getAccuracy(result, predictions));
})();
