'use strict';

var Suite = require('benchmark').Suite;
var { Matrix, MatrixColumnSelectionView, MatrixTransposeView } = require('ml-matrix');

var testSize = 120;

var data = Matrix.rand(100, testSize);
var indexes = new Array(testSize);

for (var i = 0; i < testSize; ++i) {
  indexes[i] = i;
}

var suite = new Suite();

suite
  .add('get and set column', function () {
    rfv1(data, indexes);
  })
  .add('Get column', function () {
    rfv2(data, indexes);
  })
  .add('use matrix get column', function () {
    rfv3(data, indexes);
  })
  .on('cycle', function (event) {
    console.log(String(event.target));
  }).run();

function rfv1(X, indexes) {
  var toRet = new Matrix(X.rows, indexes.length);
  for (var i = 0; i < indexes.length; ++i) {
    toRet.setColumn(i, X.getColumn(indexes[i]));
  }

  return toRet;
}

function rfv2(X, indexes) {
  var toRet = new Matrix(indexes.length, X.rows);
  for (var i = 0; i < indexes.length; ++i) {
    toRet[i] = X.getColumn(indexes[i]);
  }
  return new MatrixTransposeView(toRet);
}

function rfv3(X, indexes) {
  return new MatrixColumnSelectionView(X, indexes);
}
