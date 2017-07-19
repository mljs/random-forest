
var Suite = require('benchmark').Suite;
var Matrix = require('ml-matrix').Matrix;

var testSize = 120;

var data = Matrix.rand(1000, testSize);

var suite = new Suite();

ptv1(data);

suite
    .add('tranpose', function () {
        ptv1(data);
    })
    .add('transpose view', function () {
        ptv2(data);
    }).on('cycle', function (event) {
    console.log(String(event.target));
}).run();

function ptv1(X) {
    var newData = X.transpose();
    var predictions = new Array(newData.length);
    for (var i = 0; i < newData.length; ++i) {
        predictions[i] = newData[i];
    }

    return predictions;
}

function ptv2(X) {
    var newData = X.transposeView();
    var predictions = new Array(newData.length);
    for (var i = 0; i < newData.length; ++i) {
        predictions[i] = newData.getColumn(i);
    }

    return predictions;
}