var Suite = require('benchmark').Suite;
var Matrix = require('ml-matrix').Matrix;
var Random = require('random-js');

var data = Matrix.rand(100, 120);

var suite = new Suite();

suite
    .add('Transposing', function () {
        for(var i = 0; i < data.column; ++i) {
            fbv1(data, i);
        }
    })
    .add('Get column', function () {
        for(var i = 0; i < data.column; ++i) {
            fbv2(data, i);
        }
    }).add("Extra", function () {
        for(var i = 0; i < data.column; ++i) {
            fbv3(data, i);
        }
    }).add("Transpose view", function () {
        for(var i = 0; i < data.column; ++i) {
            fbv4(data, i);
        }
    })
    .on('cycle', function (event) {
        console.log(String(event.target));
    }).run();

function fbv1(X, n) {
    var distribution = Random.integer(0, X.column);
    var engine = Random.engines.mt19937().autoSeed();

    var Xt = X.transpose();

    var toRet = new Array(n);

    for(var i = 0; i < n; ++i) {
        toRet[i] = Xt[distribution(engine)];
    }

    return new Matrix(toRet).transpose();
}

function fbv2(X, n) {
    var distribution = Random.integer(0, X.column);
    var engine = Random.engines.mt19937().autoSeed();

    var toRet = new Matrix(X.rows, n);

    for(var i = 0; i < n; ++i) {
        toRet.setColumn(i, X.getColumn(distribution(engine)));
    }

    return toRet;
}

function fbv3(X, n) {
    var distribution = Random.integer(0, X.column);
    var engine = Random.engines.mt19937().autoSeed();

    var Xt = X.transpose();

    var toRet = new Matrix(X.rows, n);

    for(var i = 0; i < n; ++i) {
        toRet.setColumn(i, Xt[distribution(engine)]);
    }

    return toRet;
}

function fbv4(X, n) {
    var distribution = Random.integer(0, X.column);
    var engine = Random.engines.mt19937().autoSeed();

    var Xt = X.transposeView();

    var toRet = new Array(n);

    for(var i = 0; i < n; ++i) {
        toRet[i] = Xt[distribution(engine)];
    }

    return new Matrix(toRet).transposeView();
}
