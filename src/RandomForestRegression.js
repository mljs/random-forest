'use strict';

var RandomForestBase = require('./RandomForestBase');

var selectionMethods = {
    mean: mean,
    median: median
};

class RandomForestRegression extends RandomForestBase {
    constructor(options, model) {
        if(options === true) {
            super(true, model.baseModel);
        } else {
            if(options === undefined) options = {};
            if(options.selectionMethod === undefined) options.selectionMethod = 'mean';

            if(!['mean', 'median'].includes(options.selectionMethod)) {
                throw new RangeError("Unsupported selection method " + options.selectionMethod);
            }

            options.classifier = false;
            super(options);
        }
    }

    selection(values) {
        return selectionMethods[this.options.selectionMethod](values);
    }

    export() {
        var baseModel = super.export();
        return {
            baseModel: baseModel,
            name: 'RFRegression'
        };
    }

    static load(model) {
        if (model.name !== 'RFRegression') {
            throw new RangeError('Invalid model: ' + model.name);
        }

        return new RandomForestRegression(true, model);
    }
}

function mean(values) {
    var sum = 0;
    for(var i = 0; i < values.length; ++i) {
        sum += values[i];
    }

    return sum / values.length;
}

function median(values) {
    values.sort( function(a,b) {return a - b;} );
    var half = Math.floor(values.length/2);
    if(values.length % 2)
        return values[half];
    else
        return (values[half-1] + values[half]) / 2.0;
}

module.exports = RandomForestRegression;