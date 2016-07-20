'use strict';

var RandomForestBase = require('./RandomForestBase');

class RandomForestClassifier extends RandomForestBase {
    constructor(options, model) {
        if(options === true) {
            super(true, model.baseModel);
        } else {
            if(options === undefined) options = {};
            options.classifier = true;
            super(options);
        }
    }

    selection(values) {
        return mode(values);
    }

    export() {
        var baseModel = super.export();
        return {
            baseModel: baseModel,
            name: 'RFClassifier'
        };
    }

    static load(model) {
        if (model.name !== 'RFClassifier') {
            throw new RangeError('Invalid model: ' + model.name);
        }

        return new RandomForestClassifier(true, model);
    }
}

function mode(arr){
    return arr.sort((a,b) =>
        arr.filter(v => v===a).length
        - arr.filter(v => v===b).length
    ).pop();
}

module.exports = RandomForestClassifier;