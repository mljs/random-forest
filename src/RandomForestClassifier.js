'use strict';

var RandomForestBase = require('./RandomForestBase');

class RandomForestClassifier extends RandomForestBase {
    constructor(options) {
        if(options === undefined) options = {};
        options.classifier = true;
        super(options);
    }

    selection(values) {
        return mode(values);
    }
}

function mode(arr){
    return arr.sort((a,b) =>
        arr.filter(v => v===a).length
        - arr.filter(v => v===b).length
    ).pop();
}

module.exports = RandomForestClassifier;