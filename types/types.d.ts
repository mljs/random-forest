import Matrix from "ml-matrix";

declare module 'ml-random-forest' {

    interface RandomForestBaseOptions {
        /**the number of features used on each estimator.
         *        * if is an integer it selects maxFeatures elements over the sample features.
         *        * if is a float between (0, 1), it takes the percentage of features. */
        maxFeatures: number,
        /** use replacement over the sample features. */
        replacement: boolean,
        /** number of estimator to use. */
        nEstimators: number,
        /** seed for feature and samples selection, must be a 32-bit integer. */
        seed: number,
        /** use bagging over training samples. */
        useSampleBagging: boolean,
        /** options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/} */
        treeOptions: object,
        /** Don't calculate Out-Of-Bag predictions. Improves performance if set to true. */
        noOOB: boolean,
        isClassifier: boolean,
        /** the way to calculate the prediction from estimators, "mean" and "median" are supported. */
        selectionMethod?: 'mean' | 'median',
    };

    type Estimator = DecisionTreeClassifier | DecisionTreeRegression;

    class RandomForestBase extends RandomForestBaseOptions {
        n?: number;
        indexes?: number[][];
        estimators?: Estimator[];
        /** Out-Of-Bag results */
        oobResults?: {
            /** the true label or value of this row */
            true: number,
            /** all Out-Of-Bag predictions */
            all: number[],
            /** the final prediction for this row */
            predicted: number
        }[];

        /**
         * Predicts the output given the matrix to predict.
         * @param {Matrix|Array} toPredict
         * @return {Array} predictions
         */
        predict(toPredict: (number[][]) | Matrix): number[];
        /**
         * Train the decision tree with the given training set and labels.
         * @param {Matrix|Array} trainingSet
         * @param {Array} trainingValues
         */
        train(trainingSet: number[][], trainingValues: number[]): void;


        /**
         * Returns the Out-Of-Bag predictions.
         * @return {Array} predictions
         */
        predictOOB(): number[];

        /**
         * Predicts the output for every tree given the matrix to predict.
         * @param {Matrix|Array} toPredict
         * @return {Matrix} predictions
         */
        predictionValues(toPredict: (number[][]) | Matrix): Matrix
    }


    interface baseModel extends RandomForestBaseOptions {
        indexes: number[][],
        n: number,
        estimators: Estimator[]
    }

    interface RandomForestClassifierModel {
        baseModel: baseModel,
        name: 'RFClassifier',

    }
    interface RandomForestRegressionModel {
        baseModel: baseModel,
        name: 'RFRegression',
        selectionMethos: 'mean' | 'median'

    }
    class RandomForestClassifier extends RandomForestBase {
        /**
         * Create a new base random forest for a classifier or regression model.
         * @constructor
         * @param {object} options
         * @param {number} [options.maxFeatures=1.0] - the number of features used on each estimator.
         *        * if is an integer it selects maxFeatures elements over the sample features.
         *        * if is a float between (0, 1), it takes the percentage of features.
         * @param {boolean} [options.replacement=true] - use replacement over the sample features.
         * @param {number} [options.seed=42] - seed for feature and samples selection, must be a 32-bit integer.
         * @param {number} [options.nEstimators=10] - number of estimator to use.
         * @param {object} [options.treeOptions={}] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
         * @param {boolean} [options.useSampleBagging=false] - use bagging over training samples.
         * @param {boolean} [options.noOOB=false] - don't calculate Out-Of-Bag results.
         * @param {object} model - for load purposes.
         */
        constructor(options: Partial<RandomForestBaseOptions> | true, model?: RandomForestClassifierModel);
        /**
         * retrieve the prediction given the selection method.
         * @param {Array} values - predictions of the estimators.
         * @return {number} prediction
         */
        selection(values: any[]): number;
        /**
         * Export the current model to JSON.
         * @return {object} - Current model.
         */
        toJSON(): RandomForestClassifierModel;
        /**
         * Load a Decision tree classifier with the given model.
         * @param {object} model
         * @return {RandomForestClassifier}
         */
        static load(model: RandomForestClassifierModel): RandomForestClassifier;
        /**
         * Predicts the probability of a label given the matrix to predict.
         * @param {Matrix|Array} toPredict
         * @param {number} label
         * @return {Array} predictions
         */
        predictProbability(toPredict: number[][], label: number): number[];
        /**
           * Returns the confusion matrix
           * Make sure to run train first.
           * @return {number[][]} - matrix.
           */
        getConfusionMatrix(): number[][];


    }
    class RandomForestRegression extends RandomForestBase {
        /**
         * Create a new base random forest for a classifier or regression model.
         * @constructor
         * @param {object} options
         * @param {number} [options.maxFeatures=1.0] - the number of features used on each estimator.
         *        * if is an integer it selects maxFeatures elements over the sample features.
         *        * if is a float between (0, 1), it takes the percentage of features.
         * @param {boolean} [options.replacement=true] - use replacement over the sample features.
         * @param {number} [options.seed=42] - seed for feature and samples selection, must be a 32-bit integer.
         * @param {number} [options.nEstimators=10] - number of estimator to use.
         * @param {object} [options.treeOptions={}] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
         * @param {string} [options.selectionMethod="mean"] - the way to calculate the prediction from estimators, "mean" and "median" are supported.
         * @param {boolean} [options.useSampleBagging=false] - use bagging over training samples.
         * @param {boolean} [options.noOOB=false] - don't calculate Out-Of-Bag results.
         * @param {object} model - for load purposes.
         */
        constructor(options: Partial<RandomForestBaseOptions> | true, model?: RandomForestRegressionModel)
        /**
         * Load a Decision tree classifier with the given model.
         * @param {object} model
         * @return {RandomForestRegression}
         */
        static load(model: RandomForestRegressionModel): RandomForestRegression;
        /**
         * Export the current model to JSON.
         * @return {object} - Current model.
         */
        toJSON(): RandomForestRegressionModel;
        /**
         * retrieve the prediction given the selection method.
         * @param {Array} values - predictions of the estimators.
         * @return {number} prediction
         */
        selection(values: number[]): number;
    }
    export { RandomForestClassifier, RandomForestRegression, RandomForestBaseOptions }
}

