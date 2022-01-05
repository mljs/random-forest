export default {
  input: 'src/index.js',
  output: {
    file: 'random-forest.js',
    format: 'cjs',
  },
  external: [
    'ml-array-mean',
    'ml-array-median',
    'ml-cart',
    'ml-matrix',
    'random-js',
  ],
};
