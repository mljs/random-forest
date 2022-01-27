# Changelog

## [2.1.0](https://www.github.com/mljs/random-forest/compare/v2.0.3...v2.1.0) (2022-01-27)


### Features

* export RandomForestBaseOptions type ([#41](https://www.github.com/mljs/random-forest/issues/41)) ([c7c1f55](https://www.github.com/mljs/random-forest/commit/c7c1f55bced9bb6a62941fdff1dd0a605cc2f4dd))

### [2.0.3](https://www.github.com/mljs/random-forest/compare/v2.0.2...v2.0.3) (2022-01-18)


### Bug Fixes

* issues 9, 15, 17, 28, 32\ncloeses [#9](https://www.github.com/mljs/random-forest/issues/9) [#15](https://www.github.com/mljs/random-forest/issues/15) [#17](https://www.github.com/mljs/random-forest/issues/17) [#28](https://www.github.com/mljs/random-forest/issues/28) [#32](https://www.github.com/mljs/random-forest/issues/32) ([7fe6925](https://www.github.com/mljs/random-forest/commit/7fe69253deefc77f9f6030bcf376a0ca70a56206))

### [2.0.2](https://github.com/mljs/random-forest/compare/v2.0.1...v2.0.2) (2020-10-18)


### Bug Fixes

* small fixes to previous PRs ([#24](https://github.com/mljs/random-forest/issues/24)) ([a5fc52e](https://github.com/mljs/random-forest/commit/a5fc52e3f62289d12a4394926d5fee70ed450938))

## [2.0.1](https://github.com/mljs/random-forest/compare/v2.0.0...v2.0.1) (2019-10-13)


### Bug Fixes

* bump ml-cart version ([#14](https://github.com/mljs/random-forest/issues/14)) ([1883192](https://github.com/mljs/random-forest/commit/1883192965d0be4da11ae911f257a6ccb1c7a764))



# [2.0.0](https://github.com/mljs/random-forest/compare/v1.0.3...v2.0.0) (2019-06-29)


### chore

* update dependencies and remove support for Node.js 6 ([99fcd3c](https://github.com/mljs/random-forest/commit/99fcd3c))


### BREAKING CHANGES

* Node.js 6 is no longer supported



<a name="1.0.3"></a>
## [1.0.3](https://github.com/mljs/random-forest/compare/v1.0.2...v1.0.3) (2018-05-03)


### Bug Fixes

* call toJSON to include name property in estimators section ([#12](https://github.com/mljs/random-forest/issues/12)) ([1f2f833](https://github.com/mljs/random-forest/commit/1f2f833)), closes [#11](https://github.com/mljs/random-forest/issues/11)



<a name="1.0.2"></a>
## [1.0.2](https://github.com/mljs/random forest/compare/v1.0.1...v1.0.2) (2017-08-17)


### Bug Fixes

* bug related with feature bagging, now the default percentage is 100 ([4cae0dd](https://github.com/mljs/random forest/commit/4cae0dd))



<a name="1.0.1"></a>
## [1.0.1](https://github.com/mljs/random forest/compare/v1.0.0...v1.0.1) (2017-08-02)



<a name="1.0.0"></a>
# 1.0.0 (2017-08-01)


### Performance Improvements

* use columnSelectionView for retrieving features (40 times faster) ([6a3d5f4](https://github.com/mljs/random forest/commit/6a3d5f4))
* use MatrixTransposeView for Utils.retrieveFeatures (2 times faster) ([d0aaeab](https://github.com/mljs/random forest/commit/d0aaeab))
* using transpose view for predictions (500 times faster) ([5401c3a](https://github.com/mljs/random forest/commit/5401c3a))
