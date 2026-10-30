# Changelog

## [0.10.0](https://github.com/ecmwf/earthkit-workflows-anemoi/compare/0.9.1...0.10.0) (2026-10-30)


### ⚠ BREAKING CHANGES

* fetch per-member ENS initial conditions; None runs are deterministic ([#57](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/57))
* treat an int ensemble_members as a single member id ([#56](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/56))
* remove dataset creation functionality ([#55](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/55))

### Features

* Accept per-stage requirements and deprecate environment ([#54](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/54)) ([4068738](https://github.com/ecmwf/earthkit-workflows-anemoi/commit/40687384207845f9c8b9492a6b22a4d306909845))
* Fetch per-member ENS initial conditions; None runs are deterministic ([#57](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/57)) ([72de6d8](https://github.com/ecmwf/earthkit-workflows-anemoi/commit/72de6d8647d9abdb2d5ec3ea246dfc12e144fb9a))
* Remove dataset creation functionality ([#55](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/55)) ([75eb35f](https://github.com/ecmwf/earthkit-workflows-anemoi/commit/75eb35f13cc8384b6c2d1c577f05c4ba0179a67a))
* Treat an int ensemble_members as a single member id ([#56](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/56)) ([bda302e](https://github.com/ecmwf/earthkit-workflows-anemoi/commit/bda302e9b53071def324908b58ff7abcdffe67ea))

## [0.9.1](https://github.com/ecmwf/earthkit-workflows-anemoi/compare/0.9.0...0.9.1) (2026-10-22)


### Bug Fixes

* Pin anemoi-inference to ~=0.13.0 and make environment test release-safe ([#53](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/53)) ([11321de](https://github.com/ecmwf/earthkit-workflows-anemoi/commit/11321dea3e3c52555f1ccd3fc7bf4e2b2e2b29f3))
* Sort step coordinates so AIFS outputs are labelled chronologically ([#49](https://github.com/ecmwf/earthkit-workflows-anemoi/issues/49)) ([e574f5d](https://github.com/ecmwf/earthkit-workflows-anemoi/commit/e574f5d8dd9e4326057c1aafddc769ea6b4b64ee))
