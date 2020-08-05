# nnfw_api_gtest

A test framework for *nnfw_api* that is built with *gtest*.

This test framework consists of 3 kinds of tests:

- Validation Tests (fixture format `ValidationTest???`)
    - Basic positive/negative tests with simple nnpackages
- Generated Model Tests (fixture format `GenModelTest`)
    - One-time inference test with variety of generated models
- Regression Tests (fixture format `RegressionTest`, test format `GitHub###`)
    - When you see bugs/crashes while using those API
    - Must refer a github issue
- Misc Tests (fixture format `Test???`)
    - When you want to introduce any API use scenarios with any nnpackages

## nnpackages for testing

To test *nnfw_api*, we almost always need some nnpackages. Those are stored in a web server so there is no nnpackage files in the repo.

### How to add nnpackages for test

If there is no nnpackage that is sufficient for your need, you may need to create one. However it is not allowed to store nnpackage files in the repo.
If you want to add some, please leave an issue of asking for adding new nnpackages to the server. Then add `config.sh` for each nnpackage in `tests/scripts/nnfw_api_gtest_models`.

Once you have done the above steps, please register it in the test source code to make use of it. You may take a look at `NNPackages` class for details.

### Installation

You must install the test nnpackages before running the tests. They must be in the same directory with the test executable, under `nnfw_api_gtest_models/`. There is an installation script `tests/scripts/nnfw_api_gtest/install_nnfw_api_gtest_nnpackages.sh`, however the nnpackage file server is not public so it will fail.
