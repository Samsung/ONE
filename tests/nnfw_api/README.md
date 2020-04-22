# oneapi_test

A test framework for *oneapi* that is built with *gtest*.

This test framework consists of 3 kinds of tests:

- Validation Tests (fixture format `ValidationTest???`)
    - Basic positive/negative tests with simple nnpackages
- Regression Tests (fixture format `RegressionTest`, test format `GitHub###`)
    - When you see bugs/crashes while using those API
    - Must refer a github issue
- Misc Tests (fixture format `Test???`)
    - When you want to introduce any API use scenarios with any nnpackages

## nnpackages for testing

To test *oneapi*, we almost always need some nnpackages. Those are stored in a web server so there is no nnpackage files in the repo.

### How to add nnpackages for test

If there is no nnpackage that is sufficient for your need, you may need to create one. However it is not allowed to store nnpackage files in the repo.
If you want to add some, please leave an issue of asking for adding new nnpackages to the server.

Once your nnpackage has been added to the server, please register it in the test source code to make use of it. Please take a look at `NNPackages` class for details.
