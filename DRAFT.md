# DRAFT

```
~/github/YongseopKim/ONE (draft/tests/nnfw_api/GenModelTest/trainable ✗) make -f Makefile.template && ./Product/out/unittest/nnfw_api_gtest --gtest_filter=GenModelTrain.OneOp_Conv2D

Note: Google Test filter = GenModelTrain.OneOp_Conv2D
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from GenModelTrain
[ RUN      ] GenModelTrain.OneOp_Conv2D
/home/dragon/github/YongseopKim/ONE/tests/nnfw_api/lib/GenModelTrain.h:133: Failure
Expected equality of these values:
  _context->backends().size()
    Which is: 0
  1
[  FAILED  ] GenModelTrain.OneOp_Conv2D (0 ms)
[----------] 1 test from GenModelTrain (0 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (0 ms total)
[  PASSED  ] 0 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] GenModelTrain.OneOp_Conv2D

 1 FAILED TEST
```
