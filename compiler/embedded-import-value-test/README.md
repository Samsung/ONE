# embedded-import-value-test

`embedded-import-value-test` checks models imported with and without constant copying produces same output values.

The test proceeds as follows:

1. Generate random input for provided circle model.

2. Import circle model to luci in 2 modes:
   - With constant copying (default mode).
   - Without constant copying (experimental feature)

3. Compare the execution result of both modes. The result must be the same.
