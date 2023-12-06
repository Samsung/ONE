# luci-pass-value-py-test

`luci-pass-value-py-test` validates execution result values of tflite model and
circle model generated with specific optimization.

The test proceeds as follows:

Step 0: Use tflite and circle file in 'common-artifacts' folder as the source model.
   - tflite file is used as to generate reference execution result
   - circle file is used as source of optimization to apply

Step 1: Run circle2circle with given optimization option to produce transformed circle.
   - "modelfile.circle" -> circle2circle -> "modelfile.pass.circle"

Step 2: Run TFLite interpreter and luci-interpreter for the source tflite and circle, respectively.
        (with the same input tensors filled with random values)
   - "modelfile.tflite" ------> TFLite interpreter -> Execution result 1
   - "modelfile.pass.circle" -> luci-interpreter ---> Execution result 2

Step 3: Compare the execution result 1 and 2. Test is PASSED if results are sames.
