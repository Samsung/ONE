# circle-interpreter-test

`circle-interpreter-test` executes `circle-interpreter` and validates its results with tflite model's output. 

The test proceeds as follows:

Step 0: Use tflite and circle file in 'common-artifacts' folder as the source model.
   - tflite file is used as to generate reference execution result
   - circle file is used as input of `circle-interpreter`

Step 1: Run TFLite interpreter and circle-interpreter for the source tflite and circle, respectively.
        (with the same input tensors filled with random values)
   - "modelfile.tflite" ---> TFLite interpreter --> Execution result 1
   - "modelfile.circle" --> circle-interpreter ---> Execution result 2

Step 2: Compare the execution result 1 and 2. Test is PASSED if results are same.
