# q-implant-op-level-test

`q-implant-op-level-test` validates that q-implant supports common used operators.

The test proceeds as follows

Step 0: Use circle file in 'common-artifacts' folder as the source model.
   - circle file is used as source of `q-implant`

Step 1: Generate qparam file(.json) and numpy array(.npy) through the operator python file.
```
operator file -> qparam file, numpy array
```

Step 2: Generate output.circle to use q-implant
```
"circle file" + "qparam.json" -> q-implant -> "quant circle file"
```

Step 3: Dump output.circle to output.h5.
```
"output.circle" -> circle-tensordump -> "output.h5"
```

Step 4: And compare tensor values of h5 file with numpy arrays due to validate q-implant.
