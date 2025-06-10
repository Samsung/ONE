# circle-input-names

`circle-input-names` is a tool to generate input names of the Circle model's operators in JSON format.

* Example result of `circle-input-names`
  ```
  {
    "ABS" : [ "x" ],
    "ADD" : [ "x", "y" ],
    "ADD_N" : [ "inputs" ],
    "ARG_MAX" : [ "input", "dimension" ],
    "ARG_MIN" : [ "input", "dimension" ],
    ...
    "INSTANCE_NORM" : [ "input", "gamma", "beta" ],
    "RMS_NORM" : [ "input", "gamma" ],
    "ROPE" : [ "input", "sin_table", "cos_table" ]
  }
  ```
