# dalgona

## What is dalgona?

_dalgona_ is a tool for dynamic analysis of deep neural network.

## How it works?

_dalgona_ runs a user's custom analysis code (written in "Python") while performing inference. The analysis code has the form of hooks, called before/after each operator is executed. Intermediate execution results (values of activations) are passed to the hooks, so users can analyze the distribution of activations inside the hooks. The analysis result can be exported as files, log messages or any other forms, used for various purposes (model compression, optimization, etc.).

## Possible applications
- Finding quantization parameters based on the distribution of activations
- Finding sparse activations by observing the portion of zero values
- Finding the distribution of conditional variables in If-statement and While-statement
- Visualization of activation data with Python libraries

## Prerequisite
- Python (apt install python-dev python3-dev)
- Circle model (target to analyze)
- Input data of the model (hdf5 format. See imgdata2hdf5 or gen_h5_explicit_inputs.py for more details.)
- Analysis code (Python code)

## Example
```
dalgona \
 --input_model model.circle
 --input_data data.hdf5
 --analysis analysis/AnalysisTemplate.py
```

## Arguments
```
  --help            Show help message and exit
  --input_model     Input model filepath (.circle)
  --input_data      Input data filepath (.h5)
  --analysis        Analysis code filepath (.py)
  --analysis_args   (optional) String argument passed to the analysis code
```

## How to write analysis code?

_dalgona_ provides hooks which are called before/after an operator is executed.
Users can access tensors relevant to the corresponding operator inside the hooks.
The information of each operator is passed as the arguments of the hook.
For example, for a Conv2D operator, _dalgona_ provides following hooks.

```
  def Conv2DPre(self, name, input, filter, bias, padding, stride, dilation, fused_act)
  def Conv2DPost(self, name, input, filter, bias, padding, stride, dilation, output, fused_act)
```

`Conv2DPre`/`Conv2DPost` are called before/after Conv2D is executed, respectively. Users can write codes to analyze the distribution of intermediate tensors using the provided arguments.

(Note that Conv2DPost has one more argument "output", which is the execution result of the operator)

The tensor (e.g., input, filter, bias, output) is expressed as a Python dictionary as below.

```
{'name': string, 'data': numpy array, 'quantparam': {'scale': list, 'zero_point': list, 'quantized_dimension': int}}
```

We proivde a template for the analysis code in `analysis/AnalysisTemplate.py`. Users can copy the template file and modify it to write their custom analysis codes.

| List of hooks | Explanation |
| --------------|------------ |
| StartAnalysis(self) | Called when the analysis starts |
| EndAnalysis(self) | Called when the analysis ends |
| StartNetworkExecution(self, inputs) | Called when the execution of a network starts |
| EndNetworkExecution(self, outputs) | Called when the execution of a network ends |
| DefaultOpPre(self, name, opcode, inputs) | Default hook called before an operator is executed |
| DefaultOpPost(self, name, opcode, inputs, output) | Default hook called after an operator is executed |
| \<OPCODE\>Pre/Post | Hooks called before/after the corresponding operator is executed. |
