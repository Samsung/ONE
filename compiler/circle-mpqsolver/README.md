# circle-mpqsolver
_circle-mpqsolver_ provides light-weight methods for finding a high-quality mixed-precision model 
within a reasonable time.

## Methods

### Bisection
A model is split into two parts: front and back. One of them is quantized in uint8 and another in 
int16. The precision of front and back is determined by our proxy metric, upperbound of total layer 
errors. (See https://github.com/Samsung/ONE/pull/10170#discussion_r1042246598 for more details)

The boundary between the front and the back is decided by the depth of operators (depth: distance 
from input to the operator), i.e., given a depth d, layers with a depth less than d are included 
in front, and the rest are included in back. Bisection performs binary search to find a proper 
depth which achieves a qerror less than target_qerror.

In case front is quantized into Q16 the pseudocode is the following: 
```
    until |_depth_max_ - _depth_min_| <=1 do
        _current_depth_ = 0.5 * (_depth_max_ + _depth_min_)
        if Loss(_current_depth_) < _target_loss_
            _depth_max_ = _current_depth_
        else
            _depth_min_ = _current_depth_
```
, where Loss(current_depth) is the qerror of the mixied-precision model split at current_depth. 
As every iteration halves the remaining range (|depth_max - depth_min|), it converges in 
_~log2(max_depth)_ iterations.

## Usage 
Run _circle-mpqsolver_ with the following arguments.  

--data: .h5 file with test data

--input_model: Input float model initialized with min-max (recorded model)

--output_model: Output qunatized mode

--qerror_ratio: Target quantization error ratio. It should be in [0, 1]. 0 indicates qerror of full int16 model, 1 indicates qerror of full uint8 model. The lower `qerror_ratio` indicates the more accurate solution.

--bisection _mode_: input nodes should be at Q16 precision ['auto', 'true', 'false']
--visq_file: .visq.json file to be used in 'auto' mode
--save_intermediate: path to the directory where all intermediate results will be saved

```
$ ./circle-mpqsolver
  --data <.h5 data>
  --input_model <input_recorded_model>
  --output_model <output_model_pat>
  --qerror_ratio <optional value for reproducing target _qerror_ default is 0.5>
  --bisection <whether input nodes should be quantized into Q16 default is 'auto'>
  --visq_file <*.visq.json file with quantization errors>
  --save_intermediate <intermediate_results_path>
```

For example:
```
$./circle-mpqsolver
    --data dataset.h5
    --input_model model.recorded.circle
    --output_model model.q_opt.circle
    --qerror_ratio 0.4f
    --bisection true
```

It will produce _model.q_opt.circle_, which is _model.recorded.circle_ quantized to mixed precision 
using _dataset.h5_, with input nodes set to _Q16_ precision and quantization error (_qerror_) of 
_model.q_opt.circle_ will be less than
```
 _qerror(full_q16) + qerror_ratio * (qerror(full_q8) - qerror(full_q16))_
 ```
 (_full_q16_ - model quantized using Q16 precision, _full_q8_ - model quantized using Q8 precision).
