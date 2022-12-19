# circle-mpqsolver
_circle-mpqsolver_ provides light-weight methods for finding a high-quality mixed-precision model within a reasonable time. Currently only _bisection_ method is implemented. 

# bisection
So assuming the model is parameterized by single parameter (let's call it _depth_), we can split the model so that all layers with _depth_ parameter below _specified_depth_ will be in the _input_group_, while those layers with _depth_ above _specified_depth_ will be in the _output_group_ (node's _depth_ mimics distance from _input_ to the node).
Then _bisection_ searches iteratively such a _depth_ parameter which achieves exactly the _target_loss_. In case _input_group_ is quantized into Q16 the pseudocode is the following: 
```
    until |_depth_max_ - _depth_min_| <=1 do
        _current_depth_ = 0.5 * (_depth_max_ + _depth_min_)
        if Loss(_current_depth_) < _target_loss_
            _depth_max_ = _current_depth_
        else
            _depth_min_ = _current_depth_
```
, where _Loss(current_depth)_ - measure of discrepancy between input float model and quantized at _current_depth_ model. As every iteration halves the range containing _optimal_depth_ it converges in a ~_ln(depth_max - depth_min)_ number of iterations although it may produce not the _fastest_ model.
