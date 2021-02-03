# validate-onnx2circle

_validate-onnx2circle_ provides validation of onnx to optimized circle conversion
by comparing execution results of original onnx model and optimized circle model.

This is currently in experimental state.

## How to run the script

Install `onnx-runtime` inside virtual environment
```
source install_path/bin/venv/bin/activate

python -m pip --default-timeout=1000 --trusted-host pypi.org \
  --trusted-host files.pythonhost.org install onnxruntime==1.6.0

deactivate
```

Run the sctipt
```bash
cd install_path/test

driver='one/build/debug/compiler/luci-eval-driver/luci_eval_driver'
onnx_filepath='path_to_onnx_model.onnx'
circle_filepath='path_to_optimized_circle.circle'

./validate_onnx2circle.py --driver ${driver} --onnx ${onnx_filepath} --circle ${circle_filepath}
```

Output will show something like this
```
Run ONNX...
Run luci-interpreter...
Compare 0 True
```
