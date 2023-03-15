# visq

_visq_ is a module to generate a json file used to visualize layer-wise quantization errors
(https://github.com/Samsung/ONE/issues/9694).

## Example
```bash
$ ./visq --fp32_circle sample.circle \
  --q_circle sample.q.circle \
  --data test.h5 \
  --mpeir_output sample.mpeir.visq.json \
  --mse_output sample.mse.visq.json \
  --tae_output sample.tae.visq.json \
  --dump_dot_graph
```

The above command will generate
- `sample.mpeir.visq.json`: Json file that contains layer-wise mpeir.
- `sample.mse.visq.json`: Json file that conatins layer-wise mse.
- `sample.mpeir.visq.json.dot`: Dot graph for layer-wise mpeir.
- `sample.tae.visq.json.dot`: Dot graph for layer-wise tae.
- `sample.mse.visq.json.dot`: Dot graph for layer-wise mse.

## Quantization error metrics

f: Result of fp32 model
q: Result of quantized model

- MPEIR: Mean Peak Error to Interval Ratio = Average(max(|f - q|) / (max(f) - min(f) + epsilon))
epsilon: 1e-6
- MSE: Mean Squared Error = Average(square(f - q))
- TAE: Total Absolute Error = Sum(|f - q|)
