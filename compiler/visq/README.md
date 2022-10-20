# visq

_visq_ is a module to generate a json file used to visualize layer-wise quantization errors
(https://github.com/Samsung/ONE/issues/9694).

## Example
```bash
$ ./visq --fp32_circle sample.circle \
  --q_circle sample.q.circle \
  --data test.h5 \
  --mpeir_output sample.mpeir.visq.json \
  --mse_output sample.mse.visq.json
```

## Quantization error metrics

f: Result of fp32 model
q: Result of quantized model

- MPEIR: Mean Peak Error to Interval Ratio = Average(max(|f - q|) / (max(f) - min(f) + epsilon))
epsilon: 1e-6
- MSE: Mean Squared Error = Average(square(f - q))
