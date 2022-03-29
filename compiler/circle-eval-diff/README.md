# circle-eval-diff

_circle-eval-diff_ compares inference results of two circle models.

## Usage

Run circle-eval-diff with the following arguments.

--first_input_model: first model to compare (.circle).
--second_input_model: second model to compare (.circle).
--first_input_data: input data for the first model (.h5, directory). Random data will be used if this argument is not given.
--second_input_data: input data for the second model (.h5, directory). Random data will be used if this argument is not given.
--input_data_format: input data format (h5 (default), directory).
--metric: metric to compare inference results (MAE (default), etc).

```
$ ./circle-eval-diff
  --first_input_model <first_input_model>
  --second_input_model <second_input_model>
  --first_input_data <first_input_data>
  --second_input_data <second_input_data>
  --input_data_format <data_format>
  --metric <metric>
```

For example,
```
$ ./circle-eval-diff
  --first_input_model A.circle
  --second_input_model B.circle
  --first_input_data A.h5
  --second_input_data B.h5
  --input_data_format h5
  --metric MAE
```

It will print MAE (Mean Absolute Error) between the inference result of A.circle with A.h5 and that of B.circle with B.h5.

## Note

Circle models are executed by _luci-interpreter_.
