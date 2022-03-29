# circle-eval-diff

_circle-eval-diff_ is a tool to compare inference results of two circle models.

## Usage

This will run with the path to the input model (.circle), a pack of input data (.h5), and the output model (.circle).

```
$ ./circle-eval-diff <first_input_model> <second_input_model> <path_to_input_data> --metric <metric>
```

For example,
```
$ ./circle-eval-diff A.circle B.circle input.h5 --metric MAE
```

It will print MAE (Mean Absolute Error) between the inference result of A.circle and that of B.circle.
