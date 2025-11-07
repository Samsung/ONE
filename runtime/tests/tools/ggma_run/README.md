# ggma_run

`ggma_run` is a tool to run LLM model.

It takes GGMA package as input. It uses **GGMA API** internally.

## Options

- `-p, --prompt`: Specify the input prompt text (default: "Lily picked up a flower.")

## Usage

```
$ ./ggma_run path_to_ggma_package
$ ./ggma_run path_to_ggma_package -p "Your custom prompt here"
$ ./ggma_run path_to_ggma_package --prompt "What is the weather today?"
```

It will run a GGMA package to generate the output.

## Example

```
$ Product/out/bin/ggma_run tinyllama
prompt: Lily picked up a flower.
generated: { 1100, 7899, 289, 826, 351, 600, 2439, 288, 266, 3653, 31843, 1100, 7899, 289, 1261, 291, 5869, 291, 1261, 31843, 1100, 7899 }
detokenized: She liked to play with her friends. She liked to run and jump in the water. She was
```
