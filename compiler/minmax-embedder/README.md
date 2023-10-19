# minmax-embedder

_minmax-embedder_ embeds minmax to circle.

### Usage
```
Usage: ./minmax_embedder [-h] [--version] [--min_percentile MIN_PERCENTILE] [--max_percentile MAX_PERCENTILE] [-o O] circle minmax

[Positional argument]
circle                  Path to input circle model
minmax                  Path to minmax data in hdf5

[Optional argument]
-h, --help              Show help message and exit
--version               Show version information and exit
--min_percentile        Set min percentile (default: 1)
--max_percentile        Set max percentile (default: 99)
-o                      Path to output circle model
```
