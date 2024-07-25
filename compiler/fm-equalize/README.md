# fm-equalize

_fm-equalize_ performs "feature map equalization" on the given circle model with the given test data.

NOTE _fm-equalize_ uses _fme-detect_ and _fme-apply_.

## How to use

```
usage: fm-equalize [-h] -i INPUT -o OUTPUT [-f FME_PATTERNS] [-d DATA] [--verbose]

Command line tool to equalize feature map (FM) value distribution

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input circle model.
  -o OUTPUT, --output OUTPUT
                        Path to the output circle model.
  -f FME_PATTERNS, --fme_patterns FME_PATTERNS
                        Path to the json file that includes the detected equalization patterns.
  -d DATA, --data DATA  Path to the data used for FM equalization. Random data will be used if this option is not given.
  --verbose             Print logs
```
