# exo

_exo_ includes _loco_-to-_T/F Lite_ exporter (as a library).

## How to add a new TFL node

1. Add a new TFL node into `TFLNodes.lst` and `TFLNodes.h`
1. Define a knob in `Knob.lst` if you need a knob.
1. Add appropriate methods in `TFLShapeInferenceRule.cpp` and `TFLTypeInferenceRule.cpp`
1. Add a new converter under `Conversion` directory
1. Add an appropriate method in `OperationExporter.cpp`
1. Register the converter into `Convert.cpp`
