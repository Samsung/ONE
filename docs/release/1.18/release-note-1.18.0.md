# Release Note 1.18.0

## ONE Compiler

### Compiler Frontend

- More optimization pass
  - Fold DepthwiseConv2D
  - Substitute SplitV to Split
  - Expand BroadCast Const
  - Force QuantParam
