# circle2circle-dredd-recipe-test

It tests the non-functional conditions of the optimized circle binary resulting from circle2circle.

This test basically refers to the _TensorFlowLiteRecipes_ resource.
So you should add what you want to test to both of the resource and `test.lst`.

## Example

```
# TensorFlowLiteRecipes
res/TensorFlowLiteRecipes/BatchMatMulV2_000
├── test.recipe     # What you want to test
└── test.rule       # Non-functional conditions to be satisfied

# test.lst
...
Add(BatchMatMulV2_000 PASS resolve_customop_batchmatmul)
...
```

For more information on the rules, see _dredd-rule-lib_ module.
