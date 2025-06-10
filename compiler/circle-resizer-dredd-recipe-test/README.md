# circle-resizer-dredd-recipe-test

It tests non-functional conditions of a circle model resized by circle-resizer.

## How to add a test?

1. Create a directory under `res/TensorFlowLiteRecipes/` or `res/CircleRecipes/`.

2. Make a recipe (`test.recipe`) for a model under the directory.

3. Make a rule (`test.rule`) you want to test under the directory. Note, that you can find more information about dredd-test-rules in _dredd-rule-lib_ module.

4. Add a test to `test.lst` in this module using `Add` macro.
   ```
   Add(RECIPE_DIR NEW_INPUTS_SIZES)
   ```
   - `NEW_INPUTS_SIZES`: New shapes of Circle model inputs in comma-separated format like `[1,2,3],[4,5]` for a model with 2 inputs.

## Example

```
# TensorFlowLiteRecipes
res/TensorFlowLiteRecipes/PRelu_000
├── test.recipe     # What you want to test
└── test.rule       # Non-functional conditions to be satisfied

# test.lst
...
Add(PRelu_000 NEW_INPUTS_SIZES [1,4,4,5],[1,1,5])
...
```
