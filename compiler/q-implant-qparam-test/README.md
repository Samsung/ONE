# q-implant-qparam-test

`q-implant-qparam-test` validates that q-implant supports common used operators.

The test proceeds as follows

Step 1: Generate qparam file(.json) and numpy array(.npy) through the operator python file.
```
operator file -> qparam file, numpy array
```

Step 2: Generate output.circle to use q-implant
```
"circle file" + "qparam.json" -> q-implant -> "quant circle file"
```

Step 3: Dump output.circle to output.h5.
```
"output.circle" -> circle-tensordump -> "output.h5"
```

Step 4: And compare tensor values of h5 file with numpy arrays due to validate q-implant.

how to make qparam file

step 1: Choose the recipe in 'res/TensorFlowLiteRecipes' and get name of recipe.

step 2: Create folder in qparam that name is recipe name

step 3: Create `__init__.py` follow this sample.

``` python
from test_utils import TestCase
from test_utils import gen_random_tensor


class recipe_name_000_Q8(TestCase):
    def __init__(self):
        self.name = _name_

    def generate(self) -> dict:
        json_content = dict()

        # Generate operand_name
        json_content['operand_name'] = gen_random_tensor(
            "uint8",  # dtype_str
            (1),  # scale_shape
            (1),  # zerop_shape
            0,  # quantized_dimension
            (3, 3, 3, 3)) # value_shape ( such as weight, bias )

         ...

        return json_content


_name_ = 'recipe_name_000_Q8'

_test_case_ = recipe_name_000_Q8()

```
