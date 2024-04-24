## Generate circle+ file

### Pre-requisites

- Generate Circle and TrainInfo Object API Python classes from schema

```
flatc -p ../../nnpackage/schema/circle_schema.fbs --gen-onefile --gen-object-api
flatc -p ../../runtime/libs/circle-schema/include/circle_traininfo.fbs --gen-onefile --gen-object-api
```

__The file location and name can be changed.__

### How to use

- Generate circle+ file

   ```
   python3 generate_circleplus.py input_circle output_circle [options...]
   ```

- Generate circle+ file with several options
   - 10 batch size
   - SumOverBatchSize loss reduction
   - CategoricalCrossentropy loss
   - Adam optimizer
   - learning rate 0.001.

   ```bash
   python3 generate_circleplus.py ./input_model.circle ./output_model.circle \
     --batch_size 10 --loss_reduction SumOverBatchSize CategoricalCrossentropy \
     Adam --learningRate 0.001
   ```

- Show outputfile metadata

  ```bash
  python3 circleplus.py --show-meta ./output_model.circle
  ```
