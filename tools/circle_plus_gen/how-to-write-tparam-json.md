# How to write Hyperparameters JSON file

The Hyperparameters JSON file is used to store hyperparameters that should be configured before training. For quicker understanding, please refer to the examples([#1](./example/train_tparam.json), [#2](./example/tparam_sgd_scce.json)) of the JSON file. 
<br/>

The json file consists of **a single JSON object** containing the following keys and corresponding values : 

```json
{
  "optimizer" : {...}, 
  "loss" : {...},
  "batchSize" : 32, 
}
```

- "optimizer" : refer [optimizer](#optimizer) for writing corresponding value
- "loss" : refer [loss](#loss) for writing corresponding value
- "batchSize" : a number of examples processeed during each iteration 


## optimizer

An object describing optimization algorithm. This should include two keys : 

* `type` : a string to indicate optimizer 
  * You can find the `type` strings for each optimizers from [OptimizerNamer.names](https://github.com/Samsung/ONE/blob/master/tools/circle_plus_gen/lib/utils.py#L17).
  * For example, If you like to use 'adam' optimizer, you should write 'adam' for type.
* `args` : an object holding additional arguments to the chosen optimizer. These may vary depending on the optimizer type, but typically include 'learningRate'. 
  * You can obtain args lists from `table [optimizer]Options`. Please refer to [here](https://github.com/Samsung/ONE/blob/master/runtime/libs/circle-schema/include/circle_traininfo.fbs). 
  * For example, to use 'adam' optimizer, you should fill the args with 
  [`table AdamOptions`](https://github.com/Samsung/ONE/blob/1a1a52afd87154720c28420d9a6804191421d5de/runtime/libs/circle-schema/include/circle_traininfo.fbs#L63-L66) attributes. 

  ```json
  {
    ... 
    "optimizer": {
      "type": "adam",
      "args": {
          "learning_rate": 0.01,
          "beta_1": 0.9,
          "beta_2": 0.999,
          "epsilon": 1e-07
      }
    }, 
    ... 
  } 
  ``` 

**Supported Optimizers:** <br/>
  * adam 
  * sgd 


## loss

An object describing the loss function. This should include two keys :

* `type`: a string specifying which loss function to use.
  * You can find the `type` strings at [LossNamer.names](https://github.com/Samsung/ONE/blob/master/tools/circle_plus_gen/lib/utils.py#L34). 
  * For example, If you choose 'mse', you can use either 'mean squared error' or 'mse' for the type. 

* `args`: an object holding additional arguments specific to the chosen loss function. These may vary depending on the loss function type. 
  * You can obtain args list from `table [loss]Options` attributes. please refer [here](https://github.com/Samsung/ONE/blob/master/runtime/libs/circle-schema/include/circle_traininfo.fbs). 
  * In addition to `table [loss]Options` attribute, **you should add a `reduction`** to specify how loss value will be shown. 
  * For example, to use 'mse' loss function, since there is nothing in the [`table MeanSquaredErrorOptions`](https://github.com/Samsung/ONE/blob/1a1a52afd87154720c28420d9a6804191421d5de/runtime/libs/circle-schema/include/circle_traininfo.fbs#L97-L98), you only need to add `reduction` to `args`.


  ```json
  {
    ...
    "loss":{
      "type" : "mse",
      "args" : {
        "reduction" : "sum over batch size"
      }
    }
    ...
  }
  ```

**Supported Loss Functions:**
  * sparse categorical crossentropy 
  * categorical crossentropy  
  * mean squared error 
