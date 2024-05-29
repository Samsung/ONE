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

* `type` : a string to indiciate optimizer (e.g. `adam`) 
* `args` : an object holding additional arguments to the chosen optimizer. These may vary depending on the optimizer type, but typically include 'learningRate'. 

**Supported Optimizers:** <br/>
  * adam 
    | Key      | Key in args  |Data Type     | Example Values |
    |----------|-------       |--------      |----------------|
    | type     |              |string        | "adam"         |
    | args     |              |object        |                |
    |          | learningRate |number        | 0.01           |
    |          | beta1        |number        | 0.9            |
    |          | beta2        |number        | 0.999          |
    |          | epsilon      |number        | 1e-07          |

  * sgd 
    | Key      | Key in args  |Data Type     | Example Values |
    |----------|-------       |--------      |----------------|
    | type     |              |string        | "sgd"          |
    | args     |              |object        |                |
    |          | learningRate |number        | 0.001          |


## loss

An object describing the loss function. This should include two keys :

* `type`: a string specifying which loss function to use. (e.g. `mse`)
* `args`: an object holding additional arguments specific to the chosen loss function. These may vary depending on the loss function type, but typically include 'reduction'. 

**Supported Loss Functions:**
  * sparse categorical crossentropy 
    | Key     | Key in args  |Data Type | Example Values                     |
    |---------|------------- |----------|----------------                    |
    | type    |              |string    | "sparse categorical crossentropy"  |
    | args    |              |object    |                                    |
    |         | fromLogits   |boolean   | true, false                        |
    |         | reduction    |string    | "sum over batch size","sum"        |

  * categorical crossentropy  
    | Key     | Key in args  |Data Type | Example Values                     |
    |---------|------------- |----------|----------------                    |
    | type    |              |string    | "categorical crossentropy"         |
    | args    |              |object    |                                    |
    |         | fromLogits   |boolean   | true, false                        |
    |         | reduction    |string    | "sum over batch size", "sum"       |

  * mean squared error 
    | Key     | Key in args  |Data Type    | Example Values                   |
    |---------|------------- |----------   |----------------                  |
    | type    |              |string       | "mean squared error"             |
    | args    |              |object       |                                  |
    |         | reduction    |string       | "sum over batch size", "sum"     |

