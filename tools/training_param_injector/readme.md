# inject training parameter into circle model


#### 1. inject training parameter through command line

You can write training parameter in command line and rewrite mnist.cirle

```bash 
python3 main.py example/mnist.circle -v

Please input training parmameter you want to inject.
If you enter without entering, first parmater is always selected.

optimizer (adam, sgd) : 
	 ㄴlearning_rate (0.001) : 
	 ㄴbeta1 (0.9) : 
	 ㄴbeta2 (0.999) : 
	 ㄴepsilon (1e-07) : 
loss (mse, cce) : 
	ㄴreduction(SUM_OVER_BATCH_SIZE, SUM) : 
batch_size (32) : 
[INFO] sucessfully inject training parameter
[INFO] save updated circle file in example/mnist.circle
```

#### 2. inject training parameter using json file 

Here is an exmample of json file : 
```bash
➜ cat example/train_parameter.json 
{
  "optimizer":{
    "type": "Adam",
    "args": {
      "learning_rate" : 0.01,
      "beta1": 0.9,
      "beta2": 0.999,
      "epsilon": 1e-07
    }
  },
  "loss" : {
    "type": "CategoricalCrossentropy",
    "args":{
      "from_logits": true,
      "reduction": "SumOverBatchSize"
    }
  },
  "batch_size":32
}
```

pass json file using `-t` options.

```bash
➜ python3 main.py example/mnist.circle -t example/train_parameter.json -v
[INFO] sucessfully inject training parameter
[INFO] save updated circle file in example/mnist.circle
```
