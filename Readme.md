# CV Use Case for Neptune

## Installation 

Install the following packages: 
```
pip install tensorflow neptune numpy 
```

## How to run 

Run the script mnist_train.py to train the model and upload the results to Neptune:

```sh
python mnist_train.py --num_epochs 10 --learning_rate 0.001 --run_name "My experiment"
```

See the Neptune project [here](https://app.neptune.ai/o/emma.saroyan/org/Mnist/runs/compare?viewId=standard-view&dash=charts).