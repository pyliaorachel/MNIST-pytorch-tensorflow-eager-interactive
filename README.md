# MNIST Interactive Examples in PyTorch & TensorFlow Eager Mode

Modified from PyTorch MNIST official example. Recreated with TensorFlow under eager execution mode. With detailed comments & interactive interface.

[深度學習新手村：PyTorch入門（中文）](https://pyliaorachel.github.io/blog/tech/deeplearning/2017/10/16/getting-started-with-deep-learning-with-pytorch.html)

## Structure

```
pytorch/
  train.py            # Train the model
  model.py            # The defined model
  app.py              # Interactive predictor
  model               # Pretrained model, will be overriden when you start training
  test_n.png          # Sample images for the use of interactive predictor
  
tensorflow_eager/
  train.py            # Train the model
  model.py            # The defined model
  app.py              # Interactive predictor
  checkpoint, ckpt-*  # Pretrained model, the number after prefix is the final training step
  test_n.png          # Sample images for the use of interactive predictor
```

## Usage

```bash
# clone project
$ git clone https://github.com/pyliaorachel/pytorch-mnist-interactive.git
$ cd MNIST-pytorch-tensorflow-eager-interactive

# install dependencies
$ pip3 install -r requirements.txt

# train & test model
$ python3 -m pytorch.train
# ...data will be fetched to ../data/
# ...trained model will be saved to ./pytorch/model
# or
$ python3 -m tensorflow_eager.train
# ...data will be fetched to somewhere
# ...trained model will be saved to ./tensorflow_eager/checkpoint & ./tensorflow_eager/ckpt-*

# test model interactively
$ python3 -m pytorch.app --image=<path-to-image>
# or
$ python3 -m tensorflow_eager.app --image=<path-to-image>
```

## Experiments

###### Machine Settings

|OS|CPU|Memory|
|:-:|:-:|:-:|
|MacOS 10.12.4|2 GHz Intel Core i5|16 GB 1867 MHz LPDDR3|

###### Results

||TensorFlow Eager|PyTorch|
|:-:|:-|:-|
|Time|`real	6m4.446s` </br> `user	13m42.909s` </br> `sys	1m54.327s`|`real	3m59.340s` </br> `user	3m28.285s` </br> `sys	0m57.395s`|
|Avg. Loss (Test)|0.0610|0.0473|
|Accuracy| 9856/10000 (99%) | 9845/10000 (98%) |

Avg. Loss and Accuracy are expected to be more or less the same. PyTorch is half the time of TensorFlow's on CPU, while the code complexity is the same.
