# PyTorch MNIST Interactive Example

PyTorch MNIST official example with comments & interactive interface.

## Usage

```bash
# clone project
git clone https://github.com/pyliaorachel/pytorch-mnist-interactive.git
cd pytorch-mnist-interactive

# install dependencies
pip3 install -r requirements.txt

# train & test model
python3 -m src.train
# ...data will be fetched to ../data/
# ...trained model will be saved to ./src/model

# test model interactively
python3 -m src.app --image=<path-to-image>
```
