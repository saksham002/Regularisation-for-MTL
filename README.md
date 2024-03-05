# Dataset Preparation

Download the dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and place the annotations, train-test-validation split and images inside the Datasets/CelebA/ folder with names "Anno", "Eval" and "Img".

# Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch, click, numpy, torchvision, tqdm, scipy, Pillow``

The code is only tested in ``Python 3`` using ``Anaconda`` environment.

# Usage
The code base uses `configs.json` for the global configurations like dataset directories, etc.. Experiment specific parameters are provided seperately as a json file. See the `sample.json` for an example.

To train a model, use the command: 
```bash
python multi_task/train_multi_task.py --param_file=./sample.json
```
