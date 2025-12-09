# robustClassify
Here we perform evaluation of adversarial robustness of image classification models



### Installations

Create a conda enviornmnet :

`conda create -n robustness python=3.14`

Install pytorch and torchvison :

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

Install datasets library: 
`pip install datasets`

Activate the environment :

`conda activate robustness`

Download the imagenet subset of 10k images from hugging face (https://huggingface.co/datasets/Oztobuzz/ImageNet_10k/tree/main/data ).

There are 4 files train-00000-of-00004.parquet, train-00001-of-00004.parquet, train-00002-of-00004.parquet, train-00003-of-00004.parquet

Make a directory called imagenetparaquet and add these files .

Then make a directory called imagenetDataSubset and run : 

`python datasetSaver.py `
