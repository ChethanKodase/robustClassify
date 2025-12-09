# robustClassify
Here we perform evaluation of adversarial robustness of image classification models



### Installations

Create a conda enviornmnet :

`conda create -n robustness python=3.14`

Activate the environment :

`conda activate robustness`

Install pytorch and torchvison :

`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`


Install other libraries: 
`pip install datasets`

`pip install numpy`

`pip install matplotlib`


Install Kaggel: 

`pip install kaggle`

Create a new API in Kaggel and get kaggel.json file and then 


```
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

```

Then change directory to `robustClassify` using : 
`cd robustClassify`

Then run :

`kaggle datasets download -d ifigotin/imagenetmini-1000`

Uzip the downloaded file :

`unzip imagenetmini-1000.zip -d imagenet-mini`

`python efficientnetAttack2.py`



