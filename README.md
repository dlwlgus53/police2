Just in case..

```
yum install wget
yum install git
```
Add user
```
adduser test
su - test
```

conda install

```
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
bash Anaconda3-2023.07-2-Linux-x86_64.sh
source ~/.bashrc
```

```
conda create -n test python=3.8.5
conda activate test 
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

download the model in here
using gdown (pip install gdown)

gdown "https://drive.google.com/uc?id=1IWcaFJp2Q23CzaePhNLQiah26xGwtWKw"

