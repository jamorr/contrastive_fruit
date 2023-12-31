# Contrastive learning on lemons dataset using Lightly Self-Supervised learning
## Usage Instructions
Reccomended hardware required to train the model:
A computer with:
- hardware virtualization enabled and Docker installed
- ~17GB of available storage
- Nvidia GPU with at least 6GB of VRAM
- 8GB of RAM


To set up the docker container, first build the container
`docker build -t pytorch-gpu . -f Dockerfile`
The build process may take 10min+ due to the size of the python packages and the cuda dev kit.


Then to test and train the model you need to download at least one of the following datasets and extract them into a folder called fruit in the main directory of this repository.
  1.  https://github.com/softwaremill/lemon-dataset
  2.  https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset (its recommended to delete the .git and empty background directories)
  3.  https://www.kaggle.com/datasets/saurabhshahane/mango-varieties-classification

Then to run the container  
`docker run --name pytorch-container --gpus all --shm-size 8G -it --rm -v '<your dir name>':/app pytorch-gpu`  
if run from a CLI, this should spawn a bash shell inside the container  

To train the model, navigate to app/src in the container and run  
`python3 main.py --dataset_train <path to your training dataset> --weights_file <file where you want to save the model weights>`  

To test a model, navigate to app/src in the container and run  
`python3 main.py --dataset_test <path to testing dataset (directory)> --weights <file with model weights> --testing`  
currently this is only implemented for SimCLR so do not use for DINO weights.  

The results seen in the paper can be found in the results folder. The model and dataset used to produce results are found inside `annotations.json`. Some examples of KNN neighbors and the confusion matrix are also available.
