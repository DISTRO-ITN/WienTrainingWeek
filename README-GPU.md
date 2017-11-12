# Installation instructions for tensorflow with GPU support

NVIDIA CUDA compatibility table for GPUs:  
https://developer.nvidia.com/cuda-gpus  
Tensorflow-gpu needs CUDA capability of 3.0 or higher.

Tensorflow installation:  
https://www.tensorflow.org/install/

<br>
<br>
<br>

### Brief instructions for Ubuntu installation (NVIDA CUDA Toolkit 8.0 with cuDNN 6.0):

Download NVIDIA CUDA Toolkit 8.0:  
https://developer.nvidia.com/cuda-80-ga2-download-archive

There are 2 deb packages to download, each one with their own pub keys to add. For each deb package, type in terminal:  
`sudo dpkg --install <package-name>`  
`sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub`

And then:  
`sudo apt-get update. sudo apt-get install cuda`  
`export PATH=$PATH:/usr/local/cuda-8.0/bin`  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64`

Download and install NVIDIA CUDA Deep Neural Network Library (cuDNN v6.0 for CUDA 8.0):  
https://developer.nvidia.com/rdp/cudnn-download

Finally install the NVIDIA CUDA Profile Tools Interface. Type in terminal:  
`sudo apt-get install libcupti-dev`
