# Additional info for tensorflow-gpu    

NVIDIA CUDA compatibility table for GPUs:
https://developer.nvidia.com/cuda-gpus
Tensorflow-gpu needs CUDA capability of 3.0 or higher.

Tensorflow installation:
https://www.tensorflow.org/install/

NVIDIA CUDA Toolkit Installation Guide:
http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A

# Brief instructions for Ubuntu installation of NVIDA CUDA Toolkit 8.0 with cuDNN 6.0:

NVIDIA CUDA Toolkit 8.0 download:
https://developer.nvidia.com/cuda-80-ga2-download-archive

There are 2 deb packages to download, each one with their own pub keys to add.

For each deb package: 
`sudo dpkg --install <package-name>`
`sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub`

And then:
`sudo apt-get update. sudo apt-get install cuda`
`export PATH=$PATH:/usr/local/cuda-8.0/bin`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64`

NVIDIA CUDA Deep Neural Network Library (cuDNN v6.0 for CUDA 8.0):
https://developer.nvidia.com/rdp/cudnn-download
