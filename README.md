# WienTrainingWeek

1) Download Anaconda (https://www.anaconda.com/download/)

2) Install it , then use only one of this three methods.


Window Section

       ####### From yml file (choose yml and rename it without _yoursystem)

       3) conda env create -f keras-cpu.yml
       
       4) activate environment: activate keras-cpu



       ####### From requirement.txt file (choose txt and rename it without _yoursystem)

       3) conda create -n myEnv python=3

       4) activate the environment :

          activate myEnv

       5) pip install -r requirements.txt



       ####### Manual 

       3) Create an environment:    

       conda create -n myEnv python=3

       4) activate the environment :

          activate myEnv

      
       5) Install packages:

       conda install tensorflow

       conda install keras

       conda install jupyter

       conda install pillow

       conda install h5py    

       conda install -other packages-
