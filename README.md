# COREATTACCK

### 1. Datasets Download
As the used datasets is too large to upload to Github, we transfer them to Google Drive, 
[datasets_url](https://drive.google.com/file/d/19d0YDQ2bfUUFqTiXl0QzHywVgV4URiGG/view?usp=sharing).

The compressed package contains three directories, ***cache***, ***datasets***, and ***graphml***. 
Among them, ***cache*** is made to save the results of our experiment; 
***datasets*** is made to save the datasets we use in the experiment; ***graphml*** saves the 
.ml files of our datasets. Actually, only ***cache*** and ***datasets*** are used in this project.

### 2. Code Files
We define some basic functions about COREATTACK, GreedyCOREATTACK, and our baselines
in BasicMethods.py, CoreAttack.py, and NodeCollapse.py.

If someone wants to run the experiment, please run CoreAttack.py without any parameters, we apply 4 process to
conduct the COREATTACK, GreedyCOREATTACK, random baseline and heuristic baseline respectively.

Example: *python CoreAttack.py*
