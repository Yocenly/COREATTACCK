# COREATTACCK

### 1. Datasets Download
As the used datasets is too large to upload to Github, we transfer them to Google Drive, 
[datasets_url](https://drive.google.com/file/d/19d0YDQ2bfUUFqTiXl0QzHywVgV4URiGG/view?usp=sharing).

The compressed package contains three directories, ***cache***, ***datasets***, and ***graphml***.
> ***cache*** is made to save the results of our experiment; <br />
> ***datasets*** is made to save the datasets we use in the experiment; <br />
> ***graphml*** saves the .ml files of our datasets. 

Actually, only ***cache*** and ***datasets*** are used in this project.

The directory list should be formed as bellow:

    COREATTACK
    ----cache
    ----datasets
    ----graphml
    ----the other .py code files

### 2. Code Files
We define some basic functions about COREATTACK, GreedyCOREATTACK, and our baselines
in BasicMethods.py, CoreAttack.py, and NodeCollapse.py.

If someone wants to run the experiment, please run CoreAttack.py without any parameters, we apply 4 processes to
conduct the COREATTACK, GreedyCOREATTACK, random baseline and heuristic baseline respectively.

Requirements:

    numpy >= 1.20.3
    networkx >= 2.6.3
    pandas >= 1.3.4
    matplotlib >= 3.5.0

Run Interface of COREATTACK:

    python CoreAttack.py
    
Visualization of Time Consumption:

    python EfficiencyEvaluation.py

The results will be stored into ***cache*** directory.
