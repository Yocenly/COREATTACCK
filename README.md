# COREATTACCK

### 1. Datasets Download
As the datasets is too large to be uploaded to Github, we transfer them to Google Drive, 
[data_url](https://drive.google.com/file/d/1r1TqzlBdrjDP5lUTX7RwDf7RikH7dWUT/view?usp=sharing).

This compressed package contains two directories, ***cache*** and ***datasets***.
> ***cache*** saves the results of our experiments with .xslx format; <br />
> ***datasets*** saves the datasets used in the experiment; <br />

The directory list should be formed as bellow:

    COREATTACK
    ----cache
    ----datasets
    ----the other .py code files

### 2. Code Files

> **BasicMethods.py**: Some basic definitions and basic methods are defined here;<br />
> **CoreAttack.py**: Functions of COREATTACK, GreedyCOREATTACK, RED, and HDN methods;<br />
> **NodeCollapse.py**: Functions of CKC and HDE methods;<br />
> **EfficiencyEvaluation.py**: Visualization of time consumption.

### 3. Run Experiments

Requirements:

    numpy >= 1.20.3
    networkx >= 2.6.3
    pandas >= 1.3.4
    matplotlib >= 3.5.0

Run interface of edge-based methods, including COREATTACK, GreedyCOREATTACK, RED, and HDN:

    python CoreAttack.py
    
Run interface of node-based methods, including CKC and HDE:

    python NodeCollapse.py
    
Visualization of time consumption:

    python EfficiencyEvaluation.py

The results will be stored into ***cache*** directory.
