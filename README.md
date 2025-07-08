# KG-DRPC: A biological knowledge graph computational framework for drug response prediction in cancer cell lines
KG-DRPC, a novel knowledge graph-based computational framework designed to enhance the prediction of drug responses within cancer cell lines by leveraging the synergy between biological knowledge graphs (KG) and machine learning. KG-DRPC integrates multifaceted biological data, including genomics and pharmacogenomics, to construct a comprehensive KG pertaining to the drug response of cancer cell lines, named CCD-KG. 

The current version (v1.0) of the KG-DRPC model


# Environment set up for training and testing of KG-DRPC
KG-DRPC training/testing scripts require the following environmental setup:


* Software
    * Python >=3.6
    * Anaconda
        * Relevant information for installing Anaconda can be found in: https://docs.conda.io/projects/conda/en/latest/user-guide/install/.
    * tensorflow '1.15.0'
    * pandas '1.1.5'
    * numpy '1.18.4'
    * netwopandas '1.1.5'
    * ampligraph '1.3.2'
    * deepctr '0.8.4'


Required input files:
1. `Dataset/`: training set and test set in the 30 cancer types of cell line drug response
    * `Disease_dataset/`: The data set by cancer type is named by the ID of the cancer class.
    * `All.csv`: Total cancer cell line drug response dataset
    * `train_fd_.csv`,`test_fd_.csv`: A tab-delimited file for model training and testing that needs to be generated manually.
    
2. `KG_data/`: knowledge graph data
    * `Cell_Property.txt`: Subset of knowledge graphs related to cells.
    * `Drug_Property.txt`: Subset of knowledge graphs related to drugs.
    * `Gene_GO.txt`: Subset of knowledge graphs related to genes.
    * `KG_all.txt`: Subset of knowledge graphs related to Cell-drug association pairs.
    * `Cell_feature.csv`: The copy number used as the auxiliary information of the cells.
    * `finger__out.cssv`: Fingerprint features as auxiliary information for drugs.

3. `code/`: KG-DRPC part of the code
    * `data_processing.py`: Process the data to generate cross-validated datasets by cancer type
    * `KG_DRPC.py`: Model core code


    
To make prediction for (cell, drug) pairs of your interest, execute the following:

1. Make sure you have Cell_Property.txt_, Drug_Property.txt_, Gene_GO.txt_, KG_all.txt_, finger__out.csv, Cell_feature.csv

2. Run data_processing.py to obtain the training and testing datasets, which are split by cancer type.
 
3. Run KG_DRPC.py to train the model and generate results across various test sets.



     
## Contact

- Please feel free to contact us if you need any help: [hanghu@stu.ahau.edu.cn]
- __Attention__: Only real name emails will be replied. Please provide as much detail as possible about the problem you are experiencing.


