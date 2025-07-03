# Fine-Grained Deep Imitation Learning for Balancing Efficiency and Safety in Traffic Signal Control
## Setup
The code must be run on a Windows desktop system, as the simulation tool is only available for Windows. Ensure that PTV VISSIM 2022 is installed on the machine.
## Unzip files
Before execution, unzip the folder and extract the contents of data.7z, SSAM.7z, and Vissim_model.7z. Then, install all required Python packages.
## Model training
You can run the following command to train the results:
```
python train.py
```
## Model inference
Set the dataset path in the config.py file:
pathVehDemandFile = './demand/dataset1.xlsx'
You can switch to dataset2.xlsx or dataset3.xlsx to use different datasets.
To replicate the results, run the following command:
```
python test.py
```
