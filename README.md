# FedDP-Loc
This code generates differentially private synthetic location traces in a Federated Learning environment. The algorithm, the model parameters, and everything that you need to know can be found in the paper:
Lestyán, Szilvia, Gergely Ács, and Gergely Biczók. "In Search of Lost Utility: Private Location Data." Proceedings on Privacy Enhancing Technologies 3 (2022): 354-372.
https://petsymposium.org/2022/files/papers/issue3/popets-2022-0076.pdf

## Usage
1. Install dependency manager Poetry https://python-poetry.org/docs/#installation
2. Create venv and install dependencies of pyproject.toml via `Poetry install`
3. Activate venv via `Poetry shell`
4. Install dependencies via the command `pip install -r requirements.txt`
5. Create venv with libraries in requirements.txt
6. Edit `cfg/*.cfg`
7. Run `run.py` for the evaluation, which runs the following file successively:
    - Step 1: Mapping GPS to grid (`create_mapped_data.py`)
    - Step 2: Train VAE and Next-hop models (`training.py`)
    - Step 3: Generate synthetic data (`generator.py`)
    - Step 4: Evaluate the generated data (`evaluate.py`)

The output of each step is saved to "out/" and the models are saved to "saved_model"

## NOTES:
- input must be in a format of San Francisco dataset or the Porto dataset, where different trips are separated with some separator (here it is a flag) (use create_mapped_data_SF_Porto.py). In case of the GeoLife dataset this is not true, trips are continous, the recording goes on even when the individual is not moving for a longer period of time. For this we have created an alternative create_mapped_data_Geo.py code. This code cuts the input into smaller pieces. Copy the one you want to use to create_mapped_data.py and you wont have to rewrite it run.py
- output and data directories must be created: "out", "saved_model" and "datasets" -  or anything alternatively that you set in the cfg/cfg_general.json
- in cfg/cfg_general.json you have to set the GPS coordinates of the bounding box. Be careful with anomalies in the data if you want to set it automatically, data is not clean usually. 
- config files are provided for all datasets. Copy the one in use under cfg/cfg_general.json so you dont have to update its name in the code. 
- you can set the CELL_SIZE in run.py and NOT IN cfg/cfg_general.json
- you can parallelize the code by setting the CPU_CORES parameter in the config file. Other parameters are also configurable in there.
- the cfg/cfg_eps files also follow the previous usage, copy the one you would like to use under the name that ends with a number. Eg: cfg_eps1.json

## Dataset 1: Taxis of San Francisco

https://crawdad.org/epfl/mobility/20090224/
Taxi traces are split into taxi trips/rides. Each trip is assumed to belong to separate individuals. 

## Dataset 2: Taxis of Porto
https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge,+ECML+PKDD+2015
Similar to SF dataset, but in order to transform it to the same format use the code PortoToSF_format.py

## Dataset 3: GeoLife Beijing
https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F
Different from previous two, use create_mapped_data_Geo.py for prerpocessing.

