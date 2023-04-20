# COS561_final_project

### Setting Up
1. Create a new virtual environment using `conda create --name dqn --file environment.yml`
2. Activate environment using `conda activate dqn`

### Download Anarchy Data
1. Create a data directory with `mkdir data`
2. `cd data`
3. Download Anarcy PCAP data with `wget https://datasets.simula.no/downloads/anarchy-online-server-side-packet-trace-1hr.pcap`

### Download Data
1. Install gdown with `pip install gdown`
2. `cd data`
3. Download raw data with `gdown https://drive.google.com/uc?id=142hbD3ZWDJ7Lqh_SuOuFzwZPgsJ9BzxT` and processed data with `gdown https://drive.google.com/uc?id=1-YqHXIkfFYtXCysdcmoj3BgnjrLPoqvW`
4. Untar the compressed files with `tar -xvzf raw_data.tar.gz` and `tar -xvzf processed_data.tar.gz`