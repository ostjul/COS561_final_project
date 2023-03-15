# COS561_final_project

### Setting Up
1. Create a new virtual environment using `conda create --name dqn --file environment.yml`
2. Activate environment using `conda activate dqn`

### Download Anarchy Data
1. Create a data directory with `mkdir data`
2. `cd data`
3. Download Anarchy PCAP data with `wget https://datasets.simula.no/downloads/anarchy-online-server-side-packet-trace-1hr.pcap`

### Download DQN Data
1. Download zip file with `wget https://www.dropbox.com/s/q56sx4hxe93n4g5/DeepQueueNet-dataset.zip?dl=1 -O DeepQueueNet-dataset.zip`
2. Unzip using `unzip DeepQueueNet-dataset.zip`
3. Rename folder with `mv 'DeepQueueNet-synthetic data'/ dqn_data`
    * There should be the following directory structure:
        data
          |--dqn_data
              |--4-port switch
              |--fattree16

4. Delete zip file with `rm DeepQueueNet-dataset.zip`