#!/bin/bash

N_ITR=5


for i in $(seq $N_ITR)
do
    echo "itr ${i}/${N_ITR}"
    echo "Default config..."
    python src/data_gen.py --output-dir data/itr${i} --output-name sch_FIFO-tgen_Poisson-n_flows_100-n_ports_4

    echo "varying scheduler..."
    python src/data_gen.py --scheduler DRR --output-dir data/itr${i} --output-name sch_DDR-tgen_Poisson-n_flows_100-n_ports_4
    python src/data_gen.py --scheduler WFQ --output-dir data/itr${i} --output-name sch_WFQ-tgen_Poisson-n_flows_100-n_ports_4
    python src/data_gen.py --scheduler SP --output-dir data/itr${i} --output-name sch_SP-tgen_Poisson-n_flows_100-n_ports_4

    echo "varying traffic gen..."
    python src/data_gen.py --traffic-gen OnOff --output-dir data/itr${i} --output-name sch_FIFO-tgen_OnOff-n_flows_100-n_ports_4

    echo "varying num flows..."
    python src/data_gen.py --num-flows 20 --output-dir data/itr${i} --output-name sch_FIFO-tgen_Poisson-n_flows_20-n_ports_4
    python src/data_gen.py --num-flows 60 --output-dir data/itr${i} --output-name sch_FIFO-tgen_Poisson-n_flows_60-n_ports_4

    echo "varying num ports..."
    python src/data_gen.py --num-ports 8 --output-dir data/itr${i} --output-name sch_FIFO-tgen_Poisson-n_flows_100-n_ports_8
    python src/data_gen.py --num-ports 16 --output-dir data/itr${i} --output-name sch_FIFO-tgen_Poisson-n_flows_100-n_ports_16
    
done