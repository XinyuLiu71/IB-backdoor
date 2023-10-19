# 1. Generate Backdoor dataset
## In poison_data_generator directory
## poison_percentage: the poison ratio in single class, the suffix of the output path must be .npz

RUN python badnet.py --poison_percentage=0.05 --output_path="../data/badnet_5.npz"

# 2. If your FFCV ENV doesn't works well
## the sampling_datasize must lower than 5000

RUN python observe_MI.py --observe_class=1 --output_dir="result/today" --training_epochs=100 --sampling_datasize=4000

## If your FFCV ENV works good.

RUN python ffcv_writer.py --output_path=train_data.beton --dataset=train_dataset

RUN python ffcv_writer.py --output_path=test_data.beton --dataset=test_dataset

RUN python ffcv_writer.py --sampling_datasize=4000 --observe_class=0 --output_path=observe_data.beton --dataset=sample_dataset

# 3. RUN FFCV_observeMI.py
RUN python ffcv_observeMI.py --train_data_path="xxx.beton" --test_data_path="xxx.beton" --sample_data_path="xxx.beton"
