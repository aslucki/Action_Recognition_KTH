build_docker:
	docker build -t general-gpu .

run_docker:
	docker run --name work_env --runtime=nvidia --rm -ti -p 9000:9000 -v $(CURDIR):/mnt general-gpu

start_notebook:
	jupyter notebook --allow-root --no-browser --ip 0.0.0.0 --port 9000

create_mhi:
	python3 src/create_mhi.py --input_dir data/source --output_dir data/processed/MHI

extract_shallow:
	python3 src/extract_shallow_features.py \
			--input_dir data/processed/MHI \
			--output_file data/processed/shallow_features.h5
extract_deep:
	python3 src/extract_resnet_features.py \
			--input_dir data/source \
			--output_file data/processed/resnet_features.h5
train_basic_lstm:
	python3 src/train_lstm_keras.py \
			--input_file data/processed/resnet_features.h5
