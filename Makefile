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


extract_autoencoder_features:
	python3 src/extract_autoencoder_features.py \
			--autoencoder data/external/model_autoencoder.h5 \
			--input_file data/processed/resnet_features.h5 \
			--output_file data/processed/autoencoder_features.h5

calculate_optical_flow:
	python3 src/calculate_optical_flow.py \
			--input_dir data/source \
			--output_file data/processed/optical_flow_features.h5

train_svc_autoencoder:
	python3 src/train_svc_scikit.py \
			--input_file data/processed/autoencoder_features.h5 \
			--features_ds_name autoencoder_features

train_svc_hog:
	python3 src/train_svc_scikit.py \
			--input_file data/processed/shallow_features.h5 \
			--features_ds_name hog

train_svc_zernike:
	python3 src/train_svc_scikit.py \
			--input_file data/processed/shallow_features.h5 \
			--features_ds_name zernike

train_basic_lstm:
	python3 src/train_lstm_keras.py \
			--input_file data/processed/resnet_features.h5

