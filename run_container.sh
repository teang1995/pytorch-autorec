docker run -itd --rm -v $(pwd):/pytorch-autorec --gpus="device=0" --name autorec autorec:v1.0 