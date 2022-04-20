# Docker Containers for Model Serving

These containers can serve the .mar archives created by the top-level `archive-model.sh` script.
While Python dependencies can be installed using the "-r" option on `torch-model-archiver`,
we ran into trouble due to native dependencies.
This can be solved using these custom images which come with all dependencies we need.

## Building Images

To build a CPU image:
```
docker build -t coref-modelserver:cpu .
```

Build GPU compatbile image:
```
docker build --build-arg=TORCHSERVE_TAG=0.3.0-gpu -t coref-modelserver:gpu .
```

## Serve Archives using Docker

With `.mar` archives in the working directory run:

```
./serve.sh "model_name=some_model_name.mar"
```

Or to run the GPU image on GPU 0:
```
./serve.sh /dev/nvidia0 "model_name=some_model_name.mar"
```
