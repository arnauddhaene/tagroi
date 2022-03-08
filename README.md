# tagroi


### Useful commands

Launch docker container (replace xx.xx with Docker version - current install is `20.10`)

```bash
docker run --gpus all -it -v $(pwd):/workspace/ --rm nvcr.io/nvidia/pytorch:xx.xx-py3
```

Check GPU usage (auto-update every second)

```bash
watch -n 1 nvidia-smi
```

To get a bash shell within a running container (you can view running containers with ```docker ps```)

```bash
docker exec -it <container name> /bin/bash
```
