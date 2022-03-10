# tagroi


### Useful commands

Launch docker container (replace xx.xx with Docker version - current install is `20.10`)

```bash
docker run --gpus all --ipc=host -it -v $(pwd):/workspace/ --rm nvcr.io/nvidia/pytorch:xx.xx-py3
```

If you want to launch it in daemon mode (background), use this

```bash
docker run --gpus all --ipc=host -d -v $(pwd):/workspace/ nvcr.io/nvidia/pytorch:xx.xx-py3 /bin/sh -c "while true; do ping 8.8.8.8; done"
```

Check GPU usage (auto-update every second)

```bash
watch -n 1 nvidia-smi
```

To get a bash shell within a running container (you can view running containers with ```docker ps```)

```bash
docker exec -it <container name> /bin/bash
```

 ### Using tmux


New session with `tmux`. See available sessions with `tmux ls`. Attach to existing session using `tmux attach -t <session-ID>`

Kill session with `Ctrl + b` followed by typing `kill-session`.  To scroll within a session, use `Ctrl + b` followed by `[` and quit the scrolling mode with `q`.

Cheatsheet available [here](https://tmuxcheatsheet.com/).
