#! /bin/bash

tmux new -d -s tensorboard_session 'tensorboard --logdir ./projects/ --bind_all'