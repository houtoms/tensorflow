#!/bin/bash
( cmdpid=$BASHPID; (sleep 10; kill -s SIGINT $cmdpid) & exec tensorboard --logdir /tmp )
