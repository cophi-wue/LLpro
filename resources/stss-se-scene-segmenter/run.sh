#!/bin/bash
docker run -v "$(pwd)"/data/test:/data/test --name stss-eval-container stss-eval
docker cp stss-eval-container:predictions "$(pwd)"
docker stop stss-eval-container
docker rm stss-eval-container
