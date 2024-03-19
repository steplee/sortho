#!/bin/bash

docker run --rm -it -v `pwd`/sorthoViz/:/app -w /app -p 9001:9001 node:slim /bin/bash -c "npm i"
docker run --rm -it -v `pwd`/sorthoViz/:/app -w /app -p 9001:9001 node:slim /bin/bash -c "npm run dev-server"
