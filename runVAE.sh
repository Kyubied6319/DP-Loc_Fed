#!/bin/bash

#./certificates/generate.sh

echo "Starting server"

python serverVT.py $1 $2 $3 &
sleep 3  # Sleep for 3s to give the server enough time to start

# Ensure that the Keras dataset used in client.py is already cached.
echo "Opening Clients"

python tClient.py $1 $2 $3 True & python tClient.py $1 $2 $3 False 

echo "Clients finished"
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
echo "save now or never"
sleep 5
wait
