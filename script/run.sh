#!/bin/bash
source activate nerf
cd ..

# Run the command in the background using nohup
nohup python train.py --config configs/lego.txt > ./script/out.txt &

# Save the process ID (PID) to a file
echo $! > ./script/save_pid.txt

echo "Command started in the background. Log saved to out.txt and PID saved to save_pid.txt."
