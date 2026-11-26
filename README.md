# ECS 174 Final Group Project

## Commands
Load Dataset   
`python colorize.py filter --out valid_images.txt`   
   
Quick Training Sanity Check   
`python colorize.py train --list valid_images.txt --out runs\smoke --n-train 500 --n-val 100 --n-test 100 --epochs 1`   