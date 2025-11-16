# cse5100_airs_repro

## Command lines
```cmd
# Empty-16x16
python -m scripts.train_minigrid --env_id MiniGrid-Empty-16x16-v0 --mode a2c --exp_name baseline
python -m scripts.train_minigrid --env_id MiniGrid-Empty-16x16-v0 --mode a2c_re3 --exp_name re3
python -m scripts.train_minigrid --env_id MiniGrid-Empty-16x16-v0 --mode airs --exp_name airs

# DoorKey-6x6
python -m scripts.train_minigrid --env_id MiniGrid-DoorKey-6x6-v0 --mode a2c --exp_name baseline
python -m scripts.train_minigrid --env_id MiniGrid-DoorKey-6x6-v0 --mode a2c_re3 --exp_name re3
python -m scripts.train_minigrid --env_id MiniGrid-DoorKey-6x6-v0 --mode airs --exp_name airs
```
