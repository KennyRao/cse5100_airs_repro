# cse5100_airs_repro

## Command lines
Run the following commands first
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
To plot above results, run
```cmd
python -m scripts.plot_minigrid_results --results_dir results --out_path minigrid_comparison.png
```
