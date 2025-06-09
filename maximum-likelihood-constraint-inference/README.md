This repository is based on the [MATLAB implementation](https://drive.google.com/drive/folders/1h2J7o4w4J0_dpldTRpFu_jWQR8CkBbXw) provided by the authors of the following paper:
* [Scobee, Dexter RR, and S. Shankar Sastry. "Maximum Likelihood Constraint Inference for Inverse Reinforcement Learning." International Conference on Learning Representations. 2019.](https://openreview.net/forum?id=BJliakStvH)


Notes
=====
* OpenGL is required.
* For creating GIFs, install Imagemagick (`convert`) and `gifsicle`. Ensure that they can be accessed from command-line.
* `inD/000_background.png` has been created from `inD/00_background.png` using GIMP.
* `inD/00_*` are the zeroth track files from inD dataset (`data` folder in the inD dataset).
* `sim.py` contains code to read inD dataset as well as visualize through Pyglet.
* `main.py` contains code to do constraint inference on the inD dataset.
* `gridworld.py` contains code to do constraint inference on the Gridworld example from the MLCI paper.

Workflow
========
* To install pip requirements, run `pip3 install -r requirements.txt`
* To run Gridworld constraint inference example from the MLCI paper, run `python3 -B gridworld.py`
* To run the inD example,
    * Generate `pickles/trajectories.pickle` by running `python3 -B sim.py --generate_discrete_data`
    * Generate `pickles/constraints.pickle` and `pickles/new_demonstrations.pickle` by running `python3 -B main.py --do_constraint_inference`. To also generate policy plots, add `--policy_plot` flag as well.
    * Visualize the constraints and the dataset by running `python3 -B sim.py --visualize`.
    * (Optional) Produce GIFs by running `python3 -B sim.py --create_gifs`. This will create `frames.gif` and `policy.gif`.
    * Visualize the constraints and the generated demonstrations (from the final MDP) by running `python3 -B sim.py --visualize --show_new_demos`.
    * (Optional) Produce GIF by running `python3 -B sim.py --create_gifs --show_new_demos`. This will create `demos.gif`.
* To run the inD example in the multi-goal setting, run `sim.py` and `main.py` as in the previous step, but with `--multi_goal` flag as well.
