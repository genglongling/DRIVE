import viz
import pandas
import glob
import numpy as np
import time
import pyglet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--generate_discrete_data", action="store_true")
parser.add_argument("--multi_goal", action="store_true")
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--show_new_demos", action="store_true")
parser.add_argument("--create_gifs", action="store_true")
args = parser.parse_args()

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random
import torch.optim as optim
import torch.nn as nn

# continuous state

# Preprocess and filter the trajectories from the CSV file
def preprocess_and_filter_trajectories(csv_file, frame_rate=25, x_threshold=80, y_threshold=-80):
    df = pd.read_csv(csv_file)

    # Initialize the result list
    filtered_transitions = []

    # Process data for each unique trackId
    for track_id, group in df.groupby("trackId"):
        # Sort by frame to ensure correct time sequence
        group = group.sort_values("frame")

        # Check if the last state satisfies the filtering condition
        final_state = group.iloc[-1]
        first_state = group.iloc[0]
        if final_state["xCenter"] < x_threshold and final_state["yCenter"] < y_threshold and (
                first_state["xCenter"] > 160 or first_state["yCenter"] > -40):
            # Extract relevant columns for processing
            states = group[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values

            # Create transitions (current_state -> next_state)
            for i in range(len(states) - 5):
                current_state = states[i]
                next_state = states[i + 5]
                j = i
                while i - 10 < j < i + 10 or j < 0 or j > len(states) - 1:
                    j = random.randint(i - 20, i + 20)
                random_state = states[j]
                filtered_transitions.append([np.hstack((current_state, next_state)), 1])
                filtered_transitions.append([np.hstack((current_state, random_state)), 0])

    print(f"Filtered {len(filtered_transitions)} transitions.")
    return filtered_transitions


# Define the neural network model (TransitionPredictionNN)
class TransitionPredictionNN(nn.Module):
    def __init__(self, input_dim=12):
        super(TransitionPredictionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


# Load and prepare the dataset
def load_data(filtered_transitions, batch_size=64):
    # Convert the filtered transitions into tensors for training
    input_states = torch.tensor([t[0] for t in filtered_transitions], dtype=torch.float32)
    labels = torch.tensor([t[1] for t in filtered_transitions], dtype=torch.float32)

    # Create a DataLoader
    dataset = TensorDataset(input_states, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader


# Train the model
def train_model(train_loader, epochs=100, lr=0.001):
    model = TransitionPredictionNN(input_dim=12)  # Adjust input_dim based on your features
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features).squeeze()
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model


# Beam Search for trajectory generation (simplified example)
def beam_search(start_state, model, a_values, delta_t=0.4, max_depth=100):
    """
    Beam search to generate trajectories, prioritizing the highest reward.

    Args:
        start_state (list or np.ndarray): Initial state [x, y, vx, vy, ax, ay].
        model (torch.nn.Module): Trained neural network to predict state probabilities.
        a_values (list): Discrete search space for acceleration values.
        delta_t (float): Time step for state updates.
        max_depth (int): Maximum number of steps to generate trajectory.

    Returns:
        list: Generated trajectory (sequence of states).
    """
    current_state = start_state.tolist()
    trajectory = [current_state]  # Start trajectory with initial state

    for _ in range(max_depth):
        best_next_state = None
        highest_reward = -float('inf')

        # Generate all possible next states
        for ax_new in a_values:
            for ay_new in a_values:
                # Calculate the next state
                x, y, vx, vy, ax, ay = current_state
                vx_new = vx + ax_new * delta_t
                vy_new = vy + ay_new * delta_t
                x_new = x + vx_new * delta_t
                y_new = y + vy_new * delta_t
                next_state = [x_new, y_new, vx_new, vy_new, ax_new, ay_new]

                # Evaluate the next state using the model
                concatenated_state = np.hstack((current_state, next_state))
                state_tensor = torch.tensor(concatenated_state, dtype=torch.float32).unsqueeze(0)
                reward = model(state_tensor).item()

                # Update the best next state if this one has a higher reward
                if reward > highest_reward:
                    best_next_state = next_state
                    highest_reward = reward

        # Append the best state to the trajectory
        trajectory.append(best_next_state)
        current_state = best_next_state

    #print(trajectory)
    return trajectory

# Constants
F = 0
n = 35 # dimensionality of state-space
allowed_end_state = [945,946,947,948,980,981,982,983,1015,1016,1017,1018,1050,1051,1052,1053] # [320]
banned_start_state = [1087] # [361]

# Load tracks, tracksMeta, recordingMeta
tracks_files = glob.glob("inD/*_tracks.csv")
tracksMeta_files = glob.glob("inD/*_tracksMeta.csv")
recordingMeta_files = glob.glob("inD/*_recordingMeta.csv")

# Choose the 00_* files
tracks_file, tracksMeta_file, recordingMeta_file = tracks_files[F], tracksMeta_files[F], recordingMeta_files[F]

# Read tracksMeta, recordingsMeta, tracks
tm = pandas.read_csv(tracksMeta_file).to_dict(orient="records")
rm = pandas.read_csv(recordingMeta_file).to_dict(orient="records")
t = pandas.read_csv(tracks_file).groupby(["trackId"], sort=False)

# Normalization
xmin, xmax = np.inf, -np.inf
ymin, ymax = np.inf, -np.inf

bboxes = []
centerpts = []
frames = []
# iterate through groups
for k in range(t.ngroups):

    # Choose the kth track and get lists
    g = t.get_group(k).to_dict(orient="list")

    # Set attributes
    meter_to_px = 1. / rm[0]["orthoPxToMeter"]
    g["xCenterVis"] = np.array(g["xCenter"]) * meter_to_px
    g["yCenterVis"] = -np.array(g["yCenter"]) * meter_to_px
    g["centerVis"] = np.stack([np.array(g["xCenter"]), -np.array(g["yCenter"])], axis=-1) * meter_to_px
    g["widthVis"] = np.array(g["width"]) * meter_to_px
    g["lengthVis"] = np.array(g["length"]) * meter_to_px
    g["headingVis"] = np.array(g["heading"]) * -1
    g["headingVis"][g["headingVis"] < 0] += 360
    g["bboxVis"] = viz.calculate_rotated_bboxes(
        g["xCenterVis"], g["yCenterVis"],
        g["lengthVis"], g["widthVis"],
        np.deg2rad(g["headingVis"])
    )

    # M bounding boxes
    bbox = g["bboxVis"]
    centerpt = g["centerVis"]
    bboxes += [bbox]
    centerpts += [centerpt]
    frames += [g["frame"]]
    xmin, xmax = min(xmin, np.min(bbox[:, :, 0])), max(xmax, np.max(bbox[:, :, 0]))
    ymin, ymax = min(ymin, np.min(bbox[:, :, 1])), max(ymax, np.max(bbox[:, :, 1]))

# normalize
for i in range(len(bboxes)):
    bboxes[i][:, :, 0] = (bboxes[i][:, :, 0]-xmin) / (xmax-xmin) * 1000.
    bboxes[i][:, :, 1] = (bboxes[i][:, :, 1]-ymin) / (ymax-ymin) * 1000.
    centerpts[i][:, 0] = (centerpts[i][:, 0]-xmin) / (xmax-xmin) * 1000.
    centerpts[i][:, 1] = (centerpts[i][:, 1]-ymin) / (ymax-ymin) * 1000.

# See if there is a constraints.pickle
try:
    import pickle, os
    if not os.path.exists("pickles"):
        os.mkdir("pickles")
    with open('pickles/constraints.pickle', 'rb') as handle:
        constraints = pickle.load(handle)
except:
    print("\nNo constraints.pickle! Simulation rendering will not show constraints")


class DiscreteGrid(viz.Group):
    def __init__(self, x, y, w, h, arr):
        self.arr = arr
        self.itemsarr = np.array([[None for j in range(arr.shape[1])] for i in range(arr.shape[0])])
        self.allpts = [[None for j in range(arr.shape[1])] for i in range(arr.shape[0])]
        self.xsize, self.ysize = w/arr.shape[0], h/arr.shape[1]
        self.colors = {0:(0,0,0,0.5), 1:(1,0,0,0.5), 2:(0,1,0,0.5), 3:(0,0,1,0.5)}
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                pts = [[x+i*self.xsize+self.xsize/10, y+j*self.ysize+self.ysize/10],
                       [x+(i+1)*self.xsize-self.xsize/10, y+j*self.ysize+self.ysize/10],
                       [x+(i+1)*self.xsize-self.xsize/10, y+(j+1)*self.ysize-self.ysize/10],
                       [x+i*self.xsize+self.xsize/10, y+(j+1)*self.ysize-self.ysize/10]]
                self.allpts[i][j] = pts
                self.itemsarr[i][j] = viz.Rectangle(pts, color = self.colors[arr[i][j]])
        try:
            for pt in constraints["state"]:
                self.itemsarr[pt%n][pt//n].color = (1, 1, 1, 1)
        except:
            pass
        super().__init__(items = self.itemsarr.flatten().tolist())

# Draw Canvas
canvas = viz.Canvas(1000, 1000, id = "000")
canvas.set_visible(False)
pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
arr = np.zeros((n, n))
canvas.items += [DiscreteGrid(20, 60, 1000-30, 1000-60, arr)]

def localize(x, y, grid):
    for i in range(len(grid.allpts)):
        for j in range(len(grid.allpts[0])):
            pt1, pt2, pt3, pt4 = grid.allpts[i][j]
            x1, x2 = pt1[0] - grid.xsize/10, pt2[0] + grid.xsize/10
            y1, y2 = pt2[1] - grid.ysize/10, pt3[1] + grid.ysize/10
            if x1 <= x <= x2 and y1 <= y <= y2:
                return (i, j)
    return (-1, -1)

def delocalize(pt, grid):
    for i in range(len(grid.allpts)):
        for j in range(len(grid.allpts[0])):
            pt1, pt2, pt3, pt4 = grid.allpts[i][j]
            x1, x2 = pt1[0] - grid.xsize/10, pt2[0] + grid.xsize/10
            y1, y2 = pt2[1] - grid.ysize/10, pt3[1] + grid.ysize/10
            if i+j*n == pt:
                return np.array([(x1+x2)/2, (y1+y2)/2])


import matplotlib.pyplot as plt
import numpy as np

# Combine dataset and demo trajectories for plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the dataset trajectory (center points)
for demo_pts in centerpts:
    demo_pts = np.array(demo_pts)
    ax.plot(demo_pts[:, 0], demo_pts[:, 1], marker='x', color='yellow')

# If constraints exist, plot them as red triangles with x, y values
if constraints and "state" in constraints:
    pt = []
    for state_value in constraints["state"]:
        # print(state_value)
        # # Use state_value as the x-coordinate and idx as the y-coordinate (index in the list)
        pt += [delocalize(state_value, canvas.items[-1])]   # x-coordinate
    pt = np.array(pt)
    ax.plot(pt[:, 0], pt[:, 1], 'rx', markersize=10)

# finish the trajectories plotting by beam search

a_values = [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2]  # Discrete acceleration values
traj_list = []
csv_file = "./inD/00_tracks.csv"  # Replace with the actual CSV file
filtered_transitions = preprocess_and_filter_trajectories(csv_file)
train_loader = load_data(filtered_transitions)
model = train_model(train_loader, epochs=200, lr=0.001)

for i in range(100):
    # CHANGE -> Extract start state for each sample -
    start_state = train_loader.dataset[i][0][:6].numpy()

    # Generate the trajectory using Beam Search
    best_trajectory = beam_search(start_state, model, a_values, max_depth=100)

    # Recalculate xmin and ymin for the specific trajectory
    trajectory_x = [state[0] for state in best_trajectory]  # Extract all x values
    trajectory_y = [state[1] for state in best_trajectory]  # Extract all y values

    xmin_local = min(trajectory_x)
    xmax_local = max(trajectory_x)
    ymin_local = min(trajectory_y)
    ymax_local = max(trajectory_y)

    # Normalize trajectory to demo points format
    normalized_trajectory = []
    for state in best_trajectory:
        x, y, vx, vy, a_x, a_y = state

        # Normalize x and y using the local min and max
        x_normalized = (x - xmin_local) / (xmax_local - xmin_local) * 1000
        y_normalized = (y - ymin_local) / (ymax_local - ymin_local) * 1000

        # Replace original x and y with normalized values
        normalized_state = [x_normalized, y_normalized, vx, vy, a_x, a_y]
        normalized_trajectory.append(normalized_state)

    # Append normalized trajectory to the trajectory list
    traj_list.append(normalized_trajectory)

for i, traj in enumerate(traj_list):
    traj = np.array(traj)  # Convert to numpy array if not already
    ax.plot(traj[:, 0], traj[:, 1], marker='o') #label=f"Trajectory {i+1}

# Customize the plot
ax.set_title("Beamsearch Trajectories")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend()
ax.grid(True)


# Save the combined plot
output_path = "./visualization/trajectory.png"
plt.savefig(output_path, dpi=300)
plt.close()  # Close the plot to free memory
print(f"Visualization of demo and dataset trajectories saved as {output_path}")


import matplotlib.pyplot as plt
import numpy as np

# Limit to 100 trajectories
num_traj = 10
selected_traj = traj_list[:num_traj]

# Prepare data for plotting
all_time = []
all_velocity = []
all_acceleration = []

for traj_idx, traj in enumerate(selected_traj):
    time_steps = np.arange(len(traj))  # Time steps (assume uniform time intervals)
    vx = [state[2] for state in traj]  # Extract vx
    vy = [state[3] for state in traj]  # Extract vy
    ax = [state[4] for state in traj]  # Extract ax
    ay = [state[5] for state in traj]  # Extract ay

    # Combine for plotting
    all_time.append(time_steps)
    all_velocity.append((vx, vy))
    all_acceleration.append((ax, ay))

# Plot velocity and acceleration
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Velocity Plot
for traj_idx, (time, (vx, vy)) in enumerate(zip(all_time, all_velocity)):
    axes[0].plot(time, vx, label=f"Traj {traj_idx+1} - vx", alpha=0.7)
    axes[0].plot(time, vy, label=f"Traj {traj_idx+1} - vy", linestyle="dashed", alpha=0.7)

axes[0].set_title("Velocity Change Over Time")
axes[0].set_ylabel("Velocity (vx, vy)")
axes[0].legend(loc="upper right", fontsize=8, ncol=2)

# Save the velocity plot
velocity_plot_path = "./visualization/velocity_beamsearch.png"
fig.savefig(velocity_plot_path)

# Acceleration Plot
for traj_idx, (time, (ax, ay)) in enumerate(zip(all_time, all_acceleration)):
    axes[1].plot(time, ax, label=f"Traj {traj_idx+1} - ax", alpha=0.7)
    axes[1].plot(time, ay, label=f"Traj {traj_idx+1} - ay", linestyle="dashed", alpha=0.7)

axes[1].set_title("Acceleration Change Over Time")
axes[1].set_xlabel("Time Steps")
axes[1].set_ylabel("Acceleration (ax, ay)")
axes[1].legend(loc="upper right", fontsize=8, ncol=2)

# Save the acceleration plot
acceleration_plot_path = "./visualization/acceleration_beamsearch.png"
fig.savefig(acceleration_plot_path)

plt.tight_layout()
plt.show()

