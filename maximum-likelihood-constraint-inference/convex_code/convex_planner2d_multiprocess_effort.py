import cvxpy as cp
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from typing import Optional, Tuple
import matplotlib.patches as patches
import os
import matplotlib
matplotlib.use('Agg')       
from pathlib import Path
from functools import partial

import multiprocessing as mp




Vec2 = tuple[float, float]

def minimise_effort_scp(
    s_init: Vec2,
    v_init: Vec2,
    a_init: Vec2,
    s_front_traj: np.ndarray,     
    v_max: float,
    a_max: float,
    d_min: float,
    #s_goal_box: tuple[float, float, float, float], 
    s_goal_point,
    T_min: int,
    T_max: int,
    Δ_t: float,
    static_obs: list[tuple[Vec2, float]] | None = None,
    scp_iterations: int = 3 # <<< ADDED: Number of SCP iterations
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray] | None:
    
    static_obs = static_obs or []
    sx_init, sy_init = s_init
    vx_init, vy_init = v_init
    ax_init, ay_init = a_init
    # x_goal_lo, y_goal_lo, x_goal_hi, y_goal_hi = s_goal_box
    T=T_max
    pad_length = T_max - len(s_front_traj)
    if pad_length > 0:
        s_front_traj = np.pad(s_front_traj, ((0, pad_length), (0, 0)), mode='edge')


    best_solution = None

    # --- SCP Iteration Loop ---
    s_ref_traj = None 


    for i in range(scp_iterations):
        print(f"    SCP Iteration {i+1}/{scp_iterations}...")

        sx, sy = cp.Variable(T), cp.Variable(T)
        vx, vy = cp.Variable(T), cp.Variable(T)
        ax, ay = cp.Variable(T), cp.Variable(T)
        
        constr = []
        constr += [
            sx[0] == sx_init, sy[0] == sy_init,
            vx[0] == vx_init, vy[0] == vy_init,
            ax[0] == ax_init, ay[0] == ay_init,
        ]

        for t in range(T - 1):
            constr += [
                sx[t + 1] == sx[t] + vx[t] * Δ_t,
                sy[t + 1] == sy[t] + vy[t] * Δ_t,
                vx[t + 1] == vx[t] + ax[t] * Δ_t,
                vy[t + 1] == vy[t] + ay[t] * Δ_t,
                cp.norm(cp.hstack([vx[t + 1], vy[t + 1]])) <= v_max,
                cp.norm(cp.hstack([ax[t + 1], ay[t + 1]])) <= a_max,
            ]

            # --- COLLISION AVOIDANCE (The key change is here) ---
            # On the first iteration, s_ref_traj is None, so we don't add the constraint.
            # On subsequent iterations, we use the previous solution as a smart reference.
            if s_ref_traj is not None:
                s_ego_ref = s_ref_traj[t]
                s_front_t = s_front_traj[t]
                delta_s = s_ego_ref - s_front_t
                if np.linalg.norm(delta_s) > 1e-6:
                    normal_vec = delta_s / np.linalg.norm(delta_s)
                    constr += [normal_vec @ cp.hstack([sx[t] - s_front_t[0], sy[t] - s_front_t[1]]) >= d_min]
                    
            
            # (Static obstacle logic would go here
        t_final = T - 1
        if s_ref_traj is not None:
            s_ego_ref = s_ref_traj[t_final]
            s_front_t = s_front_traj[t_final] # Use the lead car's position at the final step
            delta_s = s_ego_ref - s_front_t

            if np.linalg.norm(delta_s) > 1e-6:
                normal_vec = delta_s / np.linalg.norm(delta_s)
                constr += [normal_vec @ cp.hstack([sx[t_final] - s_front_t[0], sy[t_final] - s_front_t[1]]) >= d_min]




        constr += [
            # sx[T - 1] >= x_goal_lo, sx[T - 1] <= x_goal_hi,
            # sy[T - 1] >= y_goal_lo, sy[T - 1] <= y_goal_hi,
            sx[T - 1] == s_goal_point[0],
            sy[T - 1] == s_goal_point[1] 
        ]

        objective = cp.Minimize(cp.sum_squares(ax) + cp.sum_squares(ay))
        prob = cp.Problem(objective, constr)
        prob.solve() 

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # Update the reference trajectory with the new solution
            s_ref_traj = np.vstack((sx.value, sy.value)).T
            # If this is the last iteration, we have our final solution for this T
 
            s_val = s_ref_traj
            v_val = np.vstack((vx.value, vy.value)).T
            a_val = np.vstack((ax.value, ay.value)).T
            best_solution = (T, s_val, v_val, a_val)
        else:
            # If any iteration fails, this T is infeasible. Break the SCP loop.
            is_feasible_for_T = False
            break
    # --- End of SCP Loop ---
        

    if best_solution is None:
        print("  -> No feasible trajectory found within the given time horizon.")
        return None
    
    return best_solution



def get_track_trajectory_2d(
    csv_file: str, 
    track_id: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extracts 2D position, velocity, and acceleration for a specific track_id.

    Args:
        csv_file (str): Path to the tracks CSV file.
        track_id (int): ID of the track to extract.

    Returns:
        A tuple containing three (N, 2) NumPy arrays for the trajectory:
        (s_trajectory, v_trajectory, a_trajectory)
        - s_trajectory: Position (x, y) over time.
        - v_trajectory: Velocity (vx, vy) over time.
        - a_trajectory: Acceleration (ax, ay) over time.
        Returns None if the track_id is not found in the file.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found.")
        return None

    if track_id == 104.2:
        track_id = 104
    
    track_data = df[df['trackId'] == track_id].copy()

    # Check if the track was found
    if track_data.empty:
        print(f"Warning: Track with track_id {track_id} not found in the CSV file.")
        return None
        
    track_data.sort_values('frame', inplace=True)
    

    s_trajectory = track_data[['xCenter', 'yCenter']].values
    
    v_trajectory = track_data[['xVelocity', 'yVelocity']].values
    
    a_trajectory = track_data[['xAcceleration', 'yAcceleration']].values

    return s_trajectory, v_trajectory, a_trajectory

def to_len(arr: np.ndarray, L: int) -> np.ndarray:
    """Return arr with length L along axis‑0 (repeat last row if needed)."""
    n = len(arr)
    if n == L:
        return arr
    if n > L:
        return arr[:L]
    pad = np.repeat(arr[-1:], L - n, axis=0)
    return np.vstack([arr, pad])

def test_planner_with_csv_2d(
    track,
    track_num,
    objective: str = 'effort',
    car_follow_csv: str = "Validated_Car_Following_Pairs_00.csv", 
):
    """
    Tests the 2D planner (minimise_effort_2d or minimise_effort_2d) using real vehicle data.
    """

    try:
        car_follow_df = pd.read_csv(car_follow_csv)
        track_df = pd.read_csv(track)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return


    v_mag = np.linalg.norm(track_df[['xVelocity', 'yVelocity']].values, axis=1)
    a_mag = np.linalg.norm(track_df[['xAcceleration', 'yAcceleration']].values, axis=1)
    

    V_MAX = v_mag.max()
    A_MAX = a_mag.max()
    print(f"Global limits from dataset: v_max = {V_MAX:.2f} m/s, a_max = {A_MAX:.2f} m/s^2")


    D_MIN = 10.0      
    Δt = 0.04        
    
    all_planned_trajectories = pd.DataFrame()
    
    # --- 2. Loop Through Each Car-Following Scenario ---
    unique_front_ids = car_follow_df['front_trackId'].unique()
    unique_ego_trackId = car_follow_df['ego_trackId'].unique()
    
    for front_track_id,unique_ego_trackId in zip(unique_front_ids, unique_ego_trackId):
        print(f"\n{'='*20} Planning for Front Track ID: {front_track_id} and csv number {track_num}{'='*20}")

        scenario_data = car_follow_df[car_follow_df['front_trackId'] == front_track_id].iloc[0]

        # --- 3. Lead Car Setup ---
        lead_car_traj = get_track_trajectory_2d(track, front_track_id)
        if lead_car_traj is None:
            print(f"Could not retrieve trajectory for lead car {front_track_id}. Skipping.")
            continue
        s_front_traj, v_front_traj, a_front_traj = lead_car_traj
        T_horizon = len(s_front_traj)

        # --- 3.5 real car data  ---
        real_car_traj = get_track_trajectory_2d(track, unique_ego_trackId)
        if real_car_traj is None:
            print(f"Could not retrieve trajectory for lead car {unique_ego_trackId}. Skipping.")
            continue
        s_real_traj, v_real_traj, a_real_traj = real_car_traj
        T_real_horizon = len(s_real_traj)



        # --- 4. Ego Car Setup ---
        try:
            s_init =s_real_traj[0] #(scenario_data['ego_xCenter'], scenario_data['ego_yCenter'])
            v_init =v_real_traj[0] # (scenario_data['ego_xVelocity'], scenario_data['ego_yVelocity'])
            a_init =a_real_traj[0] #(scenario_data['ego_xAcceleration'], scenario_data['ego_yAcceleration'])
        except KeyError as e:
            print(f"FATAL: Missing required ego vehicle column in {car_follow_csv}: {e}")
            print("Please ensure your CSV contains the initial 2D state of the ego vehicle.")
            return

        # --- 5. Goal Region Setup (2D Box) ---
        s_front_final = s_front_traj[-1]
        
        # Define goal center 15m behind the lead car's final position, with a 5m width/height
        
        
        
        # i=1
        # while np.linalg.norm(s_front_final - s_real_traj[-i]) < D_MIN+1:
        #     i += 1
        #     if i >= len(s_real_traj):
        #         print(f"Warning: Could not find a suitable position for the goal box behind lead car {front_track_id}. Using last position.")
        #         break

        goal_center = s_real_traj[-1]

        
        if np.linalg.norm(s_front_final - goal_center) < D_MIN or np.linalg.norm(np.array(s_init) - s_front_traj[0])< D_MIN :
            print(f"Warning: Initial or goal position too close to lead car {front_track_id}. Skipping this scenario.")
            trajectory_data = pd.DataFrame({
                'ego_track_id': [unique_ego_trackId] ,
                'front_track_id': [front_track_id] ,
                'time_step': [-1],
                'time_sec':[-1],
                'ego_vel_x':[-1], 'ego_vel_y':[-1],
                'ego_pos_x':[-1], 'ego_pos_y':[-1],
                'ego_acc_x':[-1], 'ego_acc_y':[-1],
                'lead_pos_x': [-1], 'lead_pos_y': [-1],
                'lead_vel_x': [-1], 'lead_vel_y': [-1],
                'lead_acc_x': [-1], 'lead_acc_y': [-1],
                'real_pos_x': [-1], 'real_pos_y': [-1],  
                'Orignal_goal_point_x':[-1], 'Orignal_goal_point_y': [-1],
            })
            all_planned_trajectories = pd.concat([all_planned_trajectories, trajectory_data], ignore_index=True)
            continue


        # behind_real =0
        goal_box_size = 0.1
        s_goal_box = (
            goal_center[0] - goal_box_size / 2, goal_center[1] - goal_box_size / 2,
            goal_center[0] + goal_box_size / 2, goal_center[1] + goal_box_size / 2
        )
        
        #intial distance
        # s_front_start=s_front_traj[0]
        # j=0
        # while np.linalg.norm(np.array(s_init) - s_front_traj[0])<D_MIN+1:
        #     s_front_traj=s_front_traj[j:]
        #     v_front_traj=v_front_traj[j:]
        #     a_front_traj=a_front_traj[j:]
        #     j+=1
        #     if j>=len(s_front_traj):
        #         print(f"Warning: Could not find a suitable position for the initial distance behind lead car {front_track_id}. Using last position.")
        #         break
        

        print(f"Initial Ego State: s={s_init}, v={v_init}")
        print(f"goal point: {goal_center}")
        print(f"Goal Box: x=[{s_goal_box[0]:.1f}, {s_goal_box[2]:.1f}], y=[{s_goal_box[1]:.1f}, {s_goal_box[3]:.1f}]")
        
        # --- 6. Run the Planner ---
        try:
            if objective == 'effort':
                print("Objective: Minimizing effort...")
                filename=f"plots/min_effort/trackcsv_{track_num}/trajectory_track_{front_track_id}.pdf"
                planner_result = minimise_effort_scp(
                    s_init, v_init, a_init, s_front_traj, V_MAX, A_MAX, D_MIN,
                    goal_center, T_min=1, T_max=min(T_real_horizon,2000), Δ_t=Δt, static_obs=None
                )
                if planner_result:
                    T_star, s_traj, v_traj, a_traj = planner_result
                else:
                    T_star = None


            else:
                raise ValueError(f"Unknown objective '{objective}'")

            if not T_star:
                print("Planner failed to find a solution for this scenario.")
                continue

            print(f"Feasible trajectory found in T = {T_star} steps ({T_star*Δt:.2f} seconds).")

            # --- 7. Store and Plot Results ---
            # Store data
            s_front_traj = to_len(s_front_traj, T_star)
            v_front_traj = to_len(v_front_traj, T_star)
            a_front_traj = to_len(a_front_traj, T_star)

            s_real_traj  = to_len(s_real_traj,  T_star)
            v_real_traj  = to_len(v_real_traj,  T_star)
            a_real_traj  = to_len(a_real_traj,  T_star)


            trajectory_data = pd.DataFrame({
                'ego_track_id': [unique_ego_trackId] * T_star,
                'front_track_id': [front_track_id] * T_star,
                'time_step': range(T_star),
                'time_sec': np.arange(T_star) * Δt,
                'ego_pos_x': s_traj[:, 0], 'ego_pos_y': s_traj[:, 1],
                'ego_vel_x': v_traj[:, 0], 'ego_vel_y': v_traj[:, 1],
                'ego_acc_x': a_traj[:, 0], 'ego_acc_y': a_traj[:, 1],
                'lead_pos_x': s_front_traj[:, 0], 'lead_pos_y': s_front_traj[:, 1],
                'lead_vel_x': v_front_traj[:, 0], 'lead_vel_y': v_front_traj[:, 1],
                'lead_acc_x': a_front_traj[:, 0], 'lead_acc_y': a_front_traj[:, 1],
                'real_pos_x': s_real_traj[:, 0],  'real_pos_y': s_real_traj[:, 1],
                'Orignal_goal_point_x': [s_real_traj[-1,0]]* T_star, 'Orignal_goal_point_y': [s_real_traj[-1,1]]* T_star,

            })
            all_planned_trajectories = pd.concat([all_planned_trajectories, trajectory_data], ignore_index=True)


            # --- Plotting ---
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(3, 2)
            
            # Main 2D Trajectory Plot
            ax0 = fig.add_subplot(gs[:, 0])
            ax0.plot(s_front_traj[:, 0], s_front_traj[:, 1], 'x--', color='orange', label=f'Lead Car (ID {front_track_id})')
            ax0.plot(s_real_traj[:, 0], s_real_traj[:, 1], 'x--', color='red', label=f'real Car (ID {unique_ego_trackId})')
            ax0.plot(s_traj[:, 0], s_traj[:, 1], 'o-', color='blue', markersize=2, label='Planned Ego Trajectory')
            # goal_patch = patches.Rectangle(
            #     (s_goal_box[0], s_goal_box[1]), s_goal_box[2]-s_goal_box[0], s_goal_box[3]-s_goal_box[1],
            #     linewidth=1, edgecolor='g', facecolor='g', alpha=0.3, label='Goal Region'
            # )
            # ax0.add_patch(goal_patch)
            ax0.plot(goal_center[0], goal_center[1], 'go', markersize=15, label='Goal Center', alpha=0.3)
            ax0.plot(s_init[0], s_init[1], 'go', markersize=10, label='Ego Start')
            ax0.plot(s_front_traj[0,0], s_front_traj[0,1], 'ro', markersize=10, label='Lead Start')
            ax0.plot(s_real_traj[0,0], s_real_traj[0,1], 'purple', markersize=10, label='real Start')

            ax0.set_xlabel("x position (m)")
            ax0.set_ylabel("y position (m)")
            ax0.set_title(f"2D Trajectory - Objective: {objective.capitalize()}")
            ax0.legend()
            ax0.grid(True)
            ax0.axis('equal')

            # Velocity and Acceleration Magnitude Plots
            time_axis = np.arange(T_star) * Δt
            # v_mag_planned = np.linalg.norm(v_traj, axis=1)
            # a_mag_planned = np.linalg.norm(a_traj, axis=1)
            # v_mag_real = np.linalg.norm(v_real_traj, axis=1)
            # a_mag_real = np.linalg.norm(a_real_traj, axis=1)
            # v_mag_lead = np.linalg.norm(v_front_traj, axis=1)
            # a_mag_lead = np.linalg.norm(a_front_traj, axis=1)



            ax1 = fig.add_subplot(gs[0, 1])
            # Planned Ego
            ax1.plot(time_axis, v_traj[:, 0], label='Planned vx', color='blue', linestyle='-')
            ax1.plot(time_axis, v_traj[:, 1], label='Planned vy', color='blue', linestyle='--')
            # Real Ego
            ax1.plot(time_axis, v_real_traj[:T_star, 0], label='Real vx', color='magenta', linestyle='-')
            ax1.plot(time_axis, v_real_traj[:T_star, 1], label='Real vy', color='magenta', linestyle='--')
            # Lead Car
            ax1.plot(time_axis, v_front_traj[:T_star, 0], label='Lead vx', color='orange', linestyle=':')
            ax1.plot(time_axis, v_front_traj[:T_star, 1], label='Lead vy', color='orange', linestyle='-.')
            # Limits
            ax1.axhline(y=V_MAX, color='r', linestyle='--', alpha=0.7, label=f'Limit (+/-{V_MAX:.1f})')
            ax1.axhline(y=-V_MAX, color='r', linestyle='--', alpha=0.7)
            
            ax1.set_ylabel('Velocity (m/s)')
            ax1.set_title('Velocity Components vs. Time')
            ax1.grid(True)
            ax1.legend(fontsize='small')

            # --- Acceleration Components Plot (ax2) ---
            ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
            # Planned Ego
            ax2.plot(time_axis, a_traj[:, 0], label='Planned ax', color='blue', linestyle='-')
            ax2.plot(time_axis, a_traj[:, 1], label='Planned ay', color='blue', linestyle='--')
            # Real Ego
            ax2.plot(time_axis, a_real_traj[:T_star, 0], label='Real ax', color='magenta', linestyle='-')
            ax2.plot(time_axis, a_real_traj[:T_star, 1], label='Real ay', color='magenta', linestyle='--')
            # Lead Car
            ax2.plot(time_axis, a_front_traj[:T_star, 0], label='Lead ax', color='orange', linestyle='-')
            ax2.plot(time_axis, a_front_traj[:T_star, 1], label='Lead ay', color='orange', linestyle='--')
            # Limits
            ax2.axhline(y=A_MAX, color='r', linestyle='--', alpha=0.7, label=f'Limit (+/-{A_MAX:.1f})')
            ax2.axhline(y=-A_MAX, color='r', linestyle='--', alpha=0.7)

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Acceleration (m/s²)')
            ax2.set_title('Acceleration Components vs. Time')
            ax2.grid(True)
            ax2.legend(fontsize='small')

            dist_to_goal_planned = np.linalg.norm(s_traj - goal_center, axis=1)
            dist_to_goal_real = np.linalg.norm(s_real_traj[:T_star] - goal_center, axis=1)
            dist_to_goal_lead = np.linalg.norm(s_front_traj[:T_star] - goal_center, axis=1)

            # Create the third subplot on the right
            ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
            ax3.plot(time_axis, dist_to_goal_planned, color='blue', label='Planned Traj.')
            ax3.plot(time_axis, dist_to_goal_real, color='magenta', linestyle='--', label='Real Traj.')
            ax3.plot(time_axis, dist_to_goal_lead, color='orange', linestyle='--', label='lead Traj.')
            
            ax3.set_title('Distance to Goal Point vs. Time')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Distance to Goal (m)')
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            plt.savefig(filename)
            plt.close(fig)
            print(f"Saved plot to {filename}")

        except Exception as e:
            print(f"An unexpected error occurred during planning for track {front_track_id}: {e}")

    output_csv_path = f"effort_data/track_{track_num}/all_trajectories.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    all_planned_trajectories.to_csv(output_csv_path, index=False)
    print(f"\nSaved all planned trajectories to {output_csv_path}")


def process_track(args):
    """Helper function to unpack arguments for parallel processing"""
    track_file, track_num, objective, car_follow_csv = args
    try:
        return test_planner_with_csv_2d(
            track=track_file,
            track_num=track_num,
            objective=objective,
            car_follow_csv=car_follow_csv
        )
    except Exception as e:
        print(f"Error processing track {track_num}: {e}")
        return None

def test_planner_with_csv_2d_parallel(
    objective: str = 'effort',
    car_follow_csv: str = "Validated_Car_Following_Pairs_00.csv",
    n_processes: int = None
):
    """Parallel version of the test planner."""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1  # Leave one CPU free
    
    # Get list of track files

    track_pairs = [
        (f"inD-dataset-v1.0/data/{i:02d}_tracks.csv", i)
        for i in range(33)
    ]
    # Prepare arguments for parallel processing
    process_args = [
        (track_file, track_num, objective, car_follow_csv)
        for track_file, track_num in track_pairs
    ]
    
    # Create pool and map function to tracks
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_track, process_args)
    
    # Count successful results
    successful_results = [r for r in results if r is not None]
    print(f"Completed processing {len(successful_results)} tracks successfully")
    
    return results

if __name__ == '__main__':
    matplotlib.use('Agg')
    test_planner_with_csv_2d_parallel(
        objective='effort',
        car_follow_csv="Validated_Car_Following_Pairs_00.csv",
        n_processes=None  # Use default (CPU count - 1)
    )
