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


### Filter out trajectories
if not args.multi_goal:
    cpts = []
    frms = []
    frm = 0
    for i in range(len(bboxes)):
        p, q = localize(*centerpts[i][-1, :], canvas.items[-1])
        u, v = localize(*centerpts[i][0, :], canvas.items[-1])
        if (p+q*n not in allowed_end_state) or (u+v*n in banned_start_state):
            continue
        else:
            cpts += [centerpts[i]]
            frms += [np.array(frames[i])-frames[i][0]+frm]
            frm += len(frames[i])
    centerpts = cpts
    frames = frms

# if args.show_new_demos:
#     cpts = []
#     frms = []
#     frm = 0
#     try:
#         demos = pickle.load(open("pickles/new_demonstrations.pickle", 'rb'))
#     except:
#         print("Cannot find pickles/new_demonstrations.pickle!")
#         exit(0)
#     for i in range(len(demos)):
#         demo = demos[i].reshape(-1)
#         cpt = []
#         frmvec = []
#         for item in demo:
#             cpt += [delocalize(item, canvas.items[-1])]
#             frmvec += [frm]
#             frm += 1
#         cpts += [np.array(cpt)]
#         frms += [np.array(frmvec)]
#     centerpts = cpts
#     frames = frms

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

if args.show_new_demos:
    cpts = []
    frms = []
    frm = 0
    try:
        demos = pickle.load(open("pickles/nn_demons_0.pickle", 'rb'))
    except FileNotFoundError:
        print("Cannot find pickles/nn_demons_0.pickle!")
        exit(0)

    for i in range(len(demos)):
        demo = demos[i].reshape(-1)
        cpt = []
        frmvec = []
        for item in demo:
            cpt += [delocalize(item, canvas.items[-1])]
            frmvec += [frm]
            frm += 1
        cpts += [np.array(cpt)]
        frms += [np.array(frmvec)]

    centerpts = cpts
    frames = frms

    for demo in demos:
        demo = demo.reshape(-1)
        cpt = []
        for item in demo:
            cpt += [delocalize(item, canvas.items[-1])]
        cpt = np.array(cpt)
        ax.plot(cpt[:, 0], cpt[:, 1], marker='o') #color='blue'

    # Customize the plot
    ax.set_title("EFLCE Trajectories")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True)


### Collect start points
startpts = []
endpts = []
for i in range(len(centerpts)):
    coords = localize(*centerpts[i][0, :], canvas.items[-1])
    if coords != (-1, -1):
        startpts += [coords]
    coords = localize(*centerpts[i][-1, :], canvas.items[-1])
    if coords != (-1, -1):
        endpts += [coords]


# Save the combined plot
output_path = "./visualization/nn_trajectory.png"
plt.savefig(output_path, dpi=300)
plt.close()  # Close the plot to free memory
print(f"Visualization of demo and dataset trajectories saved as {output_path}")


### Visualize
if args.visualize:
    if os.path.exists("frames"):
        import shutil
        shutil.rmtree("frames")
    if not os.path.exists("frames"):
        os.mkdir("frames")
    canvas.set_visible(True)
    tracks = [(None, None) for i in range(len(bboxes))]
    f = 0
    ff = 100
    while True:
        oldarr = np.copy(arr)
        for pq in startpts:
            p, q = pq
            arr[p][q] = 2
            canvas.items[-1].itemsarr[p][q].color = canvas.items[-1].colors[arr[p][q]]
        for i in range(len(centerpts)):
            if frames[i][0] == f:
                tracks[i] = (centerpts[i][0, :], tracks[i][1])
                p, q = localize(*tracks[i][0], canvas.items[-1])
                tracks[i] = (tracks[i][0], (p, q))
                arr[p][q] = 1
                canvas.items[-1].itemsarr[p][q].color = canvas.items[-1].colors[arr[p][q]]
            elif 0 <= f-frames[i][0] < len(frames[i]):
                tracks[i] = (centerpts[i][f-frames[i][0], :], tracks[i][1])
                p, q = localize(*tracks[i][0], canvas.items[-1])
                if tracks[i][1]:
                    pold, qold = tracks[i][1]
                    arr[pold][qold] = 0
                    canvas.items[-1].itemsarr[pold][qold].color = canvas.items[-1].colors[arr[pold][qold]]
                tracks[i] = (tracks[i][0], (p, q))
                arr[p][q] = 1
                canvas.items[-1].itemsarr[p][q].color = canvas.items[-1].colors[arr[p][q]]
            else:
                if tracks[i][1]:
                    pold, qold = tracks[i][1]
                    arr[pold][qold] = 0
                    canvas.items[-1].itemsarr[pold][qold].color = canvas.items[-1].colors[arr[pold][qold]]
                for pq in endpts:
                    p, q = pq
                    arr[p][q] = 3
                    canvas.items[-1].itemsarr[p][q].color = canvas.items[-1].colors[arr[p][q]]

        try:
            for pt in constraints["state"]:
                canvas.items[-1].itemsarr[pt%n][pt//n].color = (1, 1, 1, 1)
        except:
            pass

        f += 1
        canvas.clear()
        canvas.switch_to()
        canvas.dispatch_events()
        canvas.on_draw()
        canvas.flip()
        newarr = np.copy(arr)
        if (newarr != oldarr).any():
            pyglet.image.get_buffer_manager().get_color_buffer().save("frames/%d.png" % ff)
            ff += 1

### Create gifs
if args.create_gifs:
    try:
        print("Converting to gifs ...")
        if os.path.exists("frames"):
            filename = "frames"
            if args.show_new_demos:
                filename = "demos"
            os.system("convert -delay 40 -loop 1 frames/*.png %s.gif && rm -rf frames" % filename)
            os.system("gifsicle -i %s.gif --optimize=3 -o %s.gif" % (filename, filename))
        if os.path.exists("figures"):
            os.system("convert -delay 40 -loop 1 figures/*.png policy.gif && rm -rf figures")
            os.system("gifsicle -i policy.gif --optimize=3 -o policy.gif")
        print("done")
    except:
        print("Check that imagemagick and gifsicle are present!")
