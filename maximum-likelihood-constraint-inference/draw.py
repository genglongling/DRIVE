import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the main nodes and layers
nodes = [
    {"name": "Input Layer (12 units)", "pos": (0, 2), "stack_colors": ['red']},
    {"name": "Hidden Layer 1 (64 units)", "pos": (2, 2.5), "stack_colors": ['blue', 'lightblue']},
    {"name": "Hidden Layer 2 (64 units)", "pos": (4, 2.5), "stack_colors": ['green', 'lightgreen']},
    {"name": "Output Layer (1 unit)", "pos": (6, 2), "stack_colors": ['purple']},
    {"name": "Reward", "pos": (8, 2), "stack_colors": ['orange']},
    {"name": "...", "pos": (10, 2.5), "is_placeholder": True},
    {"name": "Selected Trajectory", "pos": (12, 2), "stack_colors": ['gold']},
]

# Define beam node positions separately for clarity
beam_nodes = [
    {"name": f"Beam Node {i+1}", "pos": (10, 1.5 + i * 0.6)} for i in range(3)
]

edges = [
    ("Input Layer (12 units)", "Hidden Layer 1 (64 units)"),
    ("Hidden Layer 1 (64 units)", "Hidden Layer 2 (64 units)"),
    ("Hidden Layer 2 (64 units)", "Output Layer (1 unit)"),
    ("Output Layer (1 unit)", "Reward"),
    ("Reward", "Beam Node 1"),
    ("Reward", "Beam Node 2"),
    ("Reward", "Beam Node 3"),
    ("Reward", "..."),
    ("Beam Node 1", "Selected Trajectory"),
    ("Beam Node 2", "Selected Trajectory"),
    ("Beam Node 3", "Selected Trajectory"),
]

# Create the figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
ax.set_xlim(-1, 13)
ax.set_ylim(0, 5)

# Draw edges (arrows)
for edge in edges:
    start_node = next(node for node in nodes + beam_nodes if node["name"] == edge[0])
    end_node = next(node for node in nodes + beam_nodes if node["name"] == edge[1])
    ax.annotate(
        '', xy=end_node["pos"], xytext=start_node["pos"],
        arrowprops=dict(arrowstyle='->', color='gray', lw=1.5)  # Smaller arrows
    )

# Draw nodes as text with stack of rectangles
for node in nodes:
    x, y = node["pos"]
    if node.get("is_placeholder"):  # Add placeholder for ellipsis
        ax.text(
            x, y, node["name"], fontsize=16, ha='center', va='center', color='black',
            bbox=dict(facecolor='none', edgecolor='none')
        )
    else:  # Draw normal node
        # Draw the stack of rectangles for the node
        stack_width = 0.6  # Increased width
        stack_height = 0.2  # Increased height
        for i, color in enumerate(node.get("stack_colors", [])):
            rect = patches.Rectangle(
                (x - stack_width / 2, y - stack_height * len(node["stack_colors"]) / 2 + i * stack_height),
                stack_width, stack_height, facecolor=color, edgecolor='black'
            )
            ax.add_patch(rect)

        # Add the node label
        ax.text(
            x, y + 0.3, node["name"], fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )

# Draw beam nodes as circles
for beam_node in beam_nodes:
    x, y = beam_node["pos"]
    circle = patches.Circle((x, y), radius=0.4, color='skyblue', ec='black', lw=1.5)  # Larger circles
    ax.add_patch(circle)
    ax.text(x, y, beam_node["name"], fontsize=10, ha='center', va='center')

# Add a rectangular region for the neural network
ax.add_patch(
    patches.Rectangle(
        (-0.5, 1), 7, 2.5, linewidth=2.5, edgecolor='blue', facecolor='none', linestyle='--'
    )
)
ax.text(3, 4, "Neural Network Region", fontsize=12, ha='center', va='center', color='blue')

# Add title
plt.title("Reinforcement Learning with Beam Search, Maximum Beam Nodes = 9x9 (81)", fontsize=16)
plt.savefig("./visualization/beamsearch_structure.png")
# Show the plot
plt.show()
