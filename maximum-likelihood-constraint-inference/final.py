import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
mdp_icl = pd.read_csv('./metrics/violation_rate_history.csv')      # MDP-ICL model
beamsearch = pd.read_csv('./metrics/violation_rate_history2.csv')  # Beamsearch baseline
eflce = pd.read_csv('./metrics/violation_rate_history3.csv')       # EFLCE model

# Option to drop the first few rows (for example, remove the first row)
mdp_icl = mdp_icl.iloc[1:]  # Removes the first row
beamsearch = beamsearch.iloc[1:]  # Removes the first row
eflce = eflce.iloc[1:]  # Removes the first row

# Optionally, you can inspect the data
print(mdp_icl.head())
print(beamsearch.head())
print(eflce.head())

# Plot the data
plt.figure(figsize=(10, 6))

# Plot each model's data
plt.plot(mdp_icl['Epoch'], mdp_icl['Violation Rate History'], label='MDP-ICL', color='blue', linewidth=2)
plt.plot(beamsearch['Epoch'], beamsearch['Violation Rate History'], label='Beamsearch Baseline', color='green', linewidth=2)
plt.plot(eflce['Epoch'], eflce['Violation Rate History'], label='EFLCE', color='red', linewidth=2)

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Violation Rate')
plt.title('Violation Rate for Different Models')

# Add a legend
plt.legend()

# Save the plot to a file
plt.tight_layout()
plt.savefig('./visualization/result_violation_rate.png')

# Optionally, display the plot
plt.show()
