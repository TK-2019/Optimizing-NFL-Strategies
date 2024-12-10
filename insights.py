df = main_df4[['receiver x', 'receiver y', 'pass_outcome']]  

# Separate successful plays
successful_plays = df[df['pass_outcome'] == 1]

# Load the NFL field image
field_img = mpimg.imread('/content/istockphoto-1406263895-612x612.jpg')  # Replace with the path to your NFL field image

plt.figure(figsize=(12, 6))

# Display the field image
plt.imshow(field_img, extent=[0, 120, 0, 53.3], aspect='auto')

# Overlay KDE plot for successful plays
sns.kdeplot(
    x=successful_plays['receiver x'],
    y=successful_plays['receiver y'],
    cmap="Reds",
    shade=True,
    alpha=0.7,
    bw_adjust=1
)

# Adjust plot limits to match the NFL field dimensions
plt.xlim(0, 120)  # Length of the field in yards (end zone to end zone)
plt.ylim(0, 53.3)  # Width of the field in yards

plt.xlabel("Field Length (yards)")
plt.ylabel("Field Width (yards)")
plt.title("Heatmap of Successful Passes on NFL Field")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Create the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x="pass_outcome", y="catchSeparation", data=main_df4, palette="coolwarm")

# Add a background grid
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Increase text size
plt.title("Catch Separation Distribution by Pass Outcome", fontsize=16)
plt.xlabel("Pass Outcome", fontsize=20)
plt.ylabel("Catch Separation", fontsize=20)

# Increase tick label size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Example categories for the radar chart
categories = ['SHOTGUN', 'EMPTY', 'I_FORM', 'SINGLEBACK', 'PISTOL', 'JUMBO', 'WILDCAT']
values = [0.38707037643207853, 0.4, 0.38, 0.4556291390728477, 0.4315068493150685, 0.75, 0.375]  

# Close the plot by appending the first value to the end
values += values[:1]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Set the radar chart scale to 1
ax.set_ylim(0, 1)

# Plot the radar chart
ax.fill(angles, values, color='red', alpha=0.7)
ax.plot(angles, values, color='red', linewidth=2)

# Remove radial ticks
ax.set_yticks([])

# Set the category labels and adjust for better visibility
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, rotation=45, ha='center', fontsize=10, color='black')

# Adding values on the plot for each category
for angle, value, label in zip(angles, values, categories):
    ax.text(angle, value + 0.05, f"{value:.2f}", ha='center', va='center', fontsize=10, color='black')

# Set title and display the plot
ax.set_title("Team Performance by Position", size=14)
plt.show()

