import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Generate some data
np.random.seed(0)
data = np.random.rand(10, 10) * 20

# Create a colormap
colormap = plt.cm.get_cmap('viridis')  # or any other colormap that you like

# Create a normalization object
norm = colors.Normalize(vmin=np.min(data), vmax=np.max(data))

# Plot the data
plt.imshow(data, cmap=colormap, norm=norm)
plt.colorbar(label='Signal Strength')
plt.title('Colored Blocks corresponding to Signal Strength')
plt.show()