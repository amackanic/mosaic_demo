import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QStatusBar
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d, gaussian_filter

# --- Simulation parameters ---
vessel_length = 10.0  # cm
num_points = 200
sine_amplitude = 0.5
sine_frequency = 1.5
z_amplitude = 0.5
z_frequency = 1.0
radius_mean = 0.2
radius_std = 0.05
radius_min = 0.08
radius_max = 0.35
frames = 200

# Vessel centerline
x = np.linspace(0, vessel_length, num_points)
y = sine_amplitude * np.sin(2 * np.pi * sine_frequency * x / vessel_length)
z = z_amplitude * np.sin(2 * np.pi * z_frequency * x / vessel_length)

# Random but smooth vessel radius
np.random.seed(42)
rand_r = np.random.normal(0, 1, num_points)
smooth_r = gaussian_filter1d(rand_r, sigma=15)
smooth_r = (smooth_r - smooth_r.min()) / (smooth_r.max() - smooth_r.min())
radius = radius_mean + radius_std * (2 * smooth_r - 1)
radius = np.clip(radius, radius_min, radius_max)

# Tangent and normal for 3D tube
from numpy.linalg import norm
dx = np.gradient(x)
dy = np.gradient(y)
dz = np.gradient(z)
tangent = np.stack([dx, dy, dz], axis=1)
tangent /= norm(tangent, axis=1)[:, None]
arbitrary = np.tile([0, 0, 1], (num_points, 1))
close_to_z = np.abs(tangent @ np.array([0, 0, 1])) > 0.99
arbitrary[close_to_z] = [0, 1, 0]
normal = np.cross(tangent, arbitrary)
normal /= norm(normal, axis=1)[:, None]
binormal = np.cross(tangent, normal)

# Tube surface for vessel
n_tube = 20
theta = np.linspace(0, 2 * np.pi, n_tube)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
X = np.zeros((num_points, n_tube))
Y = np.zeros((num_points, n_tube))
Z = np.zeros((num_points, n_tube))
for i in range(num_points):
    for j in range(n_tube):
        X[i, j] = x[i] + radius[i] * (normal[i, 0] * circle[j, 0] + binormal[i, 0] * circle[j, 1])
        Y[i, j] = y[i] + radius[i] * (normal[i, 1] * circle[j, 0] + binormal[i, 1] * circle[j, 1])
        Z[i, j] = z[i] + radius[i] * (normal[i, 2] * circle[j, 0] + binormal[i, 2] * circle[j, 1])

def tissue_texture(img_size, seed=0):
    np.random.seed(seed)
    base = np.random.normal(0.5, 0.1, (img_size, img_size))
    smooth = gaussian_filter(base, sigma=8)
    y, x = np.ogrid[:img_size, :img_size]
    streaks = 0.1 * np.sin(0.1 * x + 0.2 * y)
    tissue = 0.7 * smooth + 0.3 * streaks
    tissue = np.clip(tissue, 0, 1)
    rgb = np.zeros((img_size, img_size, 3))
    rgb[..., 0] = 0.8 * tissue + 0.2  # R
    rgb[..., 1] = 0.4 * tissue + 0.3  # G
    rgb[..., 2] = 0.4 * tissue + 0.4  # B
    return rgb

# --- Realistic cross-section (not IVUS) ---
def simulate_cross_section(local_radius, wall_irreg_seed=0):
    img_size = 128
    center = img_size // 2
    n_angles = 128
    np.random.seed(wall_irreg_seed)
    theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    # Irregular lumen and wall
    lumen_r = 0.5 * local_radius * img_size / radius_max
    wall_r = local_radius * img_size / radius_max
    lumen_irreg = 1 + 0.10 * gaussian_filter1d(np.random.normal(0, 1, n_angles), sigma=8)
    wall_irreg = 1 + 0.08 * gaussian_filter1d(np.random.normal(0, 1, n_angles), sigma=8)
    lumen_r_profile = lumen_r * lumen_irreg
    wall_r_profile = wall_r * wall_irreg
    # Create polar grid
    y, xg = np.ogrid[:img_size, :img_size]
    r = np.sqrt((xg - center) ** 2 + (y - center) ** 2)
    phi = np.arctan2(y - center, xg - center)
    phi_idx = ((phi + np.pi) / (2 * np.pi) * n_angles).astype(int) % n_angles
    lumen_r_img = lumen_r_profile[phi_idx]
    wall_r_img = wall_r_profile[phi_idx]
    # Draw wall and lumen
    img = np.ones((img_size, img_size, 3))  # RGB
    # Lumen: pinkish
    lumen_mask = r < lumen_r_img
    img[lumen_mask] = [1.0, 0.85, 0.9]
    # Wall: reddish tissue texture
    wall_mask = (r >= lumen_r_img) & (r <= wall_r_img)
    wall_tex = tissue_texture(img_size, seed=wall_irreg_seed)
    img[wall_mask] = wall_tex[wall_mask]
    # Outer: light
    ext_mask = r > wall_r_img
    img[ext_mask] = [1.0, 0.95, 0.95]
    img = gaussian_filter(img, sigma=1.2)
    img = np.clip(img, 0, 1)
    return img

# --- PyQt5 App ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Catheter Simulation UI")
        self.resize(1200, 600)
        central = QWidget()
        layout = QHBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Left: 2D cross-section view
        self.left_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.left_ax = self.left_canvas.figure.subplots()
        self.left_ax.set_title("Cross-Section View", color='k')
        self.left_ax.axis('off')
        layout.addWidget(self.left_canvas, stretch=1)

        # Right: 3D vessel/catheter view
        self.right_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.right_ax = self.right_canvas.figure.add_subplot(111, projection='3d')
        self.right_ax.set_box_aspect([vessel_length, 2, 2])
        self.right_ax.set_xlim(-0.5, vessel_length + 0.5)
        self.right_ax.set_ylim(-2, 2)
        self.right_ax.set_zlim(-2, 2)
        self.right_ax.axis('off')
        layout.addWidget(self.right_canvas, stretch=2)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Catheter Simulation Running...")

        # Animation state
        self.frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_views)
        self.timer.start(40)  # ~25 FPS

    def update_views(self):
        # Advance frame
        self.frame = (self.frame + 1) % frames
        tip_idx = int(self.frame * (num_points - 1) / (frames - 1))
        # --- 2D cross-section view ---
        local_radius = radius[tip_idx]
        cross_img = simulate_cross_section(local_radius, wall_irreg_seed=tip_idx)
        self.left_ax.clear()
        self.left_ax.imshow(cross_img, origin='lower')
        self.left_ax.set_title("Cross-Section View", color='k')
        self.left_ax.axis('off')
        self.left_canvas.draw()
        # --- 3D vessel/catheter view ---
        self.right_ax.clear()
        # 3D tissue texture for vessel wall
        tex_seed = 123
        tex_field = np.random.normal(0.5, 0.1, (num_points, n_tube))
        tex_field = gaussian_filter(tex_field, sigma=6)
        tex_field = np.clip(tex_field, 0, 1)
        facecolors = np.zeros(X[:tip_idx+1, :].shape + (4,))
        facecolors[..., 0] = 0.8 * tex_field[:tip_idx+1, :] + 0.2  # R
        facecolors[..., 1] = 0.4 * tex_field[:tip_idx+1, :] + 0.3  # G
        facecolors[..., 2] = 0.4 * tex_field[:tip_idx+1, :] + 0.4  # B
        facecolors[..., 3] = 0.7  # alpha
        self.right_ax.plot_surface(X[:tip_idx+1, :], Y[:tip_idx+1, :], Z[:tip_idx+1, :], facecolors=facecolors, linewidth=0, antialiased=False, shade=False)
        self.right_ax.plot(x[:tip_idx+1], y[:tip_idx+1], z[:tip_idx+1], 'r-', lw=3)
        self.right_ax.plot([x[tip_idx]], [y[tip_idx]], [z[tip_idx]], 'ro', markersize=8)
        self.right_ax.set_xlim(-0.5, vessel_length + 0.5)
        self.right_ax.set_ylim(-2, 2)
        self.right_ax.set_zlim(-2, 2)
        self.right_ax.axis('off')
        self.right_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 