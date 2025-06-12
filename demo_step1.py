import matplotlib
matplotlib.use('MacOSX')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# Parameters
vessel_length = 10.0  # cm
num_points = 200  # reduce for performance
sine_amplitude = 0.5  # cm
sine_frequency = 1.5  # number of sine cycles over vessel length
z_amplitude = 0.5  # cm
z_frequency = 1.0  # number of z cycles over vessel length
radius_mean = 0.25  # cm
radius_std = 0.05  # cm
radius_min = 0.08  # cm
radius_max = 0.35  # cm
tube_points = 20  # points around the tube
reveal_ahead = .50  # how far ahead of the catheter tip to reveal the vessel (in cm)

# Branch (bifurcation) parameters
branch_fraction = 0.3  # 30% along the main vessel
branch_angle = np.deg2rad(30)  # 30 degrees
branch_length = 1 # cm
branch_radius = 0.1  # smaller than main vessel
branch_points = 40
branch_reveal_ahead = 0.25  # how much of the branch to reveal ahead of the bifurcation

# 3D Centerline (simple sine wave in y and z)
x = np.linspace(0, vessel_length, num_points)
y = sine_amplitude * np.sin(2 * np.pi * sine_frequency * x / vessel_length)
z = z_amplitude * np.sin(2 * np.pi * z_frequency * x / vessel_length)

# Original radius code
np.random.seed(42)
rand_r = np.random.normal(0, 1, num_points)
smooth_r = gaussian_filter1d(rand_r, sigma=15)
smooth_r = (smooth_r - smooth_r.min()) / (smooth_r.max() - smooth_r.min())  # normalize to [0,1]
radius = radius_mean + radius_std * (2 * smooth_r - 1)  # mean ± std
radius = np.clip(radius, radius_min, radius_max)

# Simulate plaque at the end (last 10% of the vessel)
plaque_start = int(0.9 * num_points)
radius[plaque_start:] *= 0.4  # reduce radius by 60%
radius = np.clip(radius, radius_min, radius_max)

# Catheter parameters
frames = 200

# Compute tube surface
theta = np.linspace(0, 2 * np.pi, tube_points)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (tube_points, 2)

# Tangent vectors
dx = np.gradient(x)
dy = np.gradient(y)
dz = np.gradient(z)
tangent = np.stack([dx, dy, dz], axis=1)
tangent /= np.linalg.norm(tangent, axis=1)[:, None]

# Find two vectors perpendicular to tangent at each point
arbitrary = np.tile([0, 0, 1], (num_points, 1))
close_to_z = np.abs(tangent @ np.array([0, 0, 1])) > 0.99
arbitrary[close_to_z] = [0, 1, 0]
normal = np.cross(tangent, arbitrary)
normal /= np.linalg.norm(normal, axis=1)[:, None]
binormal = np.cross(tangent, normal)

# Build tube surface (no plaque)
X = np.zeros((num_points, tube_points))
Y = np.zeros((num_points, tube_points))
Z = np.zeros((num_points, tube_points))
for i in range(num_points):
    for j in range(tube_points):
        ang = 2 * np.pi * j / tube_points
        local_radius_ij = radius[i]
        X[i, j] = x[i] + local_radius_ij * (normal[i, 0] * np.cos(ang) + binormal[i, 0] * np.sin(ang))
        Y[i, j] = y[i] + local_radius_ij * (normal[i, 1] * np.cos(ang) + binormal[i, 1] * np.sin(ang))
        Z[i, j] = z[i] + local_radius_ij * (normal[i, 2] * np.cos(ang) + binormal[i, 2] * np.sin(ang))

# Find bifurcation point and tangent
bif_idx = int(branch_fraction * num_points)
bif_x, bif_y, bif_z = x[bif_idx], y[bif_idx], z[bif_idx]

# --- Branch direction: project global +y onto the normal/binormal plane at bifurcation ---
bif_tangent = np.array([dx[bif_idx], dy[bif_idx], dz[bif_idx]])
bif_tangent /= np.linalg.norm(bif_tangent)
global_y = np.array([0, 1, 0])
proj = global_y - np.dot(global_y, bif_tangent) * bif_tangent
branch_dir = proj / np.linalg.norm(proj)

# Branch centerline
branch_t = np.linspace(0, branch_length, branch_points)
branch_x = bif_x + branch_t * branch_dir[0]
branch_y = bif_y + branch_t * branch_dir[1]
branch_z = bif_z + branch_t * branch_dir[2]

# Branch tube surface
branch_theta = np.linspace(0, 2 * np.pi, tube_points)
branch_circle = np.stack([np.cos(branch_theta), np.sin(branch_theta)], axis=1)
# Branch tangent (constant)
branch_tangent = branch_dir
# Find two perpendicular vectors for branch
if np.abs(branch_tangent[2]) < 0.9:
    branch_normal = np.cross(branch_tangent, [0, 0, 1])
else:
    branch_normal = np.cross(branch_tangent, [0, 1, 0])
branch_normal = branch_normal / np.linalg.norm(branch_normal)
branch_binormal = np.cross(branch_tangent, branch_normal)

# Build branch tube surface
branch_X = np.zeros((branch_points, tube_points))
branch_Y = np.zeros((branch_points, tube_points))
branch_Z = np.zeros((branch_points, tube_points))
for i in range(branch_points):
    for j in range(tube_points):
        branch_X[i, j] = branch_x[i] + branch_radius * (
            branch_normal[0] * branch_circle[j, 0] + branch_binormal[0] * branch_circle[j, 1])
        branch_Y[i, j] = branch_y[i] + branch_radius * (
            branch_normal[1] * branch_circle[j, 0] + branch_binormal[1] * branch_circle[j, 1])
        branch_Z[i, j] = branch_z[i] + branch_radius * (
            branch_normal[2] * branch_circle[j, 0] + branch_binormal[2] * branch_circle[j, 1])

# Animation setup
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 2])
ax = fig.add_subplot(gs[0], projection='3d')
ax_ivus = fig.add_subplot(gs[1])
ax_imp = fig.add_subplot(gs[2])
ax.set_box_aspect([vessel_length, 2, 2])
ax.set_xlim(-1, vessel_length + 1)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_title('Catheter Motion Through 3D Vessel')
ax.set_axis_off()
ax.view_init(elev=30, azim=45)  # Isometric view

# Make background and panes white/transparent, hide axes and grid
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.grid(False)
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.xaxis.pane.set_alpha(0)
ax.yaxis.pane.set_alpha(0)
ax.zaxis.pane.set_alpha(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.title.set_visible(False)

# OPTIMIZATION 1: Pre-allocate arrays instead of growing lists
forward_impedance_vals = np.full(frames, np.nan)
x_positions = np.full(frames, np.nan)

# Impedance signal plot setup
forward_imp_line, = ax_imp.plot([], [], 'b--o', label='forward 1cm', markersize=3)
ax_imp.set_xlim(0, vessel_length)
ax_imp.set_ylim(0, 1.5 / (np.pi * radius_min**2))
ax_imp.set_title('Forward-Looking Impedance')
ax_imp.set_xlabel('Position (cm)')
ax_imp.set_ylabel('Impedance (a.u.)')
ax_imp.legend()

# OPTIMIZATION 2: Track surfaces globally for proper removal
vessel_surface = None
branch_surface = None

# Catheter as a growing line from the origin to the tip (actual path)
catheter_line, = ax.plot([], [], [], 'k-', lw=0.8)  # thinner black line

# OPTIMIZATION 3: Pre-allocate catheter path arrays
catheter_actual_x = np.full(frames + 10, np.nan)
catheter_actual_y = np.full(frames + 10, np.nan)
catheter_actual_z = np.full(frames + 10, np.nan)
catheter_actual_x[0] = x[0]
catheter_actual_y[0] = y[0]
catheter_actual_z[0] = z[0]
catheter_path_length = 1

catheter_offsets = [(0, 0)]  # (u, v) offsets in normal/binormal at each step
catheter_start_x = -1.0  # 1 cm before vessel

# Helper: get wall distances from IVUS image generation
def get_wall_distances(inner_wall_r, outer_wall_r):
    # Return the mean and std of the wall thickness (outer - inner) in all directions
    return np.mean(outer_wall_r - inner_wall_r), np.std(outer_wall_r - inner_wall_r)

# OPTIMIZATION 4: Simplified IVUS simulation for speed
def simulate_ivus_image_with_profile(local_radius, wall_irreg_seed=0, show_branch=False, branch_angle=None, offset=(0,0), normal=None, binormal=None, tip_pos_3d=None, x=None, y=None, z=None, radius=None, num_points=None):
    img_size = 32
    n_rays = 32
    n_samples = 32
    center = img_size // 2
    np.random.seed(wall_irreg_seed)
    y_img, x_img = np.ogrid[:img_size, :img_size]
    r = np.sqrt((x_img - center) ** 2 + (y_img - center) ** 2)
    theta = np.arctan2(y_img - center, x_img - center)
    phi = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    vessel_radius_pix = local_radius * (img_size / 2) / radius_max
    wall_thickness = 0.18 * (img_size / 2)
    inner_irreg = 1 + 0.06 * gaussian_filter1d(np.random.normal(0, 1, n_rays), sigma=2)
    outer_irreg = 1 + 0.06 * gaussian_filter1d(np.random.normal(0, 1, n_rays), sigma=2)
    inner_wall_r = vessel_radius_pix * inner_irreg
    # Compute healthy outer wall radius (based on max radius, not local_radius)
    healthy_outer_wall_r = (radius_max * (img_size / 2) / radius_max + wall_thickness) * outer_irreg
    outer_wall_r = healthy_outer_wall_r.copy()
    # Simulate plaque at the end: only reduce inner wall radius if near the end
    if tip_pos_3d is not None and x is not None:
        distances = np.linalg.norm(np.stack([x, y, z], axis=1) - tip_pos_3d, axis=1)
        nearest_idx = np.argmin(distances)
        if nearest_idx > int(0.9 * len(x)):
            inner_wall_r *= 0.4  # reduce inner diameter by 60%
            # outer_wall_r stays the same (do not change)
    vessel_map = np.zeros((img_size, img_size))
    for i, ang in enumerate(phi):
        local_inner = inner_wall_r[i]
        local_outer = outer_wall_r[i]
        mask = (r >= local_inner) & (r <= local_outer) & (np.abs(theta - ang) < np.pi / n_rays)
        vessel_map[mask] = 1.0 * (1.0 + 0.7 * np.random.rayleigh(0.7, np.count_nonzero(mask)))
    lumen_mask = r < np.interp(theta, phi, inner_wall_r)
    vessel_map[lumen_mask] = 0.05 * np.random.rayleigh(0.7, np.count_nonzero(lumen_mask))
    ext_mask = r > np.interp(theta, phi, outer_wall_r)
    vessel_map[ext_mask] = 0.1 * np.random.rayleigh(0.7, np.count_nonzero(ext_mask))
    vessel_map[center-2:center+2, center-2:center+2] += 1.5
    vessel_map = gaussian_filter(vessel_map, sigma=0.5)
    vessel_map = np.clip(vessel_map, 0, 1)
    polar_img = np.zeros((n_samples, n_rays))
    for i, ang in enumerate(phi):
        for s in range(n_samples):
            rr = s * (img_size / 2) / n_samples
            xx = int(center + rr * np.cos(ang))
            yy = int(center + rr * np.sin(ang))
            if 0 <= xx < img_size and 0 <= yy < img_size:
                val = vessel_map[yy, xx]
            else:
                val = 0
            att = np.exp(-0.025 * s)
            polar_img[s, i] = val * att + 0.08 * np.random.randn()
    polar_img -= polar_img.min()
    polar_img /= (polar_img.max() + 1e-6)
    polar_img = np.clip(1.5 * (polar_img - 0.5) + 0.5, 0, 1)
    y_idx, x_idx = np.ogrid[:img_size, :img_size]
    r_idx = np.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2)
    phi_idx = np.arctan2(y_idx - center, x_idx - center)
    phi_idx = ((phi_idx + np.pi) / (2 * np.pi) * n_rays).astype(int) % n_rays
    r_idx = np.clip((r_idx / (img_size / 2) * n_samples).astype(int), 0, n_samples-1)
    img = polar_img[r_idx, phi_idx]
    img = np.clip(img, 0, 1)
    # Mask to a circle to remove black corners
    mask = r <= (img_size / 2)
    img = img * mask
    outer_wall_r_profile = outer_wall_r
    angle_idx = ((phi_idx + n_rays) % n_rays)
    outer_wall_r_img = outer_wall_r_profile[angle_idx]
    ring_mask = np.abs(r - outer_wall_r_img) < 1.5
    img[ring_mask] = 1.0
    return img, inner_wall_r, outer_wall_r

def compute_integral_impedance(start_x, path_length_cm, x, radius, num_samples=20):
    # Compute impedance as integral of ds/area along the centerline, starting from start_x, over path_length_cm
    xs = np.linspace(start_x, min(start_x + path_length_cm, x[-1]), num_samples)
    idxs = np.searchsorted(x, xs)
    idxs = np.clip(idxs, 0, len(radius) - 1)
    areas = np.pi * (radius[idxs] ** 2)
    ds = (xs[-1] - xs[0]) / (num_samples - 1) if num_samples > 1 else 0.01
    Z = np.sum(ds / areas)
    return Z

# OPTIMIZED Animation function
def animate(i):
    global vessel_surface, branch_surface, catheter_path_length
    
    # Reset arrays on first frame only
    if i == 0 and catheter_path_length > 1:
        catheter_actual_x.fill(np.nan)
        catheter_actual_y.fill(np.nan)
        catheter_actual_z.fill(np.nan)
        forward_impedance_vals.fill(np.nan)
        x_positions.fill(np.nan)
        catheter_actual_x[0] = x[0]
        catheter_actual_y[0] = y[0]
        catheter_actual_z[0] = z[0]
        catheter_path_length = 1
        catheter_offsets.clear()
        catheter_offsets.append((0, 0))
    
    tip_idx = int(i * (num_points - 1) / (frames - 1))
    tip_x, tip_y, tip_z = x[tip_idx], y[tip_idx], z[tip_idx]
    nrm = normal[tip_idx]
    bnr = binormal[tip_idx]
    local_radius = radius[tip_idx]
    
    # OPTIMIZATION 5: Simplified catheter positioning (reduced grid search)
    grid_n = 1  # Only centerline, no offset search
    u_vals = [0]
    v_vals = [0]
    centerline_bias_weight = 0.1
    best_cost = np.inf
    best_offset = (0, 0)
    best_img = None
    best_inner = None
    best_outer = None
    
    for u in u_vals:
        for v in v_vals:
            if np.sqrt(u**2 + v**2) > 0.9 * local_radius:
                continue
            tip_pos = np.array([tip_x, tip_y, tip_z]) + u * nrm + v * bnr
            distances = np.linalg.norm(np.stack([x, y, z], axis=1) - tip_pos, axis=1)
            nearest_idx = np.argmin(distances)
            local_r = radius[nearest_idx]
            show_branch = False
            branch_angle_img = None
            if tip_idx >= bif_idx and tip_idx < bif_idx + 20:
                show_branch = True
                branch_angle_img = branch_angle
            img, inner_wall_r, outer_wall_r = simulate_ivus_image_with_profile(
                local_r, wall_irreg_seed=nearest_idx, show_branch=show_branch, 
                branch_angle=branch_angle_img, offset=(u, v), normal=nrm, binormal=bnr, 
                tip_pos_3d=tip_pos, x=x, y=y, z=z, radius=radius, num_points=num_points)
            min_dist_to_wall = np.min(inner_wall_r)
            max_possible_dist = local_r
            centerline_dist = np.sqrt(u**2 + v**2)
            cost = -min_dist_to_wall / max_possible_dist + centerline_bias_weight * (centerline_dist / max_possible_dist)
            if cost < best_cost:
                best_cost = cost
                best_offset = (u, v)
                best_img = img
                best_inner = inner_wall_r
                best_outer = outer_wall_r
    
    max_step = 0.2 * local_radius  # only allow small moves per frame
    if len(catheter_offsets) > 0:
        prev_u, prev_v = catheter_offsets[-1]
        du = np.clip(best_offset[0] - prev_u, -max_step, max_step)
        dv = np.clip(best_offset[1] - prev_v, -max_step, max_step)
        u = prev_u + du
        v = prev_v + dv
    else:
        u, v = best_offset
    
    tip_pos = np.array([tip_x, tip_y, tip_z]) + u * nrm + v * bnr
    
    # OPTIMIZATION 6: Update catheter path in pre-allocated arrays
    if catheter_path_length <= i + 1:
        catheter_actual_x[catheter_path_length] = tip_pos[0]
        catheter_actual_y[catheter_path_length] = tip_pos[1]
        catheter_actual_z[catheter_path_length] = tip_pos[2]
        catheter_path_length += 1
        catheter_offsets.append((u, v))
    else:
        catheter_actual_x[i + 1] = tip_pos[0]
        catheter_actual_y[i + 1] = tip_pos[1]
        catheter_actual_z[i + 1] = tip_pos[2]
    
    reveal_x = tip_x + reveal_ahead
    reveal_idx = np.searchsorted(x, reveal_x)
    
    # OPTIMIZATION 7: Safe surface removal
    try:
        if vessel_surface is not None:
            vessel_surface.remove()
    except (ValueError, AttributeError):
        pass
    
    # After defining radius, add:
    outer_radius = np.full_like(radius, radius_max)

    # When building the tube surface, build both inner and outer wall surfaces:
    # Build inner wall tube surface (lumen)
    X_inner = np.zeros((num_points, tube_points))
    Y_inner = np.zeros((num_points, tube_points))
    Z_inner = np.zeros((num_points, tube_points))
    for i in range(num_points):
        for j in range(tube_points):
            ang = 2 * np.pi * j / tube_points
            local_radius_ij = radius[i]
            X_inner[i, j] = x[i] + local_radius_ij * (normal[i, 0] * np.cos(ang) + binormal[i, 0] * np.sin(ang))
            Y_inner[i, j] = y[i] + local_radius_ij * (normal[i, 1] * np.cos(ang) + binormal[i, 1] * np.sin(ang))
            Z_inner[i, j] = z[i] + local_radius_ij * (normal[i, 2] * np.cos(ang) + binormal[i, 2] * np.sin(ang))

    # Build outer wall tube surface (healthy vessel)
    X_outer = np.zeros((num_points, tube_points))
    Y_outer = np.zeros((num_points, tube_points))
    Z_outer = np.zeros((num_points, tube_points))
    for i in range(num_points):
        for j in range(tube_points):
            ang = 2 * np.pi * j / tube_points
            local_radius_ij = outer_radius[i]
            X_outer[i, j] = x[i] + local_radius_ij * (normal[i, 0] * np.cos(ang) + binormal[i, 0] * np.sin(ang))
            Y_outer[i, j] = y[i] + local_radius_ij * (normal[i, 1] * np.cos(ang) + binormal[i, 1] * np.sin(ang))
            Z_outer[i, j] = z[i] + local_radius_ij * (normal[i, 2] * np.cos(ang) + binormal[i, 2] * np.sin(ang))

    # In the animate function, plot both surfaces:
    # (replace vessel_surface plotting with the following)
    facecolors_inner = np.full(X_inner[:reveal_idx, :].shape + (4,), (1.0, 0.0, 0.0, 0.15))  # transparent red
    facecolors_outer = np.full(X_outer[:reveal_idx, :].shape + (4,), (1.0, 0.0, 0.0, 0.05))  # more transparent for outer wall
    try:
        if vessel_surface is not None:
            vessel_surface.remove()
    except (ValueError, AttributeError):
        pass
    vessel_surface = ax.plot_surface(
        X_outer[:reveal_idx, :], Y_outer[:reveal_idx, :], Z_outer[:reveal_idx, :],
        facecolors=facecolors_outer, linewidth=0, antialiased=False, shade=False)
    # Plot the inner wall (lumen) as well
    try:
        if hasattr(ax, 'inner_surface') and ax.inner_surface is not None:
            ax.inner_surface.remove()
    except (ValueError, AttributeError):
        pass
    ax.inner_surface = ax.plot_surface(
        X_inner[:reveal_idx, :], Y_inner[:reveal_idx, :], Z_inner[:reveal_idx, :],
        facecolors=facecolors_inner, linewidth=0, antialiased=False, shade=False)
    
    # OPTIMIZATION 8: Proper branch surface handling
    if tip_idx >= bif_idx:
        try:
            if branch_surface is not None:
                branch_surface.remove()
        except (ValueError, AttributeError):
            pass
        
        branch_reveal_idx = int(min(branch_points, (tip_x - bif_x + branch_reveal_ahead) / branch_length * branch_points))
        branch_reveal_idx = max(branch_reveal_idx, 1)
        branch_surface = ax.plot_surface(
            branch_X[:branch_reveal_idx, :], branch_Y[:branch_reveal_idx, :], branch_Z[:branch_reveal_idx, :],
            facecolors=np.full(branch_X[:branch_reveal_idx, :].shape + (4,), (1.0, 0.0, 0.0, 0.07)), 
            linewidth=0, antialiased=False, shade=False)
    else:
        try:
            if branch_surface is not None:
                branch_surface.remove()
                branch_surface = None
        except (ValueError, AttributeError):
            branch_surface = None
    
    # OPTIMIZATION 9: Efficient catheter path visualization
    current_tip_x = catheter_actual_x[min(i+1, catheter_path_length-1)]
    if current_tip_x > 0:
        pre_vessel_x = np.linspace(catheter_start_x, 0, 10)
        pre_vessel_y = np.zeros(10)
        pre_vessel_z = np.zeros(10)
        valid_catheter = catheter_actual_x[:catheter_path_length]
        valid_catheter = valid_catheter[~np.isnan(valid_catheter)]
        full_x = np.concatenate([pre_vessel_x, valid_catheter[:i+2]])
        full_y = np.concatenate([pre_vessel_y, catheter_actual_y[:len(valid_catheter[:i+2])]])
        full_z = np.concatenate([pre_vessel_z, catheter_actual_z[:len(valid_catheter[:i+2])]])
    else:
        full_x = np.linspace(catheter_start_x, current_tip_x, 10)
        full_y = np.zeros(10)
        full_z = np.zeros(10)
    
    catheter_line.set_data(full_x, full_y)
    catheter_line.set_3d_properties(full_z)
    
    # OPTIMIZATION 10: Efficient impedance calculation
    forward_impedance = compute_integral_impedance(current_tip_x, 1.0, x, radius)
    noise_level = 0.1
    noisy_forward_impedance = forward_impedance * (1 + np.random.normal(0, noise_level))
    
    forward_impedance_vals[i] = noisy_forward_impedance
    x_positions[i] = current_tip_x
    
    # Update impedance plot efficiently
    valid_impedance = forward_impedance_vals[:i+1]
    valid_x_pos = x_positions[:i+1]
    forward_imp_line.set_data(valid_x_pos, valid_impedance)
    
    # OPTIMIZATION 11: Remove expensive canvas redraw
    # Removed: ax_imp.figure.canvas.draw() - this was the major bottleneck!
    
    # Update IVUS
    ivus_img.set_data(best_img)
    


ivus_img = ax_ivus.imshow(np.zeros((32, 32)), cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
ani = FuncAnimation(fig, animate, frames=frames, interval=30, repeat=True)

plt.show()