import cv2
import numpy as np


def nearest_upsample(lr_depth, target_shape):
    """Nearest-neighbor depth upsampling baseline."""
    H, W = target_shape
    return cv2.resize(lr_depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def bilinear_upsample(lr_depth, target_shape):
    """Bilinear depth upsampling baseline."""
    H, W = target_shape
    return cv2.resize(lr_depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def bicubic_upsample(lr_depth, target_shape):
    """Bicubic depth upsampling baseline."""
    H, W = target_shape
    res = cv2.resize(lr_depth.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return np.maximum(res, 0.0)


def fill_depth_holes(depth_map, mask=None, method='inpaint'):
    """
    Fills missing or zero-depth holes from depth sensors using Telea inpainting.
    """
    H, W = depth_map.shape
    if mask is None:
        mask = (depth_map > 0.05).astype(np.uint8)

    hole_mask = (mask == 0).astype(np.uint8)
    if np.sum(hole_mask) == 0:
        return depth_map.copy()

    max_d = np.max(depth_map)
    if max_d == 0:
        return depth_map

    normalized = (depth_map / max_d * 255.0).astype(np.uint8)
    inpainted_8u = cv2.inpaint(normalized, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    filled_depth = (inpainted_8u.astype(np.float32) / 255.0) * max_d

    return np.where(mask > 0, depth_map, filled_depth)


def guided_filter_refinement(depth_map, rgb_guide, radius=5, eps=1e-3):
    """
    Fast RGB-Guided Filter for Edge-Preserving Depth Refinement (He et al.).
    """
    if len(rgb_guide.shape) == 3:
        guide = cv2.cvtColor((rgb_guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        guide = rgb_guide.astype(np.float32)
        if guide.max() > 1.0:
            guide /= 255.0

    p = depth_map.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)

    denom = np.maximum(var_I + eps, 1e-6)
    a = cov_Ip / denom
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * guide + mean_b
    return np.clip(q, 0.0, 15.0)


def joint_bilateral_filter(depth_map, rgb_guide, diameter=7, sigma_color=0.1, sigma_space=5.0):
    """
    Joint Bilateral Filter using OpenCV cv2.ximgproc.jointBilateralFilter (with fallback).
    """
    max_d = float(np.max(depth_map))
    if max_d == 0:
        return depth_map

    d_norm = (depth_map / max_d * 255.0).astype(np.float32)
    
    if len(rgb_guide.shape) == 3:
        guide_8u = (rgb_guide * 255.0).astype(np.uint8)
    else:
        guide_8u = (rgb_guide * 255.0).astype(np.uint8)

    try:
        import cv2.ximgproc as xip
        filtered = xip.jointBilateralFilter(guide_8u, d_norm.astype(np.uint8), diameter, sigma_color * 255.0, sigma_space)
        return (filtered.astype(np.float32) / 255.0) * max_d
    except Exception:
        # Guided filter fallback
        return guided_filter_refinement(depth_map, rgb_guide, radius=int(diameter//2), eps=(sigma_color ** 2))


def detect_depth_discontinuities(depth_map, threshold=0.25):
    """
    Detects depth boundaries and discontinuities between foreground and background objects.
    """
    sobel_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_map = (grad_mag > threshold).astype(np.uint8) * 255
    return edge_map, grad_mag


def edge_gated_boundary_refinement(unet_depth, rgb_guide, radius=5, eps=1e-3, depth_edge_thresh=0.15, rgb_edge_thresh=0.20):
    """
    Edge-Gated Boundary Refinement (Fixes Texture-Copying Artifacts):
    
    CRITICAL PRINCIPLE:
    Naive global guided filtering copies texture patterns (bedsheet wrinkles, wallpaper patterns,
    wood grain, shadows) into flat depth areas where depth should remain completely smooth and planar.
    
    Depth-Discontinuity Constrained Gate:
    1. Computes depth gradient magnitude |grad(D)| from the U-Net prediction.
    2. Identifies true depth discontinuities where |grad(D)| > depth_edge_thresh.
    3. Dilates this mask by 3x3 kernel to form a localized boundary transition zone B.
    4. Intersects with structural RGB edges within B: Gate G = B * (|grad(I)| > rgb_edge_thresh).
       - On planar textured surfaces: |grad(D)| == 0 -> B == 0 -> G == 0 (100% immune to texture copying).
       - At real object silhouettes: G activates, allowing RGB guidance to sharpen object boundaries.
    5. Blends: D_hybrid = G * D_guided + (1 - G) * D_unet.
    """
    # 1. Compute RGB-guided refined candidate
    refined_depth = guided_filter_refinement(unet_depth, rgb_guide, radius=radius, eps=eps)

    # 2. Gradient magnitude of U-Net depth prediction
    sobel_dx = cv2.Sobel(unet_depth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_dy = cv2.Sobel(unet_depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_depth = np.sqrt(sobel_dx**2 + sobel_dy**2)

    # 3. Gradient magnitude of RGB luminance guide
    if len(rgb_guide.shape) == 3:
        gray = cv2.cvtColor((rgb_guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = rgb_guide.astype(np.float32)
    sobel_rx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_ry = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_rgb = np.sqrt(sobel_rx**2 + sobel_ry**2)

    # 4. Depth-Discontinuity Constrained Boundary Zone (prevents texture leakage in planar regions)
    depth_discontinuity = (grad_depth > depth_edge_thresh).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary_zone = cv2.dilate(depth_discontinuity, kernel)

    # Edge Gate activates ONLY at true object silhouettes where RGB edge guides the boundary
    edge_mask = boundary_zone * ((grad_rgb > rgb_edge_thresh) | (grad_depth > depth_edge_thresh * 1.5)).astype(np.float32)
    
    # Smooth gate for seamless spatial blending
    gate = cv2.GaussianBlur(edge_mask, (3, 3), 0.8)
    gate = np.clip(gate * 1.2, 0.0, 1.0)

    # 5. Edge-Gated Blend
    hybrid_depth = gate * refined_depth + (1.0 - gate) * unet_depth
    return np.clip(hybrid_depth, 0.0, 15.0), gate


# Alias for compatibility
edge_gated_hybrid_refinement = edge_gated_boundary_refinement
