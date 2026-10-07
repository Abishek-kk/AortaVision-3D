"""
Anatomically accurate adult human heart – procedural 3D model generator.
Version 2 – Complete redesign with proper silhouette and anatomy.

Coordinate system (right-hand, Y-up, looking from +Z):
  X+ = anatomical RIGHT  (right atrium / SVC / RCA) = viewer's RIGHT
  X- = anatomical LEFT   (LV apex / left auricle)   = viewer's LEFT
  Y+ = superior (base)
  Y- = inferior (apex)
  Z+ = anterior (facing viewer)
  Z- = posterior
"""

import numpy as np
import trimesh
import os
import math
import json
import struct

# ─────────────────────────────────────────────────────────────────────────────
# sRGB ↔ linear helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_lin(c255):
    """uint8 sRGB → linear float [0,1]."""
    v = c255 / 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

def _rgba_lin(r, g, b, a=255):
    return [_to_lin(r), _to_lin(g), _to_lin(b), a / 255.0]


# ─────────────────────────────────────────────────────────────────────────────
# TUBE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_tangents(pts):
    pts = np.asarray(pts, dtype=float)
    t = np.zeros_like(pts)
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    t[1:-1] = pts[2:] - pts[:-2]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return t / norms


def _parallel_transport(pts, tangents):
    n = len(pts)
    B = np.zeros((n, 3))
    N = np.zeros((n, 3))
    t0 = tangents[0]
    # seed frame
    ref = np.array([0, 1, 0]) if abs(t0[1]) < 0.9 else np.array([1, 0, 0])
    B[0] = np.cross(t0, ref)
    B[0] /= np.linalg.norm(B[0]) + 1e-12
    N[0] = np.cross(B[0], t0)
    for i in range(1, n):
        b = np.cross(tangents[i - 1], tangents[i])
        bl = np.linalg.norm(b)
        if bl < 1e-8:
            B[i] = B[i - 1]
        else:
            b /= bl
            ang = math.acos(max(-1, min(1, float(np.dot(tangents[i - 1], tangents[i])))))
            c, s = math.cos(ang), math.sin(ang)
            B[i] = c * B[i - 1] + s * np.cross(b, B[i - 1])
            B[i] /= np.linalg.norm(B[i]) + 1e-12
        N[i] = np.cross(B[i], tangents[i])
    return B, N


def create_solid_tube(path_pts, radius=0.05, num_sides=16):
    pts = np.asarray(path_pts, dtype=float)
    if len(pts) < 2:
        return trimesh.Trimesh()
    tangents = _compute_tangents(pts)
    B, N = _parallel_transport(pts, tangents)
    if np.isscalar(radius):
        radii = np.full(len(pts), radius)
    else:
        radii = np.asarray(radius, dtype=float)
        if len(radii) != len(pts):
            radii = np.interp(np.linspace(0, 1, len(pts)),
                              np.linspace(0, 1, len(radii)), radii)
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    verts, faces = [], []
    for i, (p, b, n_, r) in enumerate(zip(pts, B, N, radii)):
        ring = [p + r * (math.cos(a) * b + math.sin(a) * n_) for a in angles]
        base = len(verts)
        verts.extend(ring)
        if i > 0:
            prev = base - num_sides
            for j in range(num_sides):
                jn = (j + 1) % num_sides
                faces += [[prev + j, base + j, base + jn],
                           [prev + j, base + jn, prev + jn]]
    # end-caps
    c0 = np.mean(verts[:num_sides], axis=0)
    verts.append(c0)
    ci = len(verts) - 1
    for j in range(num_sides):
        faces.append([ci, j, (j + 1) % num_sides])
    c1 = np.mean(verts[-num_sides - 1:-1], axis=0)
    verts.append(c1)
    ci2 = len(verts) - 1
    base_last = ci2 - num_sides - 1
    for j in range(num_sides):
        faces.append([ci2, base_last + (j + 1) % num_sides, base_last + j])
    return trimesh.Trimesh(vertices=np.array(verts),
                           faces=np.array(faces), process=False)


def create_hollow_tube(path_pts, outer_r=0.12, inner_r=0.09, num_sides=24):
    pts = np.asarray(path_pts, dtype=float)
    if len(pts) < 2:
        return trimesh.Trimesh()
    tangents = _compute_tangents(pts)
    B, N = _parallel_transport(pts, tangents)
    def _radii(r_spec):
        if np.isscalar(r_spec):
            return np.full(len(pts), r_spec)
        r = np.asarray(r_spec, dtype=float)
        if len(r) != len(pts):
            r = np.interp(np.linspace(0, 1, len(pts)),
                          np.linspace(0, 1, len(r)), r)
        return r
    OR = _radii(outer_r)
    IR = _radii(inner_r)
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    verts, faces = [], []
    n_rings = len(pts)
    for i, (p, b, nv, ro, ri) in enumerate(zip(pts, B, N, OR, IR)):
        for r in (ro, ri):
            ring = [p + r * (math.cos(a) * b + math.sin(a) * nv) for a in angles]
            verts.extend(ring)
        if i > 0:
            prev_o = (i - 1) * 2 * num_sides
            prev_i = prev_o + num_sides
            cur_o  = i * 2 * num_sides
            cur_i  = cur_o + num_sides
            for j in range(num_sides):
                jn = (j + 1) % num_sides
                # outer
                faces += [[prev_o+j, cur_o+j, cur_o+jn],
                           [prev_o+j, cur_o+jn, prev_o+jn]]
                # inner (reversed winding)
                faces += [[prev_i+j, prev_i+jn, cur_i+jn],
                           [prev_i+j, cur_i+jn, cur_i+j]]
    # end-caps (annular rings)
    def _cap(ring_o_start, ring_i_start, reverse=False):
        for j in range(num_sides):
            jn = (j + 1) % num_sides
            a, b2 = ring_o_start + j, ring_o_start + jn
            c2, d = ring_i_start + jn, ring_i_start + j
            if reverse:
                faces.extend([[a, d, c2], [a, c2, b2]])
            else:
                faces.extend([[a, b2, c2], [a, c2, d]])
    _cap(0, num_sides, reverse=True)
    last_o = (n_rings - 1) * 2 * num_sides
    last_i = last_o + num_sides
    _cap(last_o, last_i, reverse=False)
    return trimesh.Trimesh(vertices=np.array(verts),
                           faces=np.array(faces), process=False)


def catmull_rom(ctrl_pts, samples_per_seg=8):
    """Smooth Catmull-Rom spline through control points."""
    pts = np.asarray(ctrl_pts, dtype=float)
    if len(pts) < 2:
        return pts
    # pad ends
    padded = np.vstack([2*pts[0] - pts[1], pts, 2*pts[-1] - pts[-2]])
    result = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i-1], padded[i], padded[i+1], padded[i+2]
        for k in range(samples_per_seg):
            t = k / samples_per_seg
            t2, t3 = t*t, t*t*t
            result.append(0.5 * ((2*p1)
                                  + (-p0+p2)*t
                                  + (2*p0-5*p1+4*p2-p3)*t2
                                  + (-p0+3*p1-3*p2+p3)*t3))
    result.append(pts[-1])
    return np.array(result)


# ─────────────────────────────────────────────────────────────────────────────
# HEART BODY – complete redesign
# ─────────────────────────────────────────────────────────────────────────────

def generate_heart_body(lat_res=144, lon_res=240):
    """
    Anatomically accurate heart body parametric mesh.

    Shape:
    - Large conical left ventricle, apex down-left
    - Flat wedge right ventricle on the anterior-right surface
    - Right atrium: large rounded bump, right-superior, with ear-shaped auricle
    - Left atrium: posterior superior, with small left auricle peering anteriorly
    - AV groove: narrow, dark, DIAGONAL (not a horizontal band)
    - Anterior + posterior interventricular grooves
    """
    u = np.linspace(0, np.pi, lat_res)   # 0 = base/top, π = apex/bottom
    v = np.linspace(0, 2*np.pi, lon_res, endpoint=False)  # 0 = X+/right
    U, V = np.meshgrid(u, v, indexing='ij')  # shapes: (lat, lon)

    # Fraction helpers
    apex_frac  = U / np.pi                                     # 0 at base, 1 at apex
    vent_frac  = np.clip((U - 0.37*np.pi) / (0.63*np.pi), 0, 1)  # 0 at AV groove, 1 at apex
    atrial_frac = np.clip(1.0 - U / (0.37*np.pi), 0, 1)          # 1 at top, 0 at AV groove

    # ── Base ellipsoid: tapered toward apex ───────────────────────────────
    X = 1.28 * np.sin(U) * np.cos(V)
    Y = 1.22 * np.cos(U)
    Z = 0.82 * np.sin(U) * np.sin(V)

    # Narrow toward apex
    narrow = 1.0 - 0.34 * apex_frac
    X *= narrow
    Z *= narrow

    # ── Apex tilt: apex points DOWN and to the LEFT (-X, +Z slightly) ────
    tilt = apex_frac ** 1.4
    X -= tilt * 0.38        # shift apex leftward
    Z += tilt * 0.13        # slight anterior shift at apex
    Y -= tilt * 0.08        # make apex slightly more inferior

    # ── Right atrium: prominent rounded bump right-superior ───────────────
    ra_v_c = 0.18           # slightly anterior to pure X+
    ra_vd  = np.minimum(np.abs(V - ra_v_c), 2*np.pi - np.abs(V - ra_v_c))
    ra = (0.34 * np.exp(-ra_vd**2 / (2*0.32**2))
               * np.exp(-((U - 0.23*np.pi)**2) / (2*(0.14*np.pi)**2))
               * atrial_frac)
    X += ra * 0.90
    Z += ra * 0.50

    # ── Right auricle: ear-shaped flap anterio-superior from RA ──────────
    rau_v_c = 0.60          # anterior-right
    rau_vd  = np.minimum(np.abs(V - rau_v_c), 2*np.pi - np.abs(V - rau_v_c))
    rau = (0.22 * np.exp(-rau_vd**2 / (2*0.075**2))
                * np.exp(-((U - 0.19*np.pi)**2) / (2*(0.047*np.pi)**2)))
    X += rau * 0.55
    Z += rau * 1.35
    Y += rau * 0.12

    # ── Left atrium: posterior superior ──────────────────────────────────
    la_v_c = 1.62*np.pi     # posterior-left
    la_vd  = np.minimum(np.abs(V - la_v_c), 2*np.pi - np.abs(V - la_v_c))
    la = (0.27 * np.exp(-la_vd**2 / (2*0.29**2))
               * np.exp(-((U - 0.21*np.pi)**2) / (2*(0.11*np.pi)**2))
               * atrial_frac)
    X += la * np.cos(la_v_c) * 0.86
    Z += la * np.sin(la_v_c) * 0.86

    # ── Posterior right atrium: a second distinct chamber beside the LA ──
    pra_v_c = 1.25*np.pi
    pra_vd = np.minimum(np.abs(V - pra_v_c), 2*np.pi - np.abs(V - pra_v_c))
    pra = (0.27 * np.exp(-pra_vd**2 / (2*0.33**2))
                * np.exp(-((U - 0.23*np.pi)**2) / (2*(0.15*np.pi)**2))
                * atrial_frac)
    X += pra * np.cos(pra_v_c) * 0.88
    Z += pra * np.sin(pra_v_c) * 0.78

    # ── Left auricle: small ear anterior-left (peeks beside pulmonary trunk) ──
    lau_v_c = 1.09*np.pi    # anterior-left
    lau_vd  = np.minimum(np.abs(V - lau_v_c), 2*np.pi - np.abs(V - lau_v_c))
    lau = (0.22 * np.exp(-lau_vd**2 / (2*0.075**2))
                * np.exp(-((U - 0.21*np.pi)**2) / (2*(0.052*np.pi)**2)))
    X += lau * np.cos(lau_v_c) * 1.15
    Z += lau * np.sin(lau_v_c) * 1.15
    Y += lau * 0.10

    # Shallow interatrial sulcus separates the two posterior atrial lobes.
    ia_v_c = 1.46*np.pi
    ia_vd = np.minimum(np.abs(V - ia_v_c), 2*np.pi - np.abs(V - ia_v_c))
    ia_u_mask = np.exp(-((U - 0.22*np.pi)**2) / (2*(0.15*np.pi)**2))
    ia_depth = 0.13 * np.exp(-48.0 * ia_vd**2) * ia_u_mask * atrial_frac
    X *= 1.0 - ia_depth
    Z *= 1.0 - ia_depth

    # ── Right ventricle: broad flat wedge on anterior-right face ─────────
    rv_v_c  = 0.62          # between anterior and right (V=π/2=1.57 is anterior)
    rv_vd   = np.minimum(np.abs(V - rv_v_c), 2*np.pi - np.abs(V - rv_v_c))
    rv_u_c  = 0.62*np.pi
    rv = (0.29 * np.exp(-rv_vd**2 / (2*0.46**2))
               * np.exp(-((U - rv_u_c)**2) / (2*(0.23*np.pi)**2))
               * vent_frac)
    Z += rv * 0.25
    X += rv * np.cos(V) * 0.14

    # ── Left ventricle: dominant mass, posterior-left, forms apex cone ────
    lv_v_c = 1.32*np.pi     # posterior-left
    lv_vd  = np.minimum(np.abs(V - lv_v_c), 2*np.pi - np.abs(V - lv_v_c))
    lv = (0.27 * np.exp(-lv_vd**2 / (2*0.56**2))
               * (vent_frac ** 0.55))
    X += lv * np.cos(lv_v_c) * 0.72
    Z += lv * np.sin(lv_v_c) * 0.38

    # ── AV groove: narrow, DIAGONAL (tilts lower on right, higher on left) ─
    av_u_c = 0.365*np.pi - 0.065*np.pi * np.cos(V)   # diagonal tilt
    av_dist = np.abs(U - av_u_c)
    av_depth = 0.095 * np.exp(-75.0 * av_dist**2)     # very narrow
    av_scale = 1.0 - av_depth
    X *= av_scale
    Z *= av_scale

    # ── Anterior interventricular groove (LAD groove) ────────────────────
    # Starts at AV groove center-anterior, curves left as it descends to apex
    aiv_v = 0.495*np.pi + 0.09*np.pi * np.clip((U - 0.37*np.pi)/(0.63*np.pi), 0, 1)
    aiv_dist = np.minimum(np.abs(V - aiv_v), 2*np.pi - np.abs(V - aiv_v))
    aiv = 0.085 * np.exp(-35.0 * aiv_dist**2) * np.clip((U - 0.35*np.pi)/(0.65*np.pi), 0, 1)
    aiv_sc = 1.0 - aiv
    X *= aiv_sc
    Z *= aiv_sc

    # ── Posterior interventricular groove ────────────────────────────────
    piv_v   = 1.52*np.pi
    piv_d   = np.minimum(np.abs(V - piv_v), 2*np.pi - np.abs(V - piv_v))
    piv     = 0.12 * np.exp(-26.0 * piv_d**2) * vent_frac
    piv_sc  = 1.0 - piv
    X *= piv_sc
    Z *= piv_sc

    # ── Subtle surface texture (muscle-fibre ripple) ─────────────────────
    tex = (0.0035 * np.sin(13*U + 0.5*np.sin(5*V)) * np.cos(17*V)
           + 0.0018 * np.cos(24*U + 7*V + 0.4*np.sin(9*V))
           + 0.0010 * np.sin(6*U - 19*V))
    atrial_relief = (0.0010 * np.exp(-((U - 0.22*np.pi) / (0.18*np.pi))**2)
                     * np.sin(35*V + 5*np.sin(11*U))
                     * np.cos(22*U + 3*np.sin(7*V)))
    ventricular_relief = (0.0007 * vent_frac
                          * np.sin(27*V + 5*np.sin(8*U))
                          * np.cos(33*U + 0.5*np.sin(9*V)))
    tex += atrial_relief + ventricular_relief
    X += X * tex
    Y += Y * tex * 0.3
    Z += Z * tex

    # ── Assemble ─────────────────────────────────────────────────────────
    verts_arr = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    faces = []
    for i in range(lat_res - 1):
        for j in range(lon_res):
            jn = (j + 1) % lon_res
            p1 = i*lon_res + j
            p2 = (i+1)*lon_res + j
            p3 = (i+1)*lon_res + jn
            p4 = i*lon_res + jn
            faces += [[p1, p2, p3], [p1, p3, p4]]

    top_center = len(verts_arr)
    bottom_center = top_center + 1
    verts_arr = np.vstack([
        verts_arr,
        verts_arr[:lon_res].mean(axis=0),
        verts_arr[-lon_res:].mean(axis=0),
    ])
    for j in range(lon_res):
        jn = (j + 1) % lon_res
        faces.append([top_center, jn, j])
        last = (lat_res - 1) * lon_res
        faces.append([bottom_center, last + j, last + jn])

    mesh = trimesh.Trimesh(vertices=verts_arr,
                           faces=np.array(faces), process=True)
    if mesh.volume < 0:
        mesh.invert()
    verts_arr = mesh.vertices

    # ── Vertex colours (sRGB, gamma-corrected on export) ─────────────────
    colors = np.zeros((len(verts_arr), 4), dtype=np.uint8)
    for idx, (x, y, z) in enumerate(verts_arr):
        ny = y / 1.26
        phi = math.atan2(z, x)
        if phi < 0:
            phi += 2 * math.pi
        theta = math.acos(np.clip(ny, -1.0, 1.0))

        # Diagonal AV groove fat (narrow, dark gold)
        av_groove_th = 0.365*math.pi - 0.065*math.pi * math.cos(phi)
        av_fat = abs(theta - av_groove_th) < 0.10
        fat_factor = 0.0
        if av_fat:
            fat_factor = 0.23 * math.exp(-90.0 * (theta - av_groove_th)**2)

        patch_centers = (0.50*np.pi, 1.49*np.pi, 0.08*np.pi, 1.92*np.pi)
        for patch_phi in patch_centers:
            patch_d = min(abs(phi - patch_phi), 2*np.pi - abs(phi - patch_phi))
            patch = (0.66 * math.exp(-((theta - 0.455*np.pi) / 0.12)**2)
                     * math.exp(-(patch_d / 0.24)**2))
            fat_factor = max(fat_factor, patch)

        # Interventricular grooves: dark shadowed strip
        aiv_phi = 0.495*math.pi + 0.09*math.pi * max(0, (theta - 0.37*math.pi)/(0.63*math.pi))
        aiv_d = min(abs(phi - aiv_phi), 2*math.pi - abs(phi - aiv_phi))
        groove_dark = 0.40 * math.exp(-38.0 * aiv_d**2) * max(0, (theta - 0.35*math.pi)/(0.65*math.pi))

        piv_phi = 1.52*math.pi
        piv_d = min(abs(phi - piv_phi), 2*math.pi - abs(phi - piv_phi))
        groove_dark += 0.48 * math.exp(-30.0 * piv_d**2) * max(0, (theta - 0.35*math.pi)/(0.65*math.pi))
        groove_dark = min(groove_dark, 0.55)

        # Myocardium base sRGB: deep red-brown
        mr, mg, mb = 151, 61, 58
        grain = (5.0 * math.sin(x*8 + z*6)
                 + 4.0 * math.sin(y*12 + x*4)
                 + 3.0 * math.cos(z*19 - y*7)
                 + 3.0 * math.sin(x*34 + y*27 - z*31)
                 + 2.0 * math.cos(y*43 - x*28 + z*33))
        mr = max(0, min(255, int(mr + grain + 10*ny)))
        mg = max(0, min(255, int(mg + grain*0.55 + 7*ny)))
        mb = max(0, min(255, int(mb + grain*0.42 + 5*ny)))

        # Groove darkening
        mr = int(mr * (1.0 - groove_dark))
        mg = int(mg * (1.0 - groove_dark * 0.8))
        mb = int(mb * (1.0 - groove_dark * 0.8))

        # Posterior interatrial sulcus
        ia_d = min(abs(phi - 1.46*np.pi), 2*np.pi - abs(phi - 1.46*np.pi))
        ia_dark = 0.24 * math.exp(-48.0 * ia_d**2) * math.exp(
            -((theta - 0.22*np.pi)**2) / (2*(0.15*np.pi)**2))
        mr = int(mr * (1.0 - ia_dark))
        mg = int(mg * (1.0 - ia_dark))
        mb = int(mb * (1.0 - ia_dark))

        # Fat overlay: narrow warm-tan
        fr, fg, fb = 215, 180, 118
        r = int(mr*(1-fat_factor) + fr*fat_factor)
        g = int(mg*(1-fat_factor) + fg*fat_factor)
        b = int(mb*(1-fat_factor) + fb*fat_factor)
        colors[idx] = [r, g, b, 255]

    mesh.visual.vertex_colors = colors
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# GREAT VESSELS – redesigned
# ─────────────────────────────────────────────────────────────────────────────

def build_great_vessels(heart):
    """
    Great vessels: aorta arch (dark #B71C1C, tall), pulmonary trunk
    in front of aorta, L/R pulmonary arteries, SVC/IVC flush to RA,
    four pulmonary veins entering left atrium posteriorly.
    Returns list of (name, mesh, srgb_rgba).
    """
    vessels = []

    # sRGB colors
    AORTA_C = (166, 61, 53, 255)
    PULM_C  = (184, 83, 72, 255)
    SVC_C   = (104, 91, 130, 255)
    PV_C    = (183, 91, 82, 255)

    # ── Ascending aorta → aortic arch (tall, dark, thick) ────────────────
    aorta_cp = [
        [ 0.10, 0.52, 0.20],   # aortic root
        [ 0.14, 0.92, 0.16],   # ascending aorta
        [ 0.12, 1.40, 0.08],   # upper ascending
        [-0.02, 1.78, -0.04],  # arch peak
        [-0.30, 1.82, -0.22],  # arch leftward
        [-0.52, 1.64, -0.44],  # start of descending
        [-0.60, 1.28, -0.62],  # descending aorta
        [-0.58, 0.96, -0.68],  # thoracic aorta behind the left atrium
        [-0.50, 0.64, -0.68],  # descending segment and open inferior end
    ]
    aorta_path = catmull_rom(aorta_cp, 12)
    n = len(aorta_path)
    aorta = create_hollow_tube(aorta_path,
                               outer_r=np.interp(np.linspace(0, 1, n),
                                                 [0.0, 0.62, 1.0],
                                                 [0.26, 0.22, 0.12]),
                               inner_r=np.interp(np.linspace(0, 1, n),
                                                 [0.0, 0.62, 1.0],
                                                 [0.20, 0.17, 0.085]),
                               num_sides=28)
    vessels.append(('Aorta', aorta, AORTA_C))

    # Three arch branches (brachiocephalic, L carotid, L subclavian)
    arch_branches = [
        ('Brachiocephalic', [[0.08, 1.65, 0.04], [0.26, 1.91, 0.10], [0.33, 2.08, 0.27], [0.34, 2.10, 0.40]], 0.092, 0.066),
        ('L_Carotid',       [[-0.04, 1.72, -0.02], [0.00, 1.96, 0.04], [0.00, 2.10, 0.22], [-0.01, 2.12, 0.36]], 0.078, 0.055),
        ('L_Subclavian',    [[-0.22, 1.74, -0.18], [-0.30, 1.95, -0.10], [-0.36, 2.08, 0.08], [-0.38, 2.10, 0.23]], 0.073, 0.051),
    ]
    for name, cp, ro, ri in arch_branches:
        p = catmull_rom(cp, 8)
        b = create_hollow_tube(p, outer_r=ro, inner_r=ri, num_sides=20)
        vessels.append((name, b, AORTA_C))

    # ── Pulmonary trunk: in FRONT of aorta (Z+), left of center ──────────
    pt_cp = [
        [-0.16, 0.50, 0.42],   # pulmonary valve (anterior to aorta)
        [-0.20, 0.88, 0.38],
        [-0.18, 1.24, 0.22],
        [-0.10, 1.46, 0.06],   # bifurcation into L/R
    ]
    pt_path = catmull_rom(pt_cp, 10)
    n = len(pt_path)
    pt = create_hollow_tube(pt_path,
                            outer_r=np.linspace(0.24, 0.19, n),
                            inner_r=np.linspace(0.18, 0.14, n),
                            num_sides=26)
    vessels.append(('PulmonaryTrunk', pt, PULM_C))

    # Left and right pulmonary arteries
    lpa_cp = [[-0.10, 1.46, 0.06], [-0.40, 1.40, -0.16], [-0.74, 1.28, -0.34]]
    rpa_cp = [[-0.10, 1.46, 0.06], [ 0.22, 1.40, -0.10], [ 0.56, 1.28, -0.26]]
    for name, cp in [('L_PulmArt', lpa_cp), ('R_PulmArt', rpa_cp)]:
        p = catmull_rom(cp, 8)
        b = create_hollow_tube(p, outer_r=0.164, inner_r=0.122, num_sides=22)
        vessels.append((name, b, PULM_C))

    # ── SVC: flush to right atrium (X+ side) ─────────────────────────────
    svc_cp = [[0.54, 0.50, 0.08], [0.58, 0.96, 0.06], [0.60, 1.34, 0.06], [0.60, 1.52, 0.20], [0.58, 1.55, 0.34]]
    svc_path = catmull_rom(svc_cp, 8)
    svc = create_hollow_tube(svc_path, outer_r=0.182, inner_r=0.138, num_sides=24)
    vessels.append(('SVC', svc, SVC_C))

    # ── IVC: enters RA from below ─────────────────────────────────────────
    ivc_cp = [[0.50, -0.48, -0.08], [0.52, -0.82, -0.18], [0.48, -1.10, -0.30], [0.44, -1.18, -0.38]]
    ivc_path = catmull_rom(ivc_cp, 6)
    ivc = create_hollow_tube(ivc_path, outer_r=0.142, inner_r=0.108, num_sides=22)
    vessels.append(('IVC', ivc, SVC_C))

    # Posterior pulmonary-vein ostia sit in the atrial wall and open outward.
    pv_sites = [
        ('L_PulmVein_Superior', [-0.40, 0.68, -0.55]),
        ('L_PulmVein_Inferior', [-0.40, 0.38, -0.55]),
        ('R_PulmVein_Superior', [0.30, 0.68, -0.55]),
        ('R_PulmVein_Inferior', [0.30, 0.38, -0.55]),
    ]
    heart_vertices = heart.vertices
    heart_normals = heart.vertex_normals
    for name, site in pv_sites:
        site = np.asarray(site, dtype=float)
        delta = heart_vertices - site
        nearest = int(np.argmin(np.einsum('ij,ij->i', delta, delta)))
        surface = heart_vertices[nearest]
        normal = heart_normals[nearest]
        port = np.array([
            surface - normal * 0.14,
            surface - normal * 0.035,
            surface + normal * 0.12,
            surface + normal * 0.34,
        ])
        path = catmull_rom(port, 6)
        count = len(path)
        vessel = create_hollow_tube(
            path,
            outer_r=np.linspace(0.095, 0.112, count),
            inner_r=np.linspace(0.064, 0.078, count),
            num_sides=22,
        )
        vessels.append((name, vessel, PV_C))

    return vessels


# ─────────────────────────────────────────────────────────────────────────────
# CORONARY VESSELS – surface-seated trunks and branches
# ─────────────────────────────────────────────────────────────────────────────

def _surface_curve(ctrl_pts, heart, samples_per_seg=8, lift=0.025):
    """Keep a smoothed coronary centerline seated against the myocardium."""
    vertices = heart.vertices
    normals = heart.vertex_normals

    def seat(points):
        seated = []
        for point in points:
            distances = np.sum((vertices - point) ** 2, axis=1)
            nearest = int(np.argmin(distances))
            normal = normals[nearest]
            signed_distance = np.dot(point - vertices[nearest], normal)
            seated.append(point + normal * (lift - signed_distance))
        return np.asarray(seated)

    smooth_path = catmull_rom(seat(np.asarray(ctrl_pts, dtype=float)),
                              samples_per_seg)
    return seat(smooth_path)


def build_coronary_vessels(heart):
    """
    Returns list of (name, mesh, srgb_rgba).
    Each main artery and branch follows the myocardium surface.
    """
    parts = []
    ART  = (179, 70, 59, 255)
    VEIN = (115, 91, 130, 255)

    # ────────────────────────────────────────────────────────────────────
    # LAD – left anterior descending
    # ────────────────────────────────────────────────────────────────────
    lad_cp = [
        [-0.08, 0.44, 0.58],    # left main bifurcation (at aorta base, surface Z≈0.57)
        [-0.06, 0.18, 0.64],    # upper anterior groove (surface Z≈0.58, +0.06 above)
        [-0.08,-0.04, 0.62],    # mid-upper groove
        [-0.10,-0.30, 0.60],    # mid groove (surface Z≈0.55)
        [-0.16,-0.60, 0.54],    # lower third (surface Z≈0.48, +0.06)
        [-0.22,-0.90, 0.44],    # lower (surface Z≈0.38, +0.06)
        [-0.30,-1.18, 0.32],    # near apex (surface Z≈0.27, +0.05)
        [-0.38,-1.34, 0.20],    # at apex (surface Z≈0.17, +0.03)
        [-0.42,-1.38, 0.06],    # just past apex
        [-0.36,-1.28,-0.14],    # wraps to inferior
    ]
    lad_path = _surface_curve(lad_cp, heart, 9)
    n = len(lad_path)
    lad_mesh = create_solid_tube(lad_path,
                                  radius=np.linspace(0.052, 0.020, n),
                                  num_sides=20)
    parts.append(('LAD', lad_mesh, ART))

    # LAD branches: D1, D2, D3 diagonals + apical
    lad_branches = [
        # D1: large first diagonal – sweeps left across anterior LV
        ('LAD_D1', [[-0.08,-0.04, 0.62],
                    [-0.30,-0.18, 0.58],
                    [-0.58,-0.34, 0.46],
                    [-0.76,-0.48, 0.34]], 0.048),
        # D2: second diagonal
        ('LAD_D2', [[-0.16,-0.60, 0.54],
                    [-0.40,-0.74, 0.44],
                    [-0.62,-0.88, 0.30]], 0.038),
        # D3: third diagonal
        ('LAD_D3', [[-0.22,-0.90, 0.44],
                    [-0.44,-1.00, 0.34]], 0.030),
        # Apical ramus
        ('LAD_Apical', [[-0.30,-1.18, 0.32],
                        [-0.48,-1.24, 0.24]], 0.024),
    ]
    for name, cp, r in lad_branches:
        p = _surface_curve(cp, heart, 6)
        m = create_solid_tube(p, radius=np.linspace(r, r * 0.48, len(p)), num_sides=14)
        parts.append((name, m, ART))

    # ────────────────────────────────────────────────────────────────────
    # LCX – left circumflex
    # Runs in the left AV groove (Y≈0.46, curving left then posterior)
    # Surface at AV groove: X varies, Z decreases as V moves posterior
    # ────────────────────────────────────────────────────────────────────
    lcx_cp = [
        [-0.08, 0.44, 0.58],    # left main (same start as LAD)
        [-0.30, 0.42, 0.55],    # enter left AV groove
        [-0.56, 0.36, 0.42],    # further left (Z drops as V increases)
        [-0.78, 0.22, 0.22],    # lateral AV groove
        [-0.92, 0.06, 0.00],    # posterolateral
        [-0.90,-0.10,-0.24],    # posterior AV groove
        [-0.78,-0.22,-0.50],    # posterior
        [-0.44,-0.28,-0.68],    # near posterior IVG
    ]
    lcx_path = _surface_curve(lcx_cp, heart, 9)
    n = len(lcx_path)
    lcx_mesh = create_solid_tube(lcx_path,
                                  radius=np.linspace(0.050, 0.019, n),
                                  num_sides=20)
    parts.append(('LCX', lcx_mesh, ART))

    # LCX branches: obtuse marginals
    lcx_branches = [
        # OM1: large first OM – sweeps down left lateral wall
        ('LCX_OM1', [[-0.78, 0.22, 0.22],
                     [-0.88,-0.12, 0.28],
                     [-0.84,-0.48, 0.22],
                     [-0.70,-0.76, 0.14]], 0.050),
        # OM2
        ('LCX_OM2', [[-0.90,-0.10,-0.24],
                     [-0.84,-0.40,-0.10],
                     [-0.72,-0.68, 0.04]], 0.038),
        # Posterolateral
        ('LCX_PL',  [[-0.78,-0.22,-0.50],
                     [-0.52,-0.38,-0.58],
                     [-0.24,-0.52,-0.60]], 0.030),
        # Atrial branch
        ('LCX_Atrial', [[-0.56, 0.36, 0.42],
                        [-0.50, 0.52, 0.34]], 0.024),
    ]
    for name, cp, r in lcx_branches:
        p = _surface_curve(cp, heart, 6)
        m = create_solid_tube(p, radius=np.linspace(r, r * 0.48, len(p)), num_sides=14)
        parts.append((name, m, ART))

    # ────────────────────────────────────────────────────────────────────
    # RCA – right coronary artery
    # Surface X on right side at AV groove: X≈0.54-0.57
    # Z decreases as V wraps from anterior → right → posterior
    # ────────────────────────────────────────────────────────────────────
    rca_cp = [
        [ 0.20, 0.50, 0.52],    # right coronary ostium
        [ 0.46, 0.42, 0.54],    # proximal right AV groove, just anterior
        [ 0.62, 0.28, 0.42],    # right lateral, Z dropping
        [ 0.72, 0.10, 0.24],    # lateral AV groove
        [ 0.74,-0.12, 0.06],    # right, going posteroinferior
        [ 0.70,-0.36,-0.12],    # posteroinferior
        [ 0.60,-0.58,-0.32],    # posterior
        [ 0.40,-0.80,-0.54],    # crux
        [ 0.12,-0.90,-0.66],    # PDA start
        [-0.14,-0.96,-0.62],    # PDA toward apex
    ]
    rca_path = _surface_curve(rca_cp, heart, 9)
    n = len(rca_path)
    rca_mesh = create_solid_tube(rca_path,
                                  radius=np.linspace(0.052, 0.018, n),
                                  num_sides=20)
    parts.append(('RCA', rca_mesh, ART))

    # RCA branches
    rca_branches = [
        # AM1: acute marginal – sweeps down RV anterior surface
        ('RCA_AM1', [[ 0.72, 0.10, 0.24],
                     [ 0.66,-0.22, 0.38],
                     [ 0.56,-0.52, 0.38],
                     [ 0.42,-0.78, 0.30]], 0.048),
        # AM2
        ('RCA_AM2', [[ 0.70,-0.36,-0.12],
                     [ 0.62,-0.58, 0.08],
                     [ 0.48,-0.78, 0.20]], 0.036),
        # PDA: posterior descending
        ('RCA_PDA', [[ 0.12,-0.90,-0.66],
                     [-0.06,-0.96,-0.62],
                     [-0.22,-1.02,-0.52],
                     [-0.34,-1.08,-0.38]], 0.044),
        # RV branch (proximal)
        ('RCA_RV',  [[ 0.62, 0.28, 0.42],
                     [ 0.54, 0.12, 0.50],
                     [ 0.40,-0.14, 0.52]], 0.032),
    ]
    for name, cp, r in rca_branches:
        p = _surface_curve(cp, heart, 6)
        m = create_solid_tube(p, radius=np.linspace(r, r * 0.46, len(p)), num_sides=14)
        parts.append((name, m, ART))

    for index, (artery_path, fraction, dx, dy) in enumerate([
        (lad_path, 0.22, -0.18, 0.12),
        (lad_path, 0.38, 0.19, 0.12),
        (lad_path, 0.56, -0.19, -0.08),
        (lad_path, 0.74, 0.15, -0.08),
        (rca_path, 0.14, 0.16, 0.10),
        (rca_path, 0.27, -0.15, 0.10),
        (rca_path, 0.40, 0.14, -0.08),
    ]):
        origin = artery_path[round(fraction * (len(artery_path) - 1))]
        target = origin + np.array([dx, dy, 0.0])
        branch_path = _surface_curve([origin, target], heart, 6, lift=0.012)
        radius = np.linspace(0.014, 0.005, len(branch_path))
        parts.append((f'CoronaryDistal_{index + 1}',
                      create_solid_tube(branch_path, radius=radius, num_sides=10),
                      ART))

    # ────────────────────────────────────────────────────────────────────
    # GCV / Great cardiac vein – blue, slightly inset beside the LAD.
    # ────────────────────────────────────────────────────────────────────
    gcv_cp = [
        [-0.20,-0.68, 0.47],    # near apex, slightly left of LAD, −0.06 Z
        [-0.24,-0.38, 0.54],    # ascending
        [-0.22,-0.04, 0.56],    # upper anterior groove
        [-0.34, 0.36, 0.48],    # entering left AV groove
        [-0.62, 0.26, 0.30],    # lateral AV groove
        [-0.86, 0.12,-0.10],    # posterior AV groove
        [-0.68, 0.14,-0.42],    # posterior (follows LCX groove)
        [-0.20, 0.12,-0.52],    # coronary sinus → RA
    ]
    gcv_path = _surface_curve(gcv_cp, heart, 9, lift=0.012)
    n = len(gcv_path)
    gcv = create_solid_tube(gcv_path,
                             radius=np.linspace(0.040, 0.024, n),
                             num_sides=18)
    parts.append(('GCV', gcv, VEIN))

    # Posterior coronary sinus follows the AV groove and receives the GCV.
    sinus_cp = [
        [-0.78, 0.18, -0.36],
        [-0.62, 0.23, -0.49],
        [-0.36, 0.25, -0.60],
        [ 0.00, 0.25, -0.65],
        [ 0.36, 0.23, -0.60],
        [ 0.64, 0.17, -0.47],
        [ 0.78, 0.10, -0.31],
    ]
    sinus_path = _surface_curve(sinus_cp, heart, 8, lift=0.035)
    sinus = create_solid_tube(
        sinus_path,
        radius=np.linspace(0.060, 0.044, len(sinus_path)),
        num_sides=18,
    )
    parts.append(('CoronarySinus', sinus, VEIN))

    # Posterior descending artery and middle cardiac vein share the
    # interventricular sulcus from the crux toward the apex.
    pda_cp = [
        [ 0.02, 0.20, -0.65],
        [ 0.02,-0.10, -0.62],
        [ 0.00,-0.40, -0.56],
        [-0.10,-0.72, -0.45],
        [-0.23,-1.02, -0.30],
        [-0.36,-1.20, -0.16],
    ]
    pda_path = _surface_curve(pda_cp, heart, 8, lift=0.035)
    pda = create_solid_tube(
        pda_path,
        radius=np.linspace(0.037, 0.016, len(pda_path)),
        num_sides=16,
    )
    parts.append(('PosteriorDescendingArtery', pda, ART))

    mcv_cp = [
        [ 0.11, 0.15, -0.64],
        [ 0.11,-0.12, -0.62],
        [ 0.08,-0.42, -0.55],
        [-0.02,-0.73, -0.44],
        [-0.14,-1.00, -0.30],
        [-0.26,-1.16, -0.17],
    ]
    mcv_path = _surface_curve(mcv_cp, heart, 8, lift=0.028)
    mcv = create_solid_tube(
        mcv_path,
        radius=np.linspace(0.029, 0.013, len(mcv_path)),
        num_sides=16,
    )
    parts.append(('MiddleCardiacVein', mcv, VEIN))

    for index, (fraction, dx, dy) in enumerate([
        (0.18, -0.16, -0.03),
        (0.34,  0.17, -0.02),
        (0.51, -0.16, -0.04),
        (0.67,  0.14, -0.03),
        (0.82, -0.10, -0.02),
        (0.23,  0.21,  0.07),
        (0.42, -0.20,  0.06),
        (0.60,  0.19,  0.05),
        (0.76, -0.17,  0.06),
        (0.91,  0.12,  0.04),
    ]):
        origin = pda_path[round(fraction * (len(pda_path) - 1))]
        target = origin + np.array([dx, dy, 0.0])
        branch = _surface_curve([origin, target], heart, 6, lift=0.025)
        parts.append((
            f'PosteriorVentricularBranch_{index + 1}',
            create_solid_tube(
                branch,
                radius=np.linspace(0.012, 0.0045, len(branch)),
                num_sides=10,
            ),
            ART,
        ))

    for index, (fraction, dx, dy) in enumerate([
        (0.28, -0.14,  0.06),
        (0.48,  0.15,  0.05),
        (0.68, -0.14,  0.04),
        (0.42,  0.13, -0.08),
        (0.64,  0.11, -0.07),
    ]):
        origin = mcv_path[round(fraction * (len(mcv_path) - 1))]
        target = origin + np.array([dx, dy, 0.0])
        branch = _surface_curve([origin, target], heart, 6, lift=0.020)
        parts.append((
            f'MiddleCardiacBranch_{index + 1}',
            create_solid_tube(
                branch,
                radius=np.linspace(0.010, 0.0038, len(branch)),
                num_sides=10,
            ),
            VEIN,
        ))

    return parts   # [(name, mesh, srgb_rgba), ...]

# ─────────────────────────────────────────────────────────────────────────────
# GLB EXPORTER  (PBR metallic-roughness, correct gamma, mesh names)
# ─────────────────────────────────────────────────────────────────────────────

def _pack_glb(entries):
    """
    Build a glTF 2.0 binary (.glb).

    entries: list of  (mesh, base_rgba_linear_0_1, roughness, metallic,
                       emissive_linear_0_1, name_str)
    """
    accessors, buffer_views, meshes_json, materials_json, nodes = [], [], [], [], []
    buf = bytearray()

    def _pad(b, align=4):
        rem = len(b) % align
        if rem:
            b += b'\x00' * (align - rem)

    for mi, (mesh, base_rgba, roughness, metallic, emissive, mesh_name) in enumerate(entries):
        mesh.fix_normals()
        verts  = mesh.vertices.astype(np.float32)
        # Flip X so anatomical right (RA/SVC/RCA) is on viewer's right
        # when camera is at +Z. Also flip face winding to keep normals outward.
        verts[:, 0] = -verts[:, 0]
        norms  = mesh.vertex_normals.astype(np.float32)
        norms[:, 0] = -norms[:, 0]
        faces_raw = mesh.faces.astype(np.uint32)
        # Reverse triangle winding to correct normals after X-flip
        faces  = faces_raw[:, [0, 2, 1]]

        has_colors = (mesh.visual.kind == 'vertex'
                      and mesh.visual.vertex_colors is not None
                      and len(mesh.visual.vertex_colors) == len(verts))

        # Material
        mat = {
            "pbrMetallicRoughness": {
                "baseColorFactor": list(base_rgba),
                "metallicFactor":  metallic,
                "roughnessFactor": roughness,
            },
            "emissiveFactor": list(emissive),
            "doubleSided": True,
            "name": f"mat_{mesh_name}",
        }
        materials_json.append(mat)

        # --- positions ---
        pos_bv = len(buffer_views)
        pos_data = verts.tobytes()
        buffer_views.append({
            "buffer": 0, "byteOffset": len(buf),
            "byteLength": len(pos_data), "target": 34962,
        })
        buf += pos_data; _pad(buf)

        pos_acc = len(accessors)
        accessors.append({
            "bufferView": pos_bv, "componentType": 5126,
            "count": len(verts), "type": "VEC3",
            "max": verts.max(axis=0).tolist(),
            "min": verts.min(axis=0).tolist(),
        })

        # --- normals ---
        nrm_bv = len(buffer_views)
        nrm_data = norms.tobytes()
        buffer_views.append({
            "buffer": 0, "byteOffset": len(buf),
            "byteLength": len(nrm_data), "target": 34962,
        })
        buf += nrm_data; _pad(buf)

        nrm_acc = len(accessors)
        accessors.append({
            "bufferView": nrm_bv, "componentType": 5126,
            "count": len(norms), "type": "VEC3",
        })

        # --- vertex colours (sRGB → linear) ---
        col_acc = None
        if has_colors:
            vc_srgb = mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255.0
            vc = vc_srgb.copy()
            vc[:, :3] = np.where(vc_srgb[:, :3] <= 0.04045,
                                  vc_srgb[:, :3] / 12.92,
                                  ((vc_srgb[:, :3] + 0.055) / 1.055) ** 2.4)
            col_bv = len(buffer_views)
            col_data = vc.astype(np.float32).tobytes()
            buffer_views.append({
                "buffer": 0, "byteOffset": len(buf),
                "byteLength": len(col_data), "target": 34962,
            })
            buf += col_data; _pad(buf)

            col_acc = len(accessors)
            accessors.append({
                "bufferView": col_bv, "componentType": 5126,
                "count": len(vc), "type": "VEC4",
            })

        # --- indices ---
        idx_bv = len(buffer_views)
        idx_data = faces.tobytes()
        buffer_views.append({
            "buffer": 0, "byteOffset": len(buf),
            "byteLength": len(idx_data), "target": 34963,
        })
        buf += idx_data; _pad(buf)

        idx_acc = len(accessors)
        accessors.append({
            "bufferView": idx_bv, "componentType": 5125,
            "count": faces.size, "type": "SCALAR",
        })

        attributes = {"POSITION": pos_acc, "NORMAL": nrm_acc}
        if col_acc is not None:
            attributes["COLOR_0"] = col_acc

        meshes_json.append({
            "name": mesh_name,
            "primitives": [{
                "attributes": attributes,
                "indices": idx_acc,
                "material": mi,
            }],
        })
        nodes.append({"mesh": mi, "name": mesh_name})

    gltf = {
        "asset": {"version": "2.0", "generator": "HeartModelGen-v2"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes_json,
        "materials": materials_json,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buf)}],
    }
    json_bytes = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(json_bytes) % 4:
        json_bytes += b' '
    buf_bytes = bytes(buf)
    while len(buf_bytes) % 4:
        buf_bytes += b'\x00'

    total = 12 + 8 + len(json_bytes) + 8 + len(buf_bytes)
    glb = (struct.pack('<III', 0x46546C67, 2, total)
           + struct.pack('<II', len(json_bytes), 0x4E4F534A)
           + json_bytes
           + struct.pack('<II', len(buf_bytes), 0x004E4942)
           + buf_bytes)
    return glb


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def generate_heart_model():
    print("Generating anatomically accurate human heart model (v2)...")

    # 1. Heart body
    heart = generate_heart_body()

    # 2. Great vessels
    vessel_parts = build_great_vessels(heart)  # [(name, mesh, srgb_rgba), ...]

    # 3. Coronary arteries + GCV
    coronary_parts = build_coronary_vessels(heart)  # [(name, mesh, srgb_rgba), ...]

    all_meshes = ([heart]
                  + [m for _, m, _ in vessel_parts]
                  + [m for _, m, _ in coronary_parts])
    total_tris = sum(len(m.faces) for m in all_meshes)
    print(f"  Total triangles: {total_tris}")

    # -- Material entries: (mesh, base_color_linear, roughness, metallic, emissive_linear, name)

    # Heart body: vertex-colored, very matte tissue
    entries = [(heart,
                [1.0, 1.0, 1.0, 1.0],   # vertex colors drive everything
                0.54, 0.0,
                [0.0, 0.0, 0.0],         # no emissive (vertex colors are gamma-correct)
                'HeartBody')]

    # Great vessels: hollow tubes, gamma-linearized colors
    for name, m, rgba in vessel_parts:
        bc = _rgba_lin(*rgba)
        em = [bc[0]*0.035, bc[1]*0.018, bc[2]*0.035]
        entries.append((m, bc, 0.62, 0.0, em, name))

    # Coronary arteries: thick tubes, #E53935, emissive glow
    # Veins: blue, slight emissive
    for name, m, rgba in coronary_parts:
        bc = _rgba_lin(*rgba)
        is_vein = rgba[2] > rgba[0]  # blue channel dominant
        if is_vein:
            em = [0.0, 0.0, bc[2]*0.08]
        else:
            em = [bc[0]*0.10, bc[1]*0.015, bc[2]*0.01]
        entries.append((m, bc, 0.58, 0.0, em, name))

    return entries, total_tris


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    entries, tri_count = generate_heart_model()

    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "public", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "heart.glb")

    glb_data = _pack_glb(entries)
    with open(out_file, "wb") as f:
        f.write(glb_data)

    kb = len(glb_data) / 1024
    print(f"Exported -> {out_file}  ({kb:.1f} KB,  {tri_count} triangles)")
