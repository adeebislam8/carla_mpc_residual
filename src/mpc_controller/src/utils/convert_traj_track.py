# reads global_optimal trajectory csv, parses s_m; x_m; y_m; psi_rad; kappa_radpm;
# and saves as track.txt


import numpy as np
import csv
import scipy.interpolate as interp
import math
from nav_msgs.msg import Path

def parseGlobal(msg: Path):
    ss = []
    ds = []
    vs = []
    for pose in msg.poses:
        ss.append(pose.pose.position.x)
        ds.append(pose.pose.position.y)
        vs.append(pose.pose.position.z)
    
    d_spline = interp.CubicSpline(ss, ds)
    v_spline = interp.CubicSpline(ss, vs)
    # d_spline = interp.make_interp_spline(ss, ds, k=3)
    # v_spline = interp.make_interp_spline(ss, vs, k=3)

    ss = np.array(ss)

    # dense_s = np.linspace(ss[0], ss[-1], int(ss[-1]/0.1))
    dense_s = np.linspace(ss[0], ss[-1], int(ss[-1]/0.5))

    dd_ds = d_spline.derivative()(dense_s)
    dv_ds = v_spline.derivative()(dense_s)

    return dense_s, d_spline, v_spline, dd_ds, dv_ds

def parseReference(msg: Path):

    xs = []
    ys = []

    for pose in msg.poses:
        x, y = pose.pose.position.x, pose.pose.position.y
        xs.append(x)
        ys.append(y)

    x1, x2 = xs[0], xs[1]
    y1, y2 = ys[0], ys[1]
    dist_s = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    xe, xee = xs[-1], xs[-2]
    ye, yee = ys[-1], ys[-2]
    dist_e  = ((xe - xee) ** 2 + (ye - yee) ** 2) ** 0.5

    x_slope_start, x_slope_end = (x2 - x1) / dist_s, (xe - xee) / dist_e
    y_slope_start, y_slope_end = (y2 - y1) / dist_s, (ye - yee) / dist_e



    
    ds = [0]
    distance = 0
    
    for i in range(1, len(xs)):
        
        x_now, y_now = xs[i], ys[i]
        x_prv, y_prv = xs[i-1], ys[i-1]
        distance = math.sqrt((x_now - x_prv)**2 + (y_now - y_prv)**2)
        ds.append(distance + ds[-1])

    x_spline = interp.CubicSpline(ds, xs, bc_type=((1, x_slope_start), (1, x_slope_end)))
    y_spline = interp.CubicSpline(ds, ys, bc_type=((1, y_slope_start), (1, y_slope_end)))
    # x_spline = interp.make_interp_spline(ds, xs, k=3, bc_type=((1, x_slope_start), (1, x_slope_end)))
    # y_spline = interp.make_interp_spline(ds, ys, k=3, bc_type=((1, y_slope_start), (1, y_slope_end)))
    # Generate a dense list of 's' values for interpolation
    # x_spline = interp.CubicSpline(ds, xs, bc_type="clamped")
    # y_spline = interp.CubicSpline(ds, ys, bc_type="clamped")

    # we need constant no of points to ensure constant length of knots and coefficients
    # num_points = 1000
    # dense_s = np.linspace(ds[0], ds[-1], num_points) 

    density = 0.1
    dense_s = np.linspace(ds[0], ds[-1], int(ds[-1]/density)) # Change 1000 to the density you want

    # Get first derivatives
    dx_ds = x_spline.derivative()(dense_s)
    dy_ds = y_spline.derivative()(dense_s)

    # Get second derivatives
    dx2_ds2 = x_spline.derivative(nu=2)(dense_s)
    dy2_ds2 = y_spline.derivative(nu=2)(dense_s)

    # Compute phi (slope angle)
    phi = np.arctan2(dy_ds, dx_ds)

    # Compute kappa (curvature)
    kappa = (dx_ds * dy2_ds2 - dy_ds * dx2_ds2) / (dx_ds**2 + dy_ds**2)**(1.5)

    return x_spline, y_spline, ds[-1], dense_s, phi, kappa