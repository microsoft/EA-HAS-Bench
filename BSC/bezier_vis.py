import bezier
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import interpolate
from scipy.special import comb as n_over_k
import glob, os
import cv2

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import matplotlib.pyplot as plt
import math
import numpy as np
import random
# from scipy.optimize import leastsq
import torch
from torch import nn
from torch.nn import functional as F

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

from shapely.geometry import *
from PIL import Image
import time


class Cubic_Bezier(nn.Module):
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Cubic_Bezier, self).__init__()
        self.x1 = nn.Parameter(torch.as_tensor(ctps[0], dtype=torch.float64))
        self.x2 = nn.Parameter(torch.as_tensor(ctps[2], dtype=torch.float64))
        self.y1 = nn.Parameter(torch.as_tensor(ctps[1], dtype=torch.float64))
        self.y2 = nn.Parameter(torch.as_tensor(ctps[3], dtype=torch.float64))
        self.x0 = ps[0, 0]
        self.x3 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y3 = ps[-1, 1]
        self.inner_ps = torch.as_tensor(ps[1:-1, :], dtype=torch.float64)
        self.t = torch.as_tensor(np.linspace(0, 1, 81))

    def forward(self):
        x0, x1, x2, x3, y0, y1, y2, y3 = self.control_points()
        t = self.t
        bezier_x = (1-t)*((1-t)*((1-t)*x0+t*x1)+t*((1-t)*x1+t*x2))+t*((1-t)*((1-t)*x1+t*x2)+t*((1-t)*x2+t*x3))
        bezier_y = (1-t)*((1-t)*((1-t)*y0+t*y1)+t*((1-t)*y1+t*y2))+t*((1-t)*((1-t)*y1+t*y2)+t*((1-t)*y2+t*y3))
        bezier = torch.stack((bezier_x, bezier_y), dim=1)
        diffs = bezier.unsqueeze(0) - self.inner_ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.y0, self.y1, self.y2, self.y3

    def control_points_f(self):
        return self.x0, self.x1.item(), self.x2.item(), self.x3, self.y0, self.y1.item(), self.y2.item(), self.y3



class Quartic_Bezier(nn.Module):
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Quartic_Bezier, self).__init__()
        self.x1 = nn.Parameter(torch.as_tensor(ctps[0], dtype=torch.float64))
        self.x2 = nn.Parameter(torch.as_tensor(ctps[2], dtype=torch.float64))
        self.x3 = nn.Parameter(torch.as_tensor(ctps[4], dtype=torch.float64))
        self.y1 = nn.Parameter(torch.as_tensor(ctps[1], dtype=torch.float64))
        self.y2 = nn.Parameter(torch.as_tensor(ctps[3], dtype=torch.float64))
        self.y3 = nn.Parameter(torch.as_tensor(ctps[5], dtype=torch.float64))

        self.x0 = ps[0, 0]
        self.x4 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y4 = ps[-1, 1]
        self.inner_ps = torch.as_tensor(ps[1:-1, :], dtype=torch.float64)
        self.t = torch.as_tensor(np.linspace(0, 1, 100))

    def forward(self):
        x0, x1, x2, x3, x4, y0, y1, y2, y3, y4 = self.control_points()
        t = self.t
        bezier_x = Bezier_Point(t, [x0, x1, x2, x3, x4])
        bezier_y = Bezier_Point(t, [y0, y1, y2, y3, y4])
        bezier = torch.stack((bezier_x, bezier_y), dim=1)
        diffs = bezier.unsqueeze(0) - self.inner_ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.x4, self.y0, self.y1, self.y2, self.y3, self.y4

    def control_points_f(self):
        return self.x0, self.x1.item(), self.x2.item(), self.x3.item(), self.x4, self.y0, self.y1.item(), self.y2.item(), self.y3.item(), self.y4


def Bezier_TwoPoints(t, P1, P2):
    Q1 = (1 - t) * P1 + t * P2
    return Q1

def Bezier_Points(t, points):
    """
    Returns a list of points interpolated by the Bezier process
    INPUTS:
        t            float/int; a parameterisation.
        points       list of numpy arrays; points.
    OUTPUTS:
        newpoints    list of numpy arrays; points.
    """
    newpoints = []
    #print("points =", points, "\n")
    for i1 in range(0, len(points) - 1):
        #print("i1 =", i1)
        #print("points[i1] =", points[i1])
        newpoints += [Bezier_TwoPoints(t, points[i1], points[i1 + 1])]
        #print("newpoints  =", newpoints, "\n")
    return newpoints


def Bezier_Point(t, points):
    """
    Returns a point interpolated by the Bezier process
    INPUTS:
        t            float/int; a parameterisation.
        points       list of numpy arrays; points.
    OUTPUTS:
        newpoint     numpy array; a point.
    """
    newpoints = points
    #print("newpoints = ", newpoints)
    while len(newpoints) > 1:
        newpoints = Bezier_Points(t, newpoints)
        #print("newpoints in loop = ", newpoints)
    return newpoints[0]

class Quintic_Bezier(nn.Module):
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Quintic_Bezier, self).__init__()
        self.x1 = nn.Parameter(torch.as_tensor(ctps[0], dtype=torch.float64))
        self.x2 = nn.Parameter(torch.as_tensor(ctps[2], dtype=torch.float64))
        self.x3 = nn.Parameter(torch.as_tensor(ctps[4], dtype=torch.float64))
        self.x4 = nn.Parameter(torch.as_tensor(ctps[6], dtype=torch.float64))
        
        self.y1 = nn.Parameter(torch.as_tensor(ctps[1], dtype=torch.float64))
        self.y2 = nn.Parameter(torch.as_tensor(ctps[3], dtype=torch.float64))
        self.y3 = nn.Parameter(torch.as_tensor(ctps[5], dtype=torch.float64))
        self.y4 = nn.Parameter(torch.as_tensor(ctps[7], dtype=torch.float64))

        self.x0 = ps[0, 0]
        self.x5 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y5 = ps[-1, 1]
        self.inner_ps = torch.as_tensor(ps[1:-1, :], dtype=torch.float64)
        self.t = torch.as_tensor(np.linspace(0, 1, 100))
    
    def forward(self):
        x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5 = self.control_points()
        t = self.t
        bezier_x = Bezier_Point(t, [x0, x1, x2, x3, x4, x5])
        bezier_y = Bezier_Point(t, [y0, y1, y2, y3, y4, y5])
        bezier = torch.stack((bezier_x, bezier_y), dim=1)
        diffs = bezier.unsqueeze(0) - self.inner_ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.x4, self.x5, self.y0, self.y1, self.y2, self.y3, self.y4, self.y5

    def control_points_f(self):
        return self.x0, self.x1.item(), self.x2.item(), self.x3.item(), self.x4.item(), self.x5, self.y0, self.y1.item(), self.y2.item(), self.y3.item(), self.y4.item(), self.y5


class Sextic_Bezier(nn.Module):
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Sextic_Bezier, self).__init__()
        self.x1 = nn.Parameter(torch.as_tensor(ctps[0], dtype=torch.float64))
        self.x2 = nn.Parameter(torch.as_tensor(ctps[2], dtype=torch.float64))
        self.x3 = nn.Parameter(torch.as_tensor(ctps[4], dtype=torch.float64))
        self.x4 = nn.Parameter(torch.as_tensor(ctps[6], dtype=torch.float64))
        self.x5 = nn.Parameter(torch.as_tensor(ctps[8], dtype=torch.float64))
        
        self.y1 = nn.Parameter(torch.as_tensor(ctps[1], dtype=torch.float64))
        self.y2 = nn.Parameter(torch.as_tensor(ctps[3], dtype=torch.float64))
        self.y3 = nn.Parameter(torch.as_tensor(ctps[5], dtype=torch.float64))
        self.y4 = nn.Parameter(torch.as_tensor(ctps[7], dtype=torch.float64))
        self.y5 = nn.Parameter(torch.as_tensor(ctps[9], dtype=torch.float64))


        self.x0 = ps[0, 0]
        self.x6 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y6 = ps[-1, 1]
        self.inner_ps = torch.as_tensor(ps[1:-1, :], dtype=torch.float64)
        self.t = torch.as_tensor(np.linspace(0, 1, 100))

    def forward(self):
        x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6 = self.control_points()
        t = self.t
        bezier_x = Bezier_Point(t, [x0, x1, x2, x3, x4, x5, x6])
        bezier_y = Bezier_Point(t, [y0, y1, y2, y3, y4, y5, y6])
        bezier = torch.stack((bezier_x, bezier_y), dim=1)
        diffs = bezier.unsqueeze(0) - self.inner_ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.y0, self.y1, self.y2, self.y3, self.y4, self.y5, self.y6

    def control_points_f(self):
        return self.x0, self.x1.item(), self.x2.item(), self.x3.item(), self.x4.item(), self.x5.item(), self.x6, self.y0, self.y1.item(), self.y2.item(), self.y3.item(), self.y4.item(), self.y5.item(), self.y6

def train(x, y, ctps, lr, name="Cubic"):
    x, y = np.array(x), np.array(y)
    ps = np.vstack((x, y)).transpose()
    if name == "Cubic":
        bezier = Cubic_Bezier(ps, ctps)
    elif name == "Quartic":
        bezier = Quartic_Bezier(ps, ctps)
    elif name == 'Quintic':
        bezier = Quintic_Bezier(ps, ctps)
    elif name == 'Sextic':
        bezier = Sextic_Bezier(ps, ctps)
    
    return bezier.control_points_f()


Mtk = lambda n, t, k: t**k * (1-t)**(n-k) * n_over_k(n,k)
Cubic_BezierCoeff = lambda ts: [[Mtk(3,t,k) for k in range(4)] for t in ts]
Quartic_BezierCoeff = lambda ts: [[Mtk(4,t,k) for k in range(5)] for t in ts]
Quintic_BezierCoeff = lambda ts: [[Mtk(5,t,k) for k in range(6)] for t in ts]
Sextic_BezierCoeff = lambda ts: [[Mtk(6,t,k) for k in range(7)] for t in ts]

def bezier_fit(x, y, name="Cubic"):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))
    if name == 'Cubic':
        Pseudoinverse = np.linalg.pinv(Cubic_BezierCoeff(t)) # (9,4) -> (4,9)
    elif name == 'Quartic':
        Pseudoinverse = np.linalg.pinv(Quartic_BezierCoeff(t))
    elif name == 'Quintic':
        Pseudoinverse = np.linalg.pinv(Quintic_BezierCoeff(t))
    elif name == 'Sextic':
        Pseudoinverse = np.linalg.pinv(Sextic_BezierCoeff(t))

    control_points = Pseudoinverse.dot(data)     # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1,:].flatten().tolist()
    return medi_ctp


def is_close_to_linev2(xs, ys, size, thres = 0.05):
        pts = []
        nor_pixel = int(size**0.5)
        for i in range(len(xs)):
                pts.append(Point([xs[i], ys[i]]))
        import itertools
        # iterate by pairs of points
        slopes = [(second.y-first.y)/(second.x-first.x) if not (second.x-first.x) == 0.0 else math.inf*np.sign((second.y-first.y)) for first, second in zip(pts, pts[1:])]
        st_slope = (ys[-1] - ys[0])/(xs[-1] - xs[0])
        max_dis = ((ys[-1] - ys[0])**2 +(xs[-1] - xs[0])**2)**(0.5)

        diffs = abs(slopes - st_slope)
        score = diffs.sum() * max_dis/nor_pixel

        if score < thres:
                return 0.0
        else:
                return 3.0


def get_points(pred, length, name="Cubic"):
    if name=="Cubic":
        y0, y3,  x1, x2, y1, y2 = pred
        control_points = np.array([1, x1, x2, length, y0, y1, y2, y3]).reshape(2, -1)
    elif name=="Quartic":
        y0, y4, x1, x2, x3, y1, y2, y3 = pred
        control_points = np.array([1, x1, x2, x3, length, y0, y1, y2, y3, y4]).reshape(2, -1)
    elif name=="Quintic":
        y0, y5, x1, x2, x3, x4, y1, y2, y3, y4 = pred
        control_points = np.array([1, x1, x2, x3, x4, length, y0, y1, y2, y3, y4, y5]).reshape(2, -1)
    elif name=="Sextic":
        y0, y6,  x1, x2, x3, x4, x5, y1, y2, y3, y4, y5 = pred
        control_points = np.array([1, x1, x2, x3, x4, x5, length, y0, y1, y2, y3, y4, y5, y6]).reshape(2, -1)
    # print(control_points)
    curve1 = bezier.Curve.from_nodes(control_points)
    points = [y0]

    for j in range(2, length):
        nodes2 = np.asfortranarray([
            [j,j,j],
            [-10, 50, 100 ],
        ])
        curve2 = bezier.Curve.from_nodes(nodes2)
        intersections = curve1.intersect(curve2)
        s_vals = intersections[0, :]
        if s_vals.shape[0] == 1:
            point = curve1.evaluate_multi(s_vals)
        elif s_vals.shape[0] > 1:
            point = curve1.evaluate(s_vals[0])
        elif s_vals.shape[0] == 0:
            point = curve1.evaluate(0.5)
        else:
            print(s_vals)
            point = curve1.evaluate(s_vals)

        points.append(point[1][0])
    points.append(pred[1])
    return points


if __name__ == "__main__":
    with open("/RegNet/checkpoint/sweeps/cifar/mb_v0.4/sweep.json", "r") as f:
        data = json.load(f)


    lc = [d['test_ema_epoch']['top1_err']  for d in data]


    name = "Quartic"
    name = "Quintic"
    name = "Sextic"

    for i in range(len(lc)):
        y = lc[i]
        x_data = np.array([i + 1 for i in range(len(y))])
        y_data = np.array(y)

        init_control_points = bezier_fit(x_data, y_data, name)
        size = 200 * 200
        learning_rate = is_close_to_linev2(x_data, y_data, size)
        control_points = np.array([train(x_data, y_data, init_control_points, learning_rate, name)]).reshape(2, -1)
        # print(control_points)

        curve1 = bezier.Curve.from_nodes(control_points)
        points = []
        for j in range(2, len(y)):
            nodes2 = np.asfortranarray([
                [j, j, j],
                [0, 50, 100 ],
            ])
            curve2 = bezier.Curve.from_nodes(nodes2)
            intersections = curve1.intersect(curve2)

            s_vals = intersections[0, :]
            
            if s_vals.shape[0] == 1:
                point = curve1.evaluate_multi(s_vals)
            elif s_vals.shape[0] == 3:
                point = curve1.evaluate(s_vals[0])
            else:
                print(s_vals)
                point = curve1.evaluate(s_vals)
            
            points.append(point[1][0])

        control_points = control_points.transpose()

        t_plot = np.linspace(0, 1, 100)
        Bezier_points = np.array(Sextic_BezierCoeff(t_plot)).dot(control_points)
        size = 10

        plt.plot(Bezier_points[:,0], Bezier_points[:,1], 'g-', label='fit', linewidth=1.0)
        # plt.scatter(Bezier_points[:,0], Bezier_points[:,1])

        plt.plot(control_points[:,0],
                    control_points[:,1], 'r.:', fillstyle='none', linewidth=1.0)

        plt.scatter(x_data, y_data, s=size)
        plt.plot(x_data, y_data, 'b', linewidth=1.0)

        plt.scatter([i for i in range(2, len(y))], points, s=size)
        # plt.axis('off')

        if not os.path.isdir('bezier_vis'):
                os.mkdir('bezier_vis')
        plt.savefig('bezier_vis/%s.jpg' %(i), bbox_inches='tight',dpi=400)
        plt.clf()
