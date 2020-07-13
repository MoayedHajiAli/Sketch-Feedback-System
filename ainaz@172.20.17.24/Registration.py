from LabeledObject import LabeledObject
from ObjectUtil import ObjectUtil
from Vector import Vector
from Point import Point
import autograd.numpy as np
from autograd import grad
import numpy as npn
from autograd.numpy.numpy_boxes import ArrayBox
from scipy.optimize import minimize


class Registration:

    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        self.p1, self.p2 = self.obj1.get_points(), self.obj2.get_points()
        self.x1 = np.array([float(p.x) for p in self.p1])
        self.y1 = np.array([float(p.y) for p in self.p1])
        self.x2 = np.array([float(p.x) for p in self.p2])
        self.y2 = np.array([float(p.y) for p in self.p2])

    # Transform a single point
    def transform(self, x, y, a11, a12, a13, a21, a22, a23):
        a = a11 * x + a12 * y + a13
        b = a21 * x + a22 * y + a23
        return a, b

    # calculate the turning angle based on three points coordincates
    def calc_turning(self, x0, y0, x1, y1, x2, y2) -> float:
        dot = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)
        det = (x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1)
        if dot == 0 or det == 0:
            return 0.0
        return np.arctan2(-det, -dot) * 180.0 / np.pi + 180

    # dissimilarity function for the two objects of the class
    def calc_dissimilarity(self, t):
        x = []
        y = []
        a11, a12, a13, a21, a22, a23 = t
        debug = False

        if debug:
            print("before trans", self.x2)
            print("before trans", self.y2)

        for i in range(len(self.x2)):
            x.append(a11 * self.x2[i] + a12 * self.y2[i] + a13)
            y.append(a21 * self.x2[i] + a22 * self.y2[i] + a23)

        cum1 = self.calc_turning(x[0] - 1, y[0], x[0], y[0], x[1], y[1]) * a11 / a11
        cum2 = self.calc_turning(self.x1[0] - 1, self.y1[0], self.x1[0], self.y1[0], self.x1[1],
                                 self.y1[1]) * a11 / a11

        ang = 360.0 - (cum2 - cum1)
        if ang > cum2 - cum1:
            ang = cum2 - cum1
        if debug:
            print(cum1, cum2, ang)
        tot = 0.0
        tot = (ang/180.0) ** 2
        # tot = ((x[0] - self.x1[0])**2 + (y[0] - self.y1[0])**2) * 0.01
        if debug:
            print(y[0], self.y1[0], tot)
        i = j = 0

        for _ in range(len(self.obj1)):

            if j + 2 < len(self.obj1):
                t1 = self.calc_turning(x[j], y[j], x[j + 1], y[j + 1], x[j + 2], y[j + 2]) * a11 / a11
                t2 = self.calc_turning(self.x1[i], self.y1[i], self.x1[i + 1], self.y1[i + 1], self.x1[i + 2],
                                       self.y1[i + 2]) * a11 / a11

                ang = 360.0 - (t2 - t1)
                if ang > t2 - t1:
                    ang = t2 - t1
                # tot += (ang/180.0) ** 2

                cum1 += t1
                cum2 += t2
                ang = 360.0 - (cum2 - cum1)
                if ang > cum2 - cum1:
                    ang = cum2 - cum1
                tot += (ang / 180.0) ** 2

            if i + 1 < len(self.obj1):
                ln1 = np.sqrt((x[j + 1] - x[j]) ** 2 + (y[j + 1] - y[j]) ** 2)
                ln2 = np.sqrt((self.x1[i + 1] - self.x1[j]) ** 2 + (self.y1[i + 1] - self.y1[i]) ** 2)
                #tot += (ln2 - ln1) ** 2
                if debug:
                    print("length", ln1, ln2, tot)

            mn = 10000000
            for k in range(len(x)):
                mn = min(mn, (x[k] - self.x1[i]) ** 2 + (y[k] - self.y1[i]) ** 2)
            tot += mn

            mn = 10000000
            for k in range(len(x)):
                mn = min(mn, (x[j] - self.x1[k]) ** 2 + (y[j] - self.y1[k]) ** 2)
            tot += mn

            # print(x[0], self.x1[0])
            # tot += ((x[j] - self.x1[i])**2 + (y[j] - self.y1[i])**2) * 0.01
            if debug:
                print(y[j], self.y1[i], tot)

            i = i + 1
            j = j + 1

        return tot

    def find_grad(self):
        return grad(self.calc_dissimilarity, argnum=(0))

    def optimize(self, rate=0.1):
        grad = self.find_grad()
        t = npn.random.rand(6)
        # t = npn.array([ 10.2085689,  -0.13153292,   5.02415388,  -1.66894684, -18.94775871,
        # -0.19319901])
        res = minimize(self.calc_dissimilarity, t, method="BFGS", options={'gtol': 1e-3, 'disp': True}, jac = grad, callback = Registration._track)
        print(res)
        return res.x

        print("Similarity", self.calc_dissimilarity(t))
        for _ in range(500):
            upd = grad(t)
            print(upd)
            print(self.num_grad(t))
            for i in range(6):
                t[i] += rate * -upd[i]
            print("Similarity", self.calc_dissimilarity(t))
            # print(t)
        return t

    def num_grad(self, params, e=0.1):
        perturb = npn.zeros(params.shape)
        numgrad = npn.zeros(params.shape)
        for p in range(len(params)):
            perturb[p] = e
            loss2 = self.calc_dissimilarity(params + perturb)
            loss1 = self.calc_dissimilarity(params - perturb)
            numgrad[p] = (loss2 - loss1) / (2.0 * e)

        return numgrad

    # track function for scipy minimize
    @staticmethod
    def _track(xk, state):
        print(xk)
        print(state.fun)