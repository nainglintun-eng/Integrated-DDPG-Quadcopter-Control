"""
Quadcopter Dynamics - Matched to MATLAB Full_f / Full_g

State vector (12x1):
    X = [x, y, z, u, v, w, phi, theta, psi, p, q, r]
         [0  1  2  3  4  5   6     7    8   9  10  11]

    x,y,z    - position (world frame)
    u,v,w    - linear velocity (body frame)
    phi,theta,psi - Euler angles (roll, pitch, yaw)
    p,q,r    - angular rates (body frame)

Control input (4x1):
    U = [Ft, tau_x, tau_y, tau_z]
         [0     1      2      3  ]

    Ft        - total thrust (N)
    tau_x/y/z - body-frame torques (Nm)

Dynamics (matches MATLAB):
    X_dot = Full_f(X, g, Ix, Iy, Iz) + Full_g(X, m, Ix, Iy, Iz) * U
"""

import numpy as np


class QuadcopterDynamics:
    """
    6-DOF quadcopter dynamics that exactly mirrors the MATLAB
    Full_f / Full_g formulation.

    State  : [x, y, z, u, v, w, phi, theta, psi, p, q, r]
    Control: [Ft, tau_x, tau_y, tau_z]
    """

    def __init__(self, mass=1.0, gravity=9.81, dt=0.01):
        self.m  = mass
        self.g  = gravity
        self.dt = dt

        # Inertia values  (Ix=0.3, Iy=0.4, Iz=0.5 as requested)
        self.Ixx = 0.3   # kg·m²
        self.Iyy = 0.4   # kg·m²
        self.Izz = 0.5   # kg·m²

        # Wind disturbances (kept for compatibility with existing trainers)
        self.wind_enabled        = False
        self.wind_mean           = np.zeros(3)
        self.wind_std            = np.array([0.005, 0.005, 0.003])
        self.wind_gust_prob      = 0.01
        self.wind_gust_magnitude = 0.001

    def set_wind(self, enabled=True, std=None, mean=None,
                 gust_prob=0.01, gust_mag=0.001):
        self.wind_enabled = enabled
        if std  is not None: self.wind_std  = np.array(std)
        if mean is not None: self.wind_mean = np.array(mean)
        self.wind_gust_prob      = gust_prob
        self.wind_gust_magnitude = gust_mag

    # ------------------------------------------------------------------
    #  MATLAB Full_f  (drift / unforced dynamics)
    # ------------------------------------------------------------------
    def Full_f(self, X):
        """
        Exact Python translation of MATLAB Full_f(X, grav, Ix, Iy, Iz).
        Returns 12-element numpy array.
        """
        u  = X[3];  v  = X[4];  w  = X[5]
        ph = X[6];  th = X[7];  ps = X[8]
        p  = X[9];  q  = X[10]; r  = X[11]

        g  = self.g
        Ix = self.Ixx;  Iy = self.Iyy;  Iz = self.Izz

        f = np.array([
            u,                                                           # x_dot
            v,                                                           # y_dot
            w,                                                           # z_dot
            0.0,                                                         # u_dot
            0.0,                                                         # v_dot
            g,                                                           # w_dot (gravity +z down)
            p + q*np.sin(ph)*np.tan(th) + r*np.cos(ph)*np.tan(th),     # phi_dot
            q*np.cos(ph)               - r*np.sin(ph),                  # theta_dot
            q*np.sin(ph)/np.cos(th)    + r*np.cos(ph)/np.cos(th),      # psi_dot
            (Iy - Iz) / Ix * q * r,                                     # p_dot
            (Iz - Ix) / Iy * p * r,                                     # q_dot
            (Ix - Iy) / Iz * p * q,                                     # r_dot
        ])
        return f

    # ------------------------------------------------------------------
    #  MATLAB Full_g  (input matrix)
    # ------------------------------------------------------------------
    def Full_g(self, X):
        """
        Exact Python translation of MATLAB Full_g(X, m, Ix, Iy, Iz).
        Returns 12×4 numpy array.
        """
        ph = X[6]; th = X[7]; ps = X[8]

        m  = self.m
        Ix = self.Ixx;  Iy = self.Iyy;  Iz = self.Izz

        # Thrust-to-velocity-acceleration coupling (rows 4,5,6)
        g17 = -1.0/m * (np.sin(ph)*np.sin(ps) + np.cos(ph)*np.cos(ps)*np.sin(th))
        g18 = -1.0/m * (-np.cos(ps)*np.sin(ph) + np.cos(ph)*np.sin(ps)*np.sin(th))
        g19 = -1.0/m * (np.cos(ph)*np.cos(th))

        #        Ft      tau_x    tau_y    tau_z
        g = np.array([
            [0,      0,       0,       0     ],   # x
            [0,      0,       0,       0     ],   # y
            [0,      0,       0,       0     ],   # z
            [g17,    0,       0,       0     ],   # u
            [g18,    0,       0,       0     ],   # v
            [g19,    0,       0,       0     ],   # w
            [0,      0,       0,       0     ],   # phi
            [0,      0,       0,       0     ],   # theta
            [0,      0,       0,       0     ],   # psi
            [0,      1.0/Ix,  0,       0     ],   # p
            [0,      0,       1.0/Iy,  0     ],   # q
            [0,      0,       0,       1.0/Iz],   # r
        ])
        return g

    # ------------------------------------------------------------------
    #  Euler integration  (mirrors MATLAB: X(:,i+1) = X(:,i) + dt*(f+g*U))
    # ------------------------------------------------------------------
    def euler_step(self, X, U):
        """
        One Euler step:  X_new = X + dt * (f(X) + g(X) @ U)

        X : (12,)  state
        U : (4,)   control  [Ft, tau_x, tau_y, tau_z]
        """
        wind_acc = np.zeros(3)
        if self.wind_enabled:
            wind_vel = np.random.normal(self.wind_mean, self.wind_std)
            wind_acc = 0.01 * wind_vel
            if np.random.random() < self.wind_gust_prob:
                gust = np.random.randn(3)
                gust = gust / (np.linalg.norm(gust) + 1e-9)
                wind_acc += gust * self.wind_gust_magnitude

        f    = self.Full_f(X)
        g    = self.Full_g(X)
        Xdot = f + g @ U
        Xdot[3:6] += wind_acc

        X_new        = X + self.dt * Xdot
        X_new[6:9]   = self.wrap_angles(X_new[6:9])
        return X_new

    # ------------------------------------------------------------------
    #  RK4 integration  (used by attitude_env and position_env)
    # ------------------------------------------------------------------
    def rk4_step(self, X, thrust, torques):
        """
        Compatibility shim so existing env code keeps working unchanged.

        thrust  : scalar Ft  (N)
        torques : (3,)  [tau_x, tau_y, tau_z]
        """
        U = np.array([thrust, torques[0], torques[1], torques[2]])

        def deriv(x):
            xdot = self.Full_f(x) + self.Full_g(x) @ U
            if self.wind_enabled:
                wind_vel   = np.random.normal(self.wind_mean, self.wind_std)
                xdot[3:6] += 0.01 * wind_vel
            return xdot

        k1 = deriv(X)
        k2 = deriv(X + 0.5*self.dt*k1)
        k3 = deriv(X + 0.5*self.dt*k2)
        k4 = deriv(X + self.dt*k3)

        X_new      = X + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        X_new[6:9] = self.wrap_angles(X_new[6:9])
        return X_new

    @staticmethod
    def wrap_angles(angles):
        """Wrap angles to [-pi, pi]"""
        return (angles + np.pi) % (2 * np.pi) - np.pi
