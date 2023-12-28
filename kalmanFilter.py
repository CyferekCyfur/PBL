import numpy as np
from matplotlib import pyplot as plt


class KF:
    def __init__(
        self, initial_x: float, initial_v: float, accel_variance: float
    ) -> None:
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance
        self._P = np.eye(2)

        def predict(self, dt: float) -> None:
            F = np.array([[1, dt], [0, 1]])
            new_x = F.dot(self._x)

            G = np.array([0.5 * dt**2, dt]).reshape((2, 1))
            new_P = F.dot(self.P).dot(F.T) + G.dot(G.T) * self._accel_variance

            self._P = new_P
            self._x = new_x

        @property
        def cov(self) -> np.array:
            return self._P

        @property
        def mean(self) -> np.array:
            return self._x

        @property
        def pos(self) -> float:
            return self._x[0]

        @property
        def vel(self) -> float:
            return self._x[1]


plt.ion()
plt.figure()

kalman = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)
DT = 0.1
NUM_STEPS = 1000
mus = []
covs = []
for i in range(NUM_STEPS):
    covs.append(kalman.cov)
    mus.append(kalman.mean)
    kalman.predict(dt=DT)
plt.subplot(2, 1, 1)
plt.title("Position")
plt.plot([mu[0] for mu in mus], "r")

plt.subplot(2, 1, 2)
plt.title("Velocity")
plt.show()
plt.ginput(1)
