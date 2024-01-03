import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import dot
from numpy.linalg import det


class KF:
    def __init__(
        self, initial_x: float, initial_v: float, accel_variance: float
    ) -> None:
        self._x = np.array(
            [initial_x, initial_v]
        )  # vector represening actual prediction of the velocity and position
        self._P = np.eye(2)  # uncertainty of the velocity and position of the X vector
        self._accel_variance = accel_variance  # noise of the acceleration

    @property
    def pos(self) -> float:
        return self._x[0]

    @property
    def vel(self) -> float:
        return self._x[1]

    @property
    def mean(self):
        return self._x

    @property
    def covariance(self):
        return self._P

    def predict(self, dt: float) -> None:
        F = np.array([[1, dt], [0, 1]])

        G = np.array([[0.5 * dt**2], [dt]])

        new_P = (
            F.dot(self._P).dot(np.transpose(F))
            + G.dot(np.transpose(G)) * self._accel_variance
        )

        new_x = F.dot(self._x)

        self._P = new_P
        self._x = new_x

    def update(self):
        pass


kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

plt.figure()

DT = 0.1
STEPS = 1000
# testing the filter with plots
means = []
covs = []
for i in range(STEPS):
    covs.append(kf.covariance)
    means.append(kf.mean)
    kf.predict(dt=DT)

plt.title("Pozycja")
plt.plot([el[0] for el in means], "b")
plt.show()
plt.title("Predkosc")
plt.plot([el[1] for el in means], "b")

plt.show()
plt.ginput(1)
