import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import dot
from numpy.linalg import det


class KF:
    def __init__(
        self, initial_x: float, initial_v: float, accel_variance: float
    ) -> None:
        self.x = np.array(
            [initial_x, initial_v]
        )  # vector represening actual prediction of the velocity and position
        self.P = np.eye(
            2
        )  # uncertainty of the velocity and position of the X vector, the covariance matrix
        self._accel_variance = accel_variance  # noise of the acceleration

    @property
    def pos(self) -> float:
        return self.x[0]

    @property
    def vel(self) -> float:
        return self.x[1]

    @property
    def mean(self):
        return self.x

    @property
    def covariance(self):
        return self.P

    def predict(self, dt: float) -> None:
        F = np.array([[1, dt], [0, 1]])

        G = np.array([[0.5 * dt**2], [dt]])

        new_P = (
            F.dot(self.P).dot(np.transpose(F))
            + G.dot(np.transpose(G)) * self._accel_variance
        )

        new_x = F.dot(self.x)

        self.P = new_P
        self.x = new_x

    def update(
        self, measurement: float, measurement_variance: float
    ):  # update updates the prediction with estimated state
        z = np.array([measurement])
        R = np.array([measurement_variance])

        H = np.array([1, 0]).reshape((1, 2))
        y = z - H.dot(self.x)  # innovation of the state
        S = H.dot(self.P).dot(np.transpose(H)) + R  # innovation of the covariance
        K = self.P.dot(np.transpose(H)).dot(
            np.linalg.inv(S)
        )  # basically the gain of the filter
        new_x = self.x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self.P)

        self.P = new_P
        self.x = new_x
