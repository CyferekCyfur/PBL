import matplotlib.pyplot as plt
from nowyKalman import KF

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


plt.subplot(2, 1, 1)
plt.title("Pozycja")
plt.plot([el[0] for el in means], "b")  #
plt.plot(
    [el[0] - covs[0, 0] for el, covs in zip(means, covs)], "b--"
)  # covs[0,0] is the covariance of the position, similarlly, covs[1,1] corresponds to velocity

plt.show()
plt.title("Predkosc")
plt.plot([el[1] for el in means], "b")
plt.plot([el[1] - covs[1, 1] for el, covs in zip(means, covs)], "b--")

plt.show()
plt.ginput(1)
