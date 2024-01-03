import matplotlib.pyplot as plt
from nowyKalman import KF

kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1)

plt.figure()


DT = 0.1
STEPS = 1000
MEASURE_EVERY_STEPS = 20

test_velocity = 2
test_x = 0.0
measurement_noise = 0.0001
random_interference = 3 * np.random.rand()


# testing the filter with plots
means = []
covs = []
for i in range(STEPS):
    covs.append(kf.covariance)
    means.append(kf.mean)
    kf.predict(dt=DT)
    if step != 1 and step % MEASURE_EVERY_STEPS == 0:
      kf.update(test_x + test_velocity*DT*step + random_interference, measurement_noise) 


plt.subplot(2, 1, 1)
plt.title("Pozycja")
plt.plot([el[0] for el in means], "b")  #
plt.plot(
    [el[0] - covs[0, 0] for el, covs in zip(means, covs)], "b--"
)  # covs[0,0] is the covariance of the position, similarlly, covs[1,1] corresponds to velocity
plt.grid()
plt.show()


plt.title("Predkosc")
plt.plot([el[1] for el in means], "b")
plt.plot([el[1] - covs[1, 1] for el, covs in zip(means, covs)], "b--")
plt.grid()
plt.show()
plt.ginput(1)
