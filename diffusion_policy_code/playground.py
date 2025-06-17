import numpy as np
import matplotlib.pyplot as plt


class CubicInterpolation:
    def __init__(self, q_via=None, t_via=None):
        """
        :param: name: string
            name of objective
        :param: q_via: N array
            given q array
        :param: t_via: N array
            given t array
        """
        super(self.__class__, self).__init__()
        self.q_via = q_via
        self.t_via = t_via

        assert (
            q_via.shape[0] == t_via.shape[0]
        ), "q_via and t_via must have the same length"

        self.velocity = np.zeros(self.q_via.shape[0])
        self._inferVelocity()

    def cubic(self, q0, q1, v0, v1, t0, t1):
        """
        :param: q0: float
            the first data point
        :param: q1: float
            the second data point
        :param: v0: float
            the velocity of the first data point
        :param: v1: float
            the velocity of the second data point
        :param: t0: float
            the time of the first data point
        :param: t1: float
            the time of the second data point
        """
        try:
            abs(t0 - t1) < 1e-6
        except ValueError:
            print("t0 and t1 must be different")

        T = t1 - t0
        h = q1 - q0

        a0 = q0
        a1 = v0
        a2 = (3 * h - (2 * v0 + v1) * T) / (T**2)
        a3 = (-2 * h + (v0 + v1) * T) / (T**3)
        return a0, a1, a2, a3

    def getPosition(self, t):
        """
        :param: t: float
            specified time
        :return: q: float
            output of the interpolation at time t
        """
        try:
            (t < self.t_via[0]) or (t > self.t_via[-1])
        except ValueError:
            print("The specific time error, time ranges error")

        j_array = np.where(self.t_via >= t)  # find the index of t1
        j = j_array[0][0]
        if j == 0:
            i = 0
            j = 1
        else:
            i = j - 1

        q = np.zeros((1, 3))

        # get given position
        q0 = self.q_via[i]
        v0 = self.velocity[i]
        t0 = self.t_via[i]

        q1 = self.q_via[j]
        v1 = self.velocity[j]
        t1 = self.t_via[j]

        a0, a1, a2, a3 = self.cubic(q0, q1, v0, v1, t0, t1)

        q[0, 0] = (
            a0 + a1 * (t - t0) + a2 * (t - t0) ** 2 + a3 * (t - t0) ** 3
        )  # position
        q[0, 1] = a1 + 2 * a2 * (t - t0) + 3 * a3 * (t - t0) ** 2  # velocity
        q[0, 2] = 2 * a2 + 6 * a3 * (t - t0)  # acceleration

        return q

    def _inferVelocity(self):
        mid_velocity = np.zeros(self.q_via.shape[0])
        for i in range(1, self.q_via.shape[0] - 1):
            mid_velocity[i] = (self.q_via[i] - self.q_via[i - 1]) / (
                self.t_via[i] - self.t_via[i - 1]
            )
        sign = np.sign(mid_velocity)

        for i in range(1, self.q_via.shape[0] - 2):
            if sign[i] != sign[i + 1]:
                self.velocity[i] = 0.0
            else:
                self.velocity[i] = (mid_velocity[i] + mid_velocity[i + 1]) / 2


if __name__ == "__main__":
    q_given = np.array([0, 1.6, 3.2, 2, 4, 0.2, 1.2])
    t_given = np.array([0, 1, 3, 4.5, 6, 8, 10])

    # time for interpolation
    t = np.linspace(t_given[0], t_given[-1], 1000)

    cubic_interpolation = CubicInterpolation("Cubic", q_given, t_given)
    cubic_trajectory = np.zeros(
        (t.shape[0], 3)
    )  # N x 3 array: position, velocity, acceleration

    for i in range(t.shape[0]):
        cubic_trajectory[i, :] = cubic_interpolation.getPosition(t[i])

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t_given, q_given[:], "ro")
    plt.plot(t, cubic_trajectory[:, 0], "k")
    plt.grid("on")
    plt.title("Cubic interpolation")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.xlim(t_given[0] - 1, t_given[-1] + 1)
    plt.ylim(min(q_given[:]) - 1, max(q_given[:]) + 1)

    plt.subplot(3, 1, 2)
    # plt.plot(t_given, q_given[:, 1], "ro")
    plt.plot(t, cubic_trajectory[:, 1], "k")
    plt.grid("on")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.xlim(t_given[0] - 1, t_given[-1] + 1)

    plt.subplot(3, 1, 3)
    # plt.plot(t_given, q_given[:, 2], "ro")
    plt.plot(t, cubic_trajectory[:, 2], "k")
    plt.grid("on")
    plt.xlabel("time")
    plt.ylabel("acceleration")
    plt.xlim(t_given[0] - 1, t_given[-1] + 1)
