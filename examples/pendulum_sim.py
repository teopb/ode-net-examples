from scipy.integrate import odeint
import numpy as np


def single_pendulum(y, t, b, c):
    # for passing to scipy.integrate.odeint

    # b represents friction
    # c represents initial conditions

    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]

    return dydt


def double_pendulum(y, t, g, l1, l2, m1, m2):
    # https://www.myphysicslab.com/pendulum/double-pendulum-en.html
    # b represents friction
    # g represents gravity
    # l1, l2 represent length to first mass, and from first to second mass
    # m1, m2 represents masses of first and second mass

    theta1, theta2, omega1, omega2 = y

    d_omega1_dt = (-g * (2*m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2*theta2) - 2*np.sin(theta1 - theta2) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(theta1 - theta2))) / \
        (l1 * (2*m1 + m2 - m2 * np.cos(2*theta1 - 2*theta2)))

    d_omega2_dt = (2 * np.sin(theta1 - theta2) * (omega1**2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + omega2**2 * l2 * m2 * np.cos(theta1 - theta2))) / \
        (l2 * (2*m1 + m2 - m2 * np.cos(2*theta1 - 2*theta2)))

    dydt = [omega1, omega2, d_omega1_dt, d_omega2_dt]

    return dydt


def test_single_pend():
    # Initial conditions and constants
    y0 = [np.pi/4.0, 0.0]
    b = 0
    c = 1

    steps = 100

    t = np.linspace(0, 10, steps)

    solution = odeint(single_pendulum, y0, t, args=(b, c))

    import matplotlib.pyplot as plt
    plt.plot(t, solution[:, 0], 'b', label='theta(t)')
    plt.plot(t, solution[:, 1], 'g', label='omega(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def test_double_pend():
    # Initial conditions and constants
    y0 = [np.pi/4.0, np.pi/4.0, 0.0, 0.0]
    g = 9.8
    l1 = 1
    l2 = 1
    m1 = 1
    m2 = 1

    steps = 100

    t = np.linspace(0, 10, steps)

    solution = odeint(double_pendulum, y0, t, args=(g, l1, l2, m1, m2))

    print(solution.shape)

    import matplotlib.pyplot as plt
    plt.plot(t, solution[:, 0], 'b', label='theta1(t)')
    plt.plot(t, solution[:, 1], 'g', label='theta2(t)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()
