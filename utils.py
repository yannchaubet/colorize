import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


def transport_1d(x, y):
    i_x = np.argsort(x[:, 0])
    i_y = np.argsort(y[:, 0])
    return i_x, i_y


def project2d(X, theta):
    n = len(X)
    u_theta = np.array([[np.cos(theta), np.sin(theta)]]).transpose()
    return np.matmul(X, u_theta).reshape((n, 1))


def project3d(X, theta, phi):
    n = len(X)
    u_theta_phi = np.array([[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]]).transpose()
    return np.matmul(X, u_theta_phi).reshape((n, 1))


def evolve2d(X, Y, theta, epsilon):
    u = project2d(X, theta)
    v = project2d(Y, theta)
    i_u, i_v = transport_1d(u, v)
    X_c = X.copy()
    delta = np.matmul(epsilon * (v[i_v] - u[i_u]),
                      np.array([[np.cos(theta), np.sin(theta)]]))
    # print(X_c)
    X_c[i_u] = X_c[i_u] + delta
    # print(X_c)
    return X_c, i_u, i_v


def evolve3d(X, Y, theta, phi, epsilon):
    u = project3d(X, theta, phi)
    v = project3d(Y, theta, phi)
    i_u, i_v = transport_1d(u, v)
    X_c = X.copy()
    delta = np.matmul(epsilon * (v[i_v] - u[i_u]),
                                    np.array([[np.sin(theta) * np.cos(phi),
                                               np.sin(theta) * np.sin(phi),
                                               np.cos(theta)]]))
    # print(X_c)
    X_c[i_u] = X_c[i_u] + delta
    # print(X_c)
    return X_c, i_u, i_v

# i_v[i_u] est la permutation qui envoie k-ieme sur k-ieme


def project3dpermutations(X, Y, theta, phi):
    u = project3d(X, theta, phi)
    v = project3d(Y, theta, phi)
    i_u, i_v = transport_1d(u, v)
    return i_u, i_v


def plot2d(X, Y, N, epsilon):
    fig, ax = plt.subplots()
    n = len(X)
    X_c = X.copy()
    for k in range(0, N):
        theta = 2 * np.pi * np.random.random()
        X, i_u, i_v = evolve2d(X, Y, theta, epsilon)
        if k % 1 == 0:
            plt.plot(X[:, 0], X[:, 1], 'o', color=(k/(1.4 * N), k/(1.4 * N), k/(1.4 * N)))

        print(k)
        for s in range(n):
            print(str(i_u[s]) + ' -> ' + str(i_v[s]))

    ax.plot(Y[:, 0], Y[:, 1], 'o', color='blue')
    ax.plot(X_c[:, 0], X_c[:, 1], 'o', color='red')
    print()
    print("X : " + str(X_c))
    print("Y : " + str(Y))
    plt.show()


def plot3d(X, Y, N=100, epsilon=0.05):
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    n = len(X)
    X_c = X.copy()

    for k in range(0, N):
        theta = 2 * np.pi * np.random.random()
        phi = np.pi * np.random.random()
        X, i_u, i_v = evolve3d(X, Y, theta, phi, epsilon)
        X = relocate(X)

        if k % 10 == 0:
        # if 1:
            ax.scatter(*X.transpose(), marker='o', c=X)
            print(k)

        # for s in range(n):
            # print(str(i_u[s]) + ' -> ' + str(i_v[s]))

    X_c = relocate(X_c)
    Y = relocate(Y)
    ax.scatter(*Y.transpose(), marker='o', c=Y)
    ax.scatter(*X_c.transpose(), marker='o', c=X_c)
    plt.show()


def transport3d(X, Y, N=100, epsilon=0.05):
    i_u, i_v = None, None
    percent = 0

    for k in range(0, N):
        theta = 2 * np.pi * np.random.random()
        phi = np.pi * np.random.random()
        X, i_u, i_v = evolve3d(X, Y, theta, phi, epsilon)
        print(cost(X, Y, i_u, i_v))

        p = int(100 * k / N)
        if p > percent:
            print(str(p) + "%")
            percent += 1

    return i_u, i_v


def shift_(sigma):
    t = np.zeros(len(sigma))
    t[0] = sigma[-1]
    t[1:] = sigma[:len(sigma) - 1]
    return t


def unshift_(sigma):
    t = np.zeros(len(sigma))
    t[:len(t) - 1] = sigma[1:]
    t[len(t) - 1] = sigma[0]
    return t


def shift(sigma, k):
    t = np.ones(len(sigma))
    t[:-k] = sigma[k:]
    t[-k:] = sigma[:k]
    return t


def invert(sigma):
    return np.argsort(sigma)


def col_lig(sigma, w):
    sigma_c = sigma % w
    sigma_l = np.round(sigma / w, 0)
    return sigma_c, sigma_l


def dist(sigma, sigma_bis, w, l):
    sigma_c, sigma_l = col_lig(sigma, w)
    sigma_bis_c, sigma_bis_l = col_lig(sigma_bis, w)
    delta = np.sum(np.abs(sigma_c - sigma_bis_c) + np.abs(sigma_l - sigma_bis_l))
    return delta / ((w + l) * w * l)


def cost(X, Y, i_X, i_Y, alpha=0, w=0, l=0):
    c = 0

    if alpha > 0:
        i_X_inverse = invert(i_X)
        sigma = i_X_inverse[i_Y]

        for k in [-1, 1, -l, l, -l - 1, -l + 1, l - 1, l + 1]:
            sigma_bis = shift(sigma, k)
            delta = dist(sigma, sigma_bis, w, l)
            c += delta

        return np.sqrt(np.sum((X[i_X] - Y[i_Y]) ** 2)) + alpha * c

    return np.sqrt(np.sum((X[i_X] - Y[i_Y]) ** 2))


def transport3d_bis(X, Y, N=100, alpha=0, w=0, l=0):
    i_u_final, i_v_final = None, None
    percent = 0
    best_cost = np.inf

    for k in range(0, N):
        theta = 2 * np.pi * np.random.random()
        phi = np.pi * np.random.random()
        i_u, i_v = project3dpermutations(X, Y, theta, phi)
        c = cost(X, Y, i_u, i_v, alpha=alpha, w=w, l=l)

        if c < best_cost:
            best_cost = c
            i_u_final = i_u
            i_v_final = i_v

        p = int(100 * k / N)

        if p > percent + 1:
            print(str(p) + "%")
            percent = p

    print("Coût : " + str(best_cost))

    return i_u_final, i_v_final


def transport3d_max(X, Y, N=100, alpha=0, w=0, l=0):
    i_u_final, i_v_final = None, None
    percent = 0
    best_cost = 0

    for k in range(0, N):
        theta = 2 * np.pi * np.random.random()
        phi = np.pi * np.random.random()
        i_u, i_v = project3dpermutations(X, Y, theta, phi)
        c = cost(X, Y, i_u, i_v, alpha=alpha, w=w, l=l)

        if c > best_cost:
            best_cost = c
            i_u_final = i_u
            i_v_final = i_v

        p = int(100 * k / N)

        if p > percent + 1:
            print(str(p) + "%")
            percent = p

    print("Coût : " + str(best_cost))

    return i_u_final, i_v_final


def relocate(Z):
    Z[Z > 1] = 1
    Z[Z < 0] = 0
    return Z


def show3d(X, Y, i_X, i_Y, n):
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    U = np.arange(0, len(X))
    np.random.shuffle(U)
    U = U[:n]
    print(len(U))
    X_sorted = X[i_X]
    Y_sorted = Y[i_Y]
    ax.scatter(*X_sorted[U].transpose(), marker='o', c=X_sorted[U])
    ax.scatter(*Y_sorted[U].transpose(), marker='s', c=Y_sorted[U])

    for k in U:
        ax.plot([X_sorted[k, 0], Y_sorted[k, 0]],
                [X_sorted[k, 1], Y_sorted[k, 1]],
                [X_sorted[k, 2], Y_sorted[k, 2]],
                color='grey')


def compute_cost(X, Y, Z, t):
    return (1 - t) * np.sum((X - Z) ** 2) + t * np.sum((Y - Z) ** 2)


def compute_gradient(X, Y, Z, t):
    return 2 * (1 - t) * (X - Z) * Z + 2 * t * (Y - Z) * Z


def geodesic_interpolation(X, Y, t, N_iterations=100, eta=0.1, show_cost=False):
    Z = (1 - t) * X + t * Y

    for iter in range(N_iterations):
        delta = compute_gradient(X, Y, Z, t)
        Z = Z - eta * delta

        if show_cost:
            print(compute_cost(X, Y, Z, t))

    return Z

