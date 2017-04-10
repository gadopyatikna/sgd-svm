import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_error_history(errors, i):
    n = len(errors)
    plt.plot(xrange(n), errors)
    plt.savefig('plots/{}-error-history.png'.format(i))
    plt.clf()
    plt.close()

def plot_classifier(W, data, Y):
    fig = plt.figure()

    if W.shape[0] - 1 == 2:
        # dim(w) - dim(bias) == 2 then it is 2d data
        ax = fig.add_subplot(111)
        for x, y in zip(data, Y):
            ax.scatter(x[0], x[1], color = ( 'red' if y == 1 else 'black' ))

        # plot decision boundary
        xax = [np.min(data[:,0]) , np.max(data[:,0])]
        yax = [-(W[0]*x0)/W[1] for x0 in xax]
        ax.plot(xax, yax)
        plt.savefig('plots/2d-linsep-data.png')

    else:
        ax = fig.add_subplot(111, projection='3d')
        # plot decision plane
        plane_x, plane_y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
        plane_z = -(W[1]*plane_x+W[2]*plane_y)/W[0]
        ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.2)
        # plot data points
        for x, y in zip(data, Y):
            ax.scatter(x[0], x[1], x[2], color = ( 'red' if y == 1 else 'black' ))
        plt.savefig('plots/3d-nonlinsep-data.png')
    plt.clf()
    plt.close()

def transform_X(data, space=None):
    N = data.shape[0]
    # include bias
    b = np.ones((N, 1))
    if not space:
        return np.concatenate((data, b), axis=1)

    # add 3rd dimension where data will be separable
    elif space == '3d':
        return np.array(( data[:,0], data[:,1], data[:,0]**2+data[:,1]**2, np.ones(N) )).T

def compute_error(data, Y, W, lamb):
    N = data.shape[0]
    errors = [max(0, 1 - y * np.dot(x, W)) + lamb / 2. * np.dot(W, W.T) for x, y in zip(data, Y)]
    return np.sum(errors)/(N+.0), np.sum([1 for x, y in zip(data, Y) if np.dot(W, x)*y < 1])

def sgd(data, Y, error_threshold, C=1000, l_rate=1.):
    N = data.shape[0]
    # set lambda
    lamb = 1./(C*N)
    dim = data.shape[1]

    error_history = []

    # init weights randomly for training
    W = np.random.rand(dim)

    # stopping criterion = classifier 2fold CV error
    avg_val_error = np.inf

    # epoch counter
    epoch = 0
    while avg_val_error > error_threshold:

        # divide dataset into training and validating subsets
        mask = np.random.rand(N) > 0.5
        avg_val_error = 0

        # perform k fold
        for fold in [mask, ~mask]:
            trn_data, trn_Y = data[fold], Y[fold]
            val_data, val_Y = data[~fold], Y[~fold]

            for x, y in zip(trn_data, trn_Y):
                if y * (np.dot(x, W)) < 1:
                    W -= l_rate * ( lamb * W - y * x )
                else:
                    W -= l_rate * lamb * W

            val_error, n_val_errors = compute_error(val_data, val_Y, W, lamb)
            print 'n_val_errs = ' + str(n_val_errors)
            avg_val_error += val_error

        # averaging validation error
        avg_val_error = avg_val_error/2
        error_history.append(avg_val_error)

        print 'epoch {}\nerror value = {}'.format(epoch, avg_val_error)

        epoch += 1

    return W, error_history

if __name__ == "__main__":
    for i in ['linsep', 'nonlinsep']:
        path = 'data/{}-traindata.csv'.format(i)
        data_ = np.genfromtxt(path, dtype='float', delimiter=',')

        path = 'data/{}-trainclass.csv'.format(i)
        Y = np.genfromtxt(path, delimiter=',').reshape(-1, 1)

        data = transform_X(data_, space= '3d' if i == 'nonlinsep' else None)
        W, errors = sgd(data, Y, error_threshold=0.006 if i == 'nonlinsep' else 0.001, l_rate=1)

        plot_classifier(W, data, Y)
        plot_error_history(errors, i)