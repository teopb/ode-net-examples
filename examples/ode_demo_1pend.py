import os
import argparse
import time
import numpy as np
import pickle
from sys import platform as sys_pf

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]])
t = torch.linspace(0., 25., args.data_size)
# constants
b = 0.0
c = 1.0
true_A = torch.tensor([[0, 1], [-c, -b]])


class Lambda(nn.Module):

    def forward(self, t, y):
        # y = [theta, omega]
        # y_p = y
        # y_p[:, 0] = torch.sin(y[:, 0])
        y_p = torch.cat([torch.sin(torch.narrow(y, 1, 0, 1)), torch.narrow(y, 1, 1, 1)], dim=1)
        return torch.mm(y_p, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    # print(f"s: {s}")
    batch_y0 = true_y[s]  # (M, D)
    # print(f"batch_y0 = {batch_y0.shape}")
    batch_t = t[:args.batch_time]  # (T)
    # print(batch_t)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    # print(f"batch_y = {batch_y.shape}")
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


if args.viz:
    makedirs('single_png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4), facecolor='white')
    ax_traj = fig.add_subplot(111, frameon=False)
    # ax_phase = fig.add_subplot(132, frameon=False)
    # ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title(f'Trajectories, iteration = {itr}')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('theta (blue), omega (green)')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], 'b-', t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], 'b--', t.numpy(), pred_y.numpy()[:, 0, 1], 'g--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('single_png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # print(y.shape)
        y_p = torch.cat([torch.sin(torch.narrow(y, -1, 0, 1)), torch.narrow(y, -1, 1, 1)], dim=1)
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    loss_list = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                loss_list.append(loss.item())
                ii += 1

        end = time.time()

    with open("single_loss.txt", "wb") as fp:   # Pickling
        pickle.dump(loss_list, fp)

    if args.viz:
        plt.clf()
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig('single_train_loss.png')
