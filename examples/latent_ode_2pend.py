import os
import argparse
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint as sp_odeint
import pendulum_sim as pen_sim
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--viz', action='store_true')
# parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def generate_double_pend(batch=1000,
                         ntotal=500,
                         nsample=100,
                         start_theta_1=np.pi/4,
                         start_theta_2=np.pi/4,
                         stop_t=5,  # approximately equal to 6pi
                         noise_std=.1,
                         l1=1.,
                         l2=1.,
                         m1=1,
                         m2=1,
                         g=9.8,
                         savefig=True):
    """
    Args:
      batch: batch dimension
      ntotal: total number of datapoints per set
      nsample: number of sampled datapoints for model fitting
      start_theta_1: first arm starting theta value
      start_theta_2: second arm starting theta value
      stop: ending t value
      noise_std: observation noise standard deviation
      l1, l2, m1, m2, g: parameters of the double pendulum
      savefig: plot the ground truth for sanity check

    Returns:
      Tuple where first element is true trajectory of size (ntotal, 2),
      second element is noisy observations of size (batch, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """
    y0 = [start_theta_1, start_theta_2, 0.0, 0.0]

    orig_ts = np.linspace(0, stop_t, num=ntotal)
    samp_ts = orig_ts[:nsample]
    sample_plot_range = np.zeros((batch, nsample))

    original = sp_odeint(pen_sim.double_pendulum, y0, orig_ts,
                         args=(g, l1, l2, m1, m2))

    # remove omega values
    original = original[:, 0:2]
    samp_trajs = []

    for i in range(batch):

        # don't sample t0 very near the start or the end
        t0_idx = npr.randint(0, ntotal - nsample)

        sample_plot_range[i, :] = orig_ts[t0_idx:t0_idx + nsample]

        samp_traj = original[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    samp_trajs = np.stack(samp_trajs, axis=0)

    return original, samp_trajs, orig_ts, samp_ts, sample_plot_range


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=5, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=5, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=5, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


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


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


if __name__ == '__main__':
    latent_dim = 5
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    batch = 1000
    noise_std = .3
    a = 0.
    b = .3
    ntotal = 1000
    nsample = 100
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate toy double pendulum data
    orig_trajs, samp_trajs, orig_ts, samp_ts, sample_plot_range = generate_double_pend(
        batch=batch,
        noise_std=noise_std
    )

    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    orig_ts_tensor = torch.from_numpy(orig_ts).float()

    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, batch).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    if args.viz:
        if not os.path.exists('double_png'):
            os.makedirs('double_png')
        from sys import platform as sys_pf
        if sys_pf == 'darwin':
            import matplotlib
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(5, 5), facecolor='white')
        ax_traj = fig.add_axes((0.15, 0.15, .85, .85), frameon=False)

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            samp_trajs = checkpoint['samp_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']

    try:
        loss_list = []
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

            loss_list.append(-loss_meter.avg)

            if args.viz:

                # print('test5')

                ax_traj.cla()

                full_pred_z = odeint(func, z0, orig_ts_tensor).permute(1, 0, 2)
                full_pred_x = dec(full_pred_z)
                full_traj = full_pred_x.detach().numpy()

                orig_traj = orig_trajs.cpu().numpy()

                ax_traj.set_title('Trajectories')
                ax_traj.set_xlabel('t')
                ax_traj.set_ylabel('theta 1 (blue), theta 2 (green)')
                ax_traj.plot(orig_ts, orig_traj[:, 0], 'b-', orig_ts,
                             orig_traj[:, 1], 'g-')
                ax_traj.plot(orig_ts, full_traj[0, :, 0], 'b--', orig_ts,
                             full_traj[0, :, 1], 'g--')

                plt.draw()
                plt.savefig('double_png/{:03d}'.format(itr))
                plt.pause(0.001)

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'orig_trajs': orig_trajs,
                'samp_trajs': samp_trajs,
                'orig_ts': orig_ts,
                'samp_ts': samp_ts,
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.viz:
        plt.clf()
        plt.plot(loss_list)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig('double_train_loss.png')

    with open("double_loss.txt", "wb") as fp:   # Pickling
        pickle.dump(loss_list, fp)
