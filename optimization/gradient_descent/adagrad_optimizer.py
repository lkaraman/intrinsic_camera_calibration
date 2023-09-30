import numpy as np
from tqdm import tqdm

from kitti_frame_importer import KittiFrameImporter
from loss_calculator import LossCalculator
from optimization_parameters import OptimizationParameters


class AdagradOptimizer:
    def __init__(self, batch_size: int, step: float, learning_rate: float) -> None:
        self.batch_size = batch_size
        self.step = step
        self.learning_rate = learning_rate

        importer = KittiFrameImporter('/home/kira/Desktop/Customer/kitti/master')

        self.loss_calculator = LossCalculator(importer=importer)
        self.opt_params = OptimizationParameters()

        self.gradient_step = np.eye(self.opt_params.number_of_parameters) * 50

        # self.loss_calculator.load_frame(frame_id=1800000069)

    def optimize(self):
        # Originally I named the image/lidar files with the prefix of the batch name ('18')
        frames_18 = 18e8 + np.arange(250)
        frames = np.hstack((frames_18, frames_18, frames_18))

        np.random.shuffle(frames)
        num_of_parameters = self.opt_params.number_of_parameters
        Ie = np.eye(num_of_parameters) * 1e-3

        G = np.zeros((num_of_parameters, num_of_parameters))

        for kk, i in tqdm(enumerate(frames)):
            # Skip the problematic frames
            if int(i) in (177, 178, 179, 180, 1400000005):
                continue

            k = kk % self.batch_size + 1
            self.loss_calculator.load_frame(i)
            loss_function = self.loss_calculator.compute_loss_function(self.opt_params)

            if k == 1:
                g = np.zeros((1, num_of_parameters))
                avg_losses = []

            for j in range(num_of_parameters):
                theta_plus = self.opt_params.X + self.gradient_step[:, [j]]
                theta_minus = self.opt_params.X - self.gradient_step[:, [j]]

                g[0, j] = (1 - 1 / k) * g[0, j] + 1 / k * 1 / (2 * self.step) * (
                        loss_function(theta_plus) - loss_function(theta_minus))

            lossme = loss_function(self.opt_params.X)
            avg_losses.append(lossme)
            # print(f'Loss: {lossme}')

            if k == self.batch_size:
                G += g.T @ g

                self.opt_params.X -= (
                        np.diag((self.learning_rate * np.linalg.inv(np.sqrt(np.diag(np.diag(G + Ie)))))) * g).T

                print(np.average(avg_losses))

                print(self.opt_params.X)

                print(f'Frame {int(i)}')
                # self.loss_calculator.plot_current(opt_params=self.opt_params)



