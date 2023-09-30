import numpy as np

from kitti_frame_importer import KittiFrameImporter
from loss_calculator import LossCalculator
from optimization.particle_swarm.particle import Particle
from optimization_parameters import OptimizationParameters

from tqdm import tqdm

number_of_particles = 30
minibatch_size = 5
number_of_frames = 100


class ParticleSwarm:
    def __init__(self):
        self.particles: list[Particle] = []

        for i in range(number_of_particles):
            self.particles.append(
                Particle(i)
            )

        self.loss_calculator = LossCalculator(
            KittiFrameImporter('/home/luka/Desktop/Customer/kira/master')
        )

        self.optimization_parameters = OptimizationParameters()

    def optimize(self):
        frames_18 = 18e8 + np.arange(250)

        frames = frames_18
        np.random.shuffle(frames)
        for num, i in tqdm(enumerate(frames)):

            if int(i) in (177, 178, 179, 180, 1400000005):
                continue

            self.loss_calculator.load_frame(frame_id=i)

            for p in self.particles:
                p.loss(loss_function=self.loss_calculator.compute_loss_function(self.optimization_parameters))

            if num % minibatch_size + 1 == minibatch_size:
                for p in self.particles:
                    p.update_at_end_of_minibatch()

                for p in self.particles:
                    p.sync(self.connection_strategy())

                for p in self.particles:
                    p.update()

                X = np.mean([i.l for i in self.particles], axis=0)
                # X = [i.X for i in self.particles]
                print(X)

        #         L = [p.l for p in self.particles]
        #         print("---------------------------")
        #         print(L)
        #
        # L = [p.l for p in self.particles]
        # print("---------------------------")
        # print(L)

    def connection_strategy(self):

        def get_connections(p: Particle):
            identifier = p.identifier

            return p, self.particles[identifier - 2], self.particles[identifier - 1], self.particles[
                (identifier + 1) % number_of_particles], self.particles[(identifier + 2) % number_of_particles]

        return get_connections


