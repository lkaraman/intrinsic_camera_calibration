import numpy as np

# Intrinsic boundaries
fx_bound = (600, 800)
fy_bound = (600, 800)
cx_bound = (500, 700)
cy_bound = (100, 300)

w = 0.7215
# c = 1.19
c = 0.5

class Particle:
    def __init__(self, identifier: int) -> None:
        self.identifier = identifier

        fx = np.random.uniform(low=fx_bound[0], high=fx_bound[1])
        fy = np.random.uniform(low=fy_bound[0], high=fy_bound[1])
        cx = np.random.uniform(low=cx_bound[0], high=cx_bound[1])
        cy = np.random.uniform(low=cy_bound[0], high=cy_bound[1])

        vfx = np.random.uniform(low=-fx_bound[1] + fx_bound[0],
                                high=fx_bound[1] - fx_bound[0])

        vfy = np.random.uniform(low=-fy_bound[1] + fy_bound[0],
                                high=fy_bound[1] - fy_bound[0])

        vcx = np.random.uniform(low=-cx_bound[1] + cx_bound[0],
                                high=cx_bound[1] - cx_bound[0])

        vcy = np.random.uniform(low=-cy_bound[1] + cy_bound[0],
                                high=cy_bound[1] - cy_bound[0])

        self.X = np.vstack((fx, fy, cx, cy))
        self.P = np.copy(self.X)

        self.V = np.vstack((vfx, vfy, vcx, vcy))

        self.ex = 0
        self.ep = 0

        self.errors = []
        self.cys = []

    def loss(self, loss_function):
        self.ex += loss_function(self.X)
        self.ep += loss_function(self.P)

        self.errors.append(self.ex)
        self.cys.append(self.X[3][0])

    def update_at_end_of_minibatch(self):
        if self.ex < self.ep:
            self.P = self.X

    def sync(self, connections):
        neighboor_particles = connections(self)
        index_of_min_error = np.argmin([i.ep for i in neighboor_particles])
        self.l = np.copy(neighboor_particles[index_of_min_error].P)

    def update(self):
        r1 = np.random.rand()
        r2 = np.random.rand()

        self.V = w * self.V + c * r1 * (self.X - self.P) + c * r2 * (self.X - self.l)
        # self.V = w * self.V + c * r1 * (self.X - self.P)
        # print(self.V)
        self.X += self.V

        self.ex = 0
        self.ep = 0

        if self.X[0] < fx_bound[0]:
            self.X[0] = fx_bound[0]
            self.V[0] = -self.V[0]/2
        if self.X[0] > fx_bound[1]:
            self.X[0] = fx_bound[1]
            self.V[0] = -self.V[0] / 2

        if self.X[1] < fy_bound[0]:
            self.X[1] = fy_bound[0]
            self.V[1] = -self.V[1] / 2
        if self.X[1] > fy_bound[1]:
            self.X[1] = fy_bound[1]
            self.V[1] = -self.V[1] / 2


        if self.X[2] < cx_bound[0]:
            self.X[2] = cx_bound[0]
            self.V[2] = -self.V[2] / 2
        if self.X[2] > cx_bound[1]:
            self.X[2] = cx_bound[1]
            self.V[2] = -self.V[2] / 2

        if self.X[3] < cy_bound[0]:
            self.X[3] = cy_bound[0]
            self.V[3] = -self.V[3] / 2
        if self.X[3] > cy_bound[1]:
            self.X[3] = cy_bound[1]
            self.V[3] = -self.V[3] / 2