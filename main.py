from optimization.gradient_descent.adagrad_optimizer import AdagradOptimizer
from optimization.particle_swarm.particle_swarm import ParticleSwarm

if __name__ == '__main__':
    adagrad_opt = AdagradOptimizer(
        batch_size=20,
        step=0.1,
        learning_rate=30
    )

    adagrad_opt.optimize()

    # particle_swarm = ParticleSwarm()
    # particle_swarm.optimize()