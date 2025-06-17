# Hyperparamètres optimisés
self.batch_size = 256  # Augmenté de 128 à 256
self.gamma = 0.99
self.epsilon = 1.0
self.epsilon_min = 0.01
self.epsilon_decay = 0.995
self.learning_rate = 0.001
self.target_update_freq = 2000  # Augmenté de 1000 à 2000
self.memory_size = 50000  # Réduit de 100000 à 50000
self.warmup_steps = 1000  # Réduit de 5000 à 1000 