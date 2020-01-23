from drl.agents.q_learning_agent import QLearningAgent
from drl.environments.discretized_cartpole import DiscretizedCartPole

epochs = 100
environment = DiscretizedCartPole()
agent = QLearningAgent(environment, epochs=epochs)
current_state = environment.starting_state

for epoch in range(epochs):
    environment.env.render()
    action = agent.predict(current_state)
    observation_binned, reward, done, info = environment.step(action)
    info['epoch'] = epoch
    agent.update(observation_binned, reward, done, info)
    current_state = observation_binned
