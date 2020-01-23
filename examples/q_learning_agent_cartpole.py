from drl.agents.q_learning_agent import QLearningAgent
from drl.environments.discretized_cartpole import DiscretizedCartPole

epochs = 300
environment = DiscretizedCartPole(bins=10)
agent = QLearningAgent(environment, epochs=epochs, gamma=0.8)

for epoch in range(epochs):
    current_state = environment.reset()
    total_reward = 0.0
    while not environment.is_terminal():
        environment.env.render()
        action = agent.predict(current_state)
        observation_binned, reward, done, info = environment.step(action)
        total_reward += reward
        info['epoch'] = epoch
        if done and total_reward+reward < environment.env._max_episode_steps:
            reward = -25
        agent.update(observation_binned, reward, done, info)
        current_state = observation_binned
    print(f"{epoch} {total_reward}")
