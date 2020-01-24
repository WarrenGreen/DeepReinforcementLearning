import gym

from drl.agents.reinforce import REINFORCE

epochs = 10000
environment = gym.make('CartPole-v0')
environment.seed(543)
agent = REINFORCE(environment)
log_interval = 10

running_reward = 0

for epoch in range(epochs):
    current_state = environment.reset()
    agent.reset()
    done = False
    ep_reward = 0
    while not done:
        environment.render()
        action = agent.predict(current_state)
        observation, reward, done, info = environment.step(action)
        ep_reward = agent.update(observation, reward, done, info)
        current_state = observation

    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    if epoch % log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            epoch, ep_reward, running_reward))
    if running_reward > environment.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, ep_reward))
        break
