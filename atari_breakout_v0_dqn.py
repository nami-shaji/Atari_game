
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from google.colab import drive
import gym.wrappers
# Creating a folder in Google Disk
drive.mount('/content/gdrive', force_remount=True)



# Observation (wrapped)
env = make_atari('BreakoutNoFrameskip-v4')

# We use wrap_deepmind to pre-process the image into grayscale, and to take 4
# frames and collapse them. This is to speed up our learning rate.
env = wrap_deepmind(env, frame_stack=True, scale=True)
obs = np.array(env.reset())



num_actions = 4

# Our Deep Learning RL Model
def create_q_model():
  # Input layer, 84 x84 pixels, 4 frames stacked
  inputs = layers.Input(shape=(84, 84, 4,))

  layer1 = layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
  layer2 = layers.Conv2D(64, 4, strides=2, activation='relu')(layer1)
  layer3 = layers.Conv2D(64, 3, strides=1, activation='relu')(layer2)
  layer4 = layers.Flatten()(layer3)
  layer5 = layers.Dense(512, activation='relu')(layer4)

  # Output layer
  action = layers.Dense(num_actions, activation='linear')(layer5)

  return keras.Model(inputs=inputs, outputs=action)

# The first model makes the predictions for Q-Values which are used to make an action
model = create_q_model()

# Target Model
model_target = create_q_model()

gamma = 0.99 # Discount facotr for past rewards

# Setting epsilon decay parameters
epsilon = 1.0

epsilon_max = 1.0
epsilon_min = 0.2

epsilon_interval = (epsilon_max - epsilon_min)

# Number of frames for exploration
epsilon_greedy_frames = 1000000.0

# Number of frames to take random action and observe output
epsilon_random_frames = 50000

# Maximum Replay Buffer volume
max_memory_length = 190000

# Size of batch taken from replay buffer
batch_size = 32
max_steps_per_episode = 10000

# Train the model after 20 actions
update_after_actions = 20

# How often to update the target network
update_target_network = 10000

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

loss_function = keras.losses.Huber()

# Experience replay buffers
env = make_atari('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=True, scale=True)


## These arrays will store all state, actions and results from actions
## and will be used for learning and improving
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []

episode_score_history = []

loss_array = []
score_array = []

episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

while True: # Until solved
  state = np.array(env.reset())

  episode_reward = 0

  for timestamp in range(1, max_steps_per_episode):
    frame_count+=1

    # Use epsilon-greedy for exploration
    if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
      # Make a random action
      action = np.random.choice(num_actions)
    else:
      # Predict action Q-Values from environment state
      state_tensor = tf.convert_to_tensor(state)
      state_tensor = tf.expand_dims(state_tensor, 0)
      action_probs = model(state_tensor, training=False)

      # Take best action
      action = tf.argmax(action_probs[0]).numpy()

    # Decay the probability of choosing a random action
    if frame_count < epsilon_greedy_frames:
      epsilon -= epsilon_interval / epsilon_greedy_frames
      epsilon = max(epsilon, epsilon_min)

    # Apply the action in the environment
    state_next, reward, done, _ = env.step(action)
    state_next = np.array(state_next)
    
    score_array.append(reward)
    episode_reward += reward

    # Save actions and states in replay buffer - to learn from
    action_history.append(action)
    state_history.append(state)
    state_next_history.append(state_next)
    done_history.append(done)
    rewards_history.append(reward)
    state = state_next

    # Update every 20th frame and once batch size is over 32
    if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
      # Get indicies of samples for replay buffer
      indicies = np.random.choice(range(len(done_history)), size=batch_size)

      # Using list comprehension to sample from replay buffer
      state_sample = np.array([state_history[i] for i in indicies])
      state_next_sample = np.array([state_next_history[i] for i in indicies])
      rewards_sample = [rewards_history[i] for i in indicies]
      action_sample = [action_history[i] for i in indicies]
      done_sample = tf.convert_to_tensor(
          [float(done_history[i]) for i in indicies]
      )

      # Build updated Q-Values for the sampled future states
      # Use the target model for stability
      future_rewards = model_target.predict(state_next_sample)

      # Q value = reward + discount factor * expected future reward
      updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

      # If final frame, set the last value to -1
      updated_q_values = updated_q_values * (1 - done_sample) - done_sample

      # Create a mask so we only calculate loss on the updated Q-Values
      masks = tf.one_hot(action_sample, num_actions)

      with tf.GradientTape() as tape:
        # Train the model on the states and updated Q-Values
        q_values = model(state_sample)

        # Apply the mask to the Q-Value to get the Q-Value for action taken
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

        # Calculate loss between new Q-Value and old Q-Value
        loss = loss_function(updated_q_values, q_action)
        loss_array.append(loss)

      # Backpropagation
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if frame_count % update_target_network == 0:
      model_target.set_weights(model.get_weights())

      # log
      template = "Running Reward: {:.2f} at episode {}, frame count {} epsilon {:.3f}, loss {:.5f}"
      print(template.format(running_reward, episode_count, frame_count, epsilon, loss))

    # Limit state and reward history - otherwise training session will crash
    if len(rewards_history) > max_memory_length:
      del rewards_history[:1]
      del state_history[:1]
      del state_next_history[:1]
      del action_history[:1]
      del done_history[:1]

    if done:
      break
  
  # Update running reward to check condition of solving
  episode_reward_history.append(episode_reward)
  running_reward = np.mean(episode_reward_history)

  episode_count += 1

  # Condition to consider the task solved
  # We chose a mean running reward of 1 as we don't have a machine that is capable
  # of training for long periods of time
  if running_reward > 1: 
    print(f'Solved at episode {episode_count}!')
    break

plt.plot(episode_reward_history)



def make_env():
  env = make_atari('BreakoutNoFrameskip-v4')
  env = wrap_deepmind(env, frame_stack=True, scale=True)
  return env

env = make_env()
env = gym.wrappers.Monitor(env, './vid5_1', force=True)

observation = env.reset()
info = 0
reward_window = []
reward_signal_history = []
epsilon_history = []

hits = []
bltd = 100 # Total amount of blocks to destroy

for i_episode in range(1):
  reward_window = []
  epsilon = 0
  for t in range(4000):
    if epsilon > np.random.rand(1)[0]:
      action = np.random.choice(num_actions)
    else:
      state_tensor = tf.convert_to_tensor(observation)
      state_tensor = tf.expand_dims(state_tensor, 0)
      action_probs = model(state_tensor, training=False)
      action = tf.argmax(action_probs[0]).numpy()

    observation, reward, done, info = env.step(action)
    hits.append(reward)
    reward_window.append(reward)
    if len(reward_window) > 200:
      del reward_window[:1]

    if len(reward_window) == 200 and np.sum(reward_window) == 0:
      epsilon = 0.01
    else:
      epsilon = 0.0001
    
    epsilon_history.append(epsilon)
    reward_signal_history.append(reward)

    if done:
      print(f'Lost one life after {t+1} timesteps.')
      print(info)

      # Plot epsilon and reward signal
      fig, ax=plt.subplots(figsize=(20,3))
      ax.plot(epsilon_history, color='red')
      ax.set_ylabel('Epsilon', color='red', fontsize=14)
      ax2=ax.twinx()
      ax2.plot(reward_signal_history, color='blue')
      ax2.set_ylabel('Reward Signal', color='blue', fontsize=14)
      plt.show()

      epsilon_history = []
      reward_signal_history = []

      bltd = bltd-np.sum(hits)
      hits=[]
      print(f'Bricks left to destroy: {bltd}.')
      if info['ale.lives'] == 0:
        break

      env.reset()

env.close()

