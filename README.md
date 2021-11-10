# snake_AI

Using machine learning algorithms to teach a A.I. how to play the classic game snake.
Implements reinforced learning and Deep Q Networks to maximize the cumulative reward based on the current state of the environment.

## snake_pygame

Made using pygame module
-play_step(action)
--returns reward, done(game_over bool), score

## Model

Made with PyTorch

Linear_QNet (DQN) uses a feed foreward neural network with one hidden layer size 256 and a Relu activation function.

Q Value = Quality of action
1. Init Q Value (= init model)
2. Choose action (model.predict) ->returns action based on maximized cumulative reward
3. Perform action
4. Measure Reward
5. Update Q value (+train model)
6. Repeat

## Agent

#### Main file

###### Training method:

-state = get_state(from game)
-action = get_move(based on state of game environment):
  ->model.predict(best action based on reward)
  
 reward, game_over, score = game.play_step(action)
 new_state = get_state(game)
 
 add single move to short term memory
 once game_over = True add batch of short term memories to long term memory
 
 ###### model.train()
 
How new Q-Value is calculated

s = state

a = action

lr = learning rate

gamma = discount rate

R = reward

** Function **

NewQ(s,a) = Q(s,a) + lr[R(s,a) + gamma*maxQ'(s',a') - Q(s,a)

Uses optimizer.Adam() method an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.
 
Criterion of the Mean Squared error is pair with the built in backwards() method to complete the training process.
