Deep Reinforcemnet Learning for playing Flappy Bird Game. This is a tensorflow version, the original MxNet version is [here](https://github.com/yenchenlin/DeepLearningFlappyBird)

1. dependences?

- tensorflow >= 1.2
- numpy
- opencv
- pygame

2. how to play the game?
```python
python train.py
```

param `use_double_q` indicate algorithm whether perform original DQN or double DQN

3. score in training step

x: episodes
y: scores in each episode

![img](https://github.com/YannZyl/TF-flappybird/blob/master/images/origin.png)