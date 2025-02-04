import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import matplotlib.pyplot as plt


class TradingBot(gym.Env):

    def __init__(self,data, env, amount = 10000):
        self.env = env
        self.amount = amount
        self.data = data
