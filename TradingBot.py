import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import matplotlib.pyplot as plt


class TradingBot(gym.Env):

    def __init__(self,data, env, amount = 10000):
        super(TradingBot, self).__init__()
        self.env = env
        self.initial_amount = amount
        self.initial_balance = self.initial_amount
        self.balance = self.initial_amount
        self.current_step = 0
        self.data = data
        self.state = 0
        self.stocks_owned = 0
        self.stock_price_history = data['Close']
        self.data = data['Date']


        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,))  # (Action, Amount) where Action: -1: Buy, 0: Hold, 1: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

        self.done = False
        self.render = pd.DataFrame()
        self.current_portfolio_value = 0


    def reset(self,seed= None, options=None):
        self.current_step = 0
        self.stocks_owned = 0
        self.balance = self.initial_amount
        self.done = False
        self.current_portfolio_value = self.initial_amount
        return self._get_obs(), {}


    def _get_obs(self):
        return np.array([self.stock_price_history[self.current_step]])

    def step(self, action):
        prev_portfolio_value = self.balance if self.current_step == 0 else self.current_portfolio_value + self.stocks_owned * self.stock_price_history[self.current_step]
        current_price = self.stock_price_history[self.current_step]
        amount = int(self.initial_balance*action[1] / current_price)

        if action[0] > 0:
            amount = min(self.initial_balance*action[1] / current_price, self.balance / current_price)
            if self.balance >= current_price*amount:
                self.stocks_owned += amount
                self.balance -= amount * current_price

        elif action[0] < 0:
            amount = min(amount, self.stocks_owned)
            if self.stocks_owned > 0:
                self.stocks_owned -= amount
                self.balance += current_price * amount

        current_portfolio_value = self.balance + self.stocks_owned * current_price
        excess_return = current_portfolio_value - prev_portfolio_value
        risk_free_rate = 0.02
        std_deviation = np.sqrt(self.stock_price_history[:self.current_step+1])
        sharp_ratio = (excess_return - risk_free_rate) / std_deviation if std_deviation != 0 else 0
        reward = sharp_ratio
        


        self.current_step += 1


    def render(self):
        pass

    def seed(self, seed):
        pass
