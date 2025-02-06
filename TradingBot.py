import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

class TradingBot(gym.Env):

    def __init__(self,data, amount = 10000):
        super(TradingBot, self).__init__()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        #self.env = env
        self.initial_amount = amount
        self.initial_balance = self.initial_amount
        self.balance = self.initial_amount
        self.current_step = 0
        self.data = data
        self.state = 0
        self.stocks_owned = 0
        self.stock_price_history = data['Close']
        self.volume = data['Volume']
        self.date = data['date']


        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), shape=(2,))  # (Action, Amount) where Action: -1: Buy, 0: Hold, 1: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))

        self.done = False
        self.render_df = pd.DataFrame()
        self.current_portfolio_value = 0


    def reset(self,seed= None, options=None):
        self.current_step = 0
        self.stocks_owned = 0
        self.balance = self.initial_amount
        self.done = False
        self.current_portfolio_value = self.initial_amount
        return self._get_obs(), {}


    def _get_obs(self):
        return np.array([self.stock_price_history[self.current_step],self.volume[self.current_step]])

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
        std_deviation = std_deviation.iloc[0]
        if std_deviation != 0:
            pass
        else:
            std_deviation = 0
        sharp_ratio = (excess_return - risk_free_rate) / std_deviation
        reward = sharp_ratio

        self.render(action,amount,current_portfolio_value)
        obs = self._get_obs()

        self.current_step += 1

        if self.current_step == len(self.data["Close"]):
            done = True
        else:
            done = False

        self.done = done
        info = {}

        return obs, reward, done, False, info


    def render(self,action, amount, current_portfolio_value, mode = None):
        current_date = self.date[self.current_step]
        today_action = 'buy' if action[0] > 0 else 'sell'
        current_price = self.stock_price_history[self.current_step]

        if mode == 'human':
            print(f"Step:{self.current_step}, Date: {current_date}, Market Value: {current_portfolio_value:.2f}, Balance: {self.balance:.2f}, Stock Owned: {self.stocks_owned}, Stock Price: {current_price:.2f}, Today Action: {today_action}:{amount}")
        else:
            pass
        dict = {
            'Date': [current_date], 'market_value': [current_portfolio_value], 'balance': [self.balance],
            'stock_owned': [self.stocks_owned], 'price': [current_price], 'action': [today_action], 'amount': [amount]
        }

        step_df = pd.DataFrame.from_dict(dict)
        self.render_df = pd.concat([self.render_df, step_df], ignore_index=True)

    def renderAll(self):
        return self.render_df

    def seed(self, seed):
        pass


stock = yf.Ticker("PYPL")
history = stock.history(start='2019-3-1', end='2020-3-1')
history['date'] = pd.to_datetime(history.index)
env = TradingBot(history)

model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000, progress_bar=True)
model.save("ppo_aapl")

stock = yf.Ticker("TSLA")
history = stock.history(start='2020-3-1', end='2021-3-1')
history['date'] = pd.to_datetime(history.index)
env = TradingBot(history)

model = PPO.load("ppo_aapl",env=env)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(len(history['Close'])):
    action, _state = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)

dataF = env.renderAll()
print(dataF)
#xValue = np.linspace(0,100,num=len(dataF['market_value']))

plt.plot(dataF["Date"],dataF['market_value'])
plt.show()
