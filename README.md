# 🂡 Blackjack Reinforcement Learning Agent

This project implements a Deep Q-Learning agent to play Blackjack using OpenAI Gym's `Blackjack-v1` environment. The model incorporates custom reward shaping, action heuristics, and training optimizations to improve decision-making in a stochastic card game environment.

---

## 🧠 Features

* ✅ **Deep Q-Network (DQN)** with LayerNorm + LeakyReLU
* 🎯 **Custom reward shaping** using bust probability and dealer dynamics
* 🎲 **Custom action heuristics** to guide early exploration
* 🧪 **Testing module** to evaluate trained performance
* 📂 **Model checkpointing** based on best episode reward
* 📉 Tracks **epsilon-greedy decay** and supports early stopping logic

---

## 📁 Project Structure

```
📆 Blackjack-RL-Agent
🔼👨‍💼 model.py            # QNetwork architecture (with LayerNorm and LeakyReLU)
🔼📅 train.py            # Training loop and agent logic
🔼🔧 utils.py            # Custom reward shaping and state encoding
🔼🔢 test.py             # Evaluation script to test the trained model
🔼📥 best_black_jack.pth # Saved best-performing model
🔼📄 README.md           # Project documentation
🔼📁 requirements.txt    # Python dependencies
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python train.py
```

This will:

* Train the DQN agent for `20,000` episodes
* Save the best model to `best_black_jack.pth`

### 3. Test the Agent

```bash
python test.py
```

This will:

* Load the trained model
* Run it over `10,000` episodes
* Output win/draw/lose rates and average reward

---

## 📊 Performance

| Metric      | Value (Approx.) |
| ----------- | --------------- |
| Win Rate    | \~43%           |
| Draw Rate   | \~9%            |
| Loss Rate   | \~48%           |
| Avg. Reward | \~ -0.04        |

---

## ⚙️ Model Details

* Input: Encoded state (player sum, dealer card, usable ace)
* Architecture: `state_dim -> 32 -> 16 -> 8 -> action_dim`
* Activations: `LeakyReLU`
* Normalization: `LayerNorm` after linear layers
* Optimizer: `Adam`
* Loss: `MSE`
* Discount Factor: `γ = 0.99`
* Replay Buffer: `deque`
* Epsilon Decay: `ε = 1.0 → 0.01`

---

## 🧹 Customizations

* **Reward shaping**: Encourages actions based on:
  * Player bust chance
* **Action heuristics**: Uses rule-based logic during early episodes
* **Model checkpointing**: Saves best model based on highest episode reward

---

## 🔮 Future Improvements

* Prioritized Experience Replay (PER)
* Dueling DQN (tested but overfitted)
* Double DQN (partially integrated)
* Monte Carlo Tree Search (MCTS) hybridization
* Better domain-specific heuristics

---

## 📜 License

MIT License

---

## 🙌 Acknowledgments

* OpenAI Gymnasium `Blackjack-v1`
* PyTorch
* Chatgpt for helping me with the code and rules of blackjack

---

Enjoy the game, and may the odds be ever in your favor 🎲🂡!
