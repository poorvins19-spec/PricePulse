
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from collections import deque

# Load dataset
df = pd.read_csv("dataset/pricing_data.csv").head(300)

# DQN settings
ACTIONS = [-0.05, 0, 0.05]
GAMMA = 0.95
EPSILON = 1.0
BATCH_SIZE = 32

# Prepare states
states = df[
    ["current_price", "demand", "inventory", "competitor_price"]
].values.astype("float32")

mean = states.mean(axis=0)
std = states.std(axis=0)
std[std == 0] = 1
states = (states - mean) / std

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(3, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

memory = deque(maxlen=1000)

# Reward calculation
def calculate_reward(row, action):
    price = row["current_price"]
    quantity = row["sales_quantity"]
    inventory = row["inventory"]
    competitor = row["competitor_price"]

    new_price = price * (1 + action)
    estimated_demand = quantity * (1 - 1.2 * action)
    sold = min(max(0, estimated_demand), inventory)

    revenue = new_price * sold
    penalty = max(0, estimated_demand - inventory) * new_price * 0.5
    competitor_penalty = max(0, new_price - competitor * 1.10) * sold * 0.1

    return (revenue - penalty - competitor_penalty) / 1000


# Training
print("PRICEPULSE - DQN TRAINING", flush=True)
print("-------------------------", flush=True)

for episode in range(5):
    total_reward = 0

    for i in range(len(df)):
        state = states[i]

        if random.random() < EPSILON:
            action_index = random.randrange(3)
        else:
            q_values = model(state.reshape(1, 4), training=False).numpy()[0]
            action_index = int(np.argmax(q_values))

        action = ACTIONS[action_index]
        reward = calculate_reward(df.iloc[i], action)

        next_state = states[(i + 1) % len(df)]

        memory.append((state, action_index, reward, next_state))
        total_reward += reward

        # Train once every 10 transactions
        if len(memory) >= BATCH_SIZE and i % 10 == 0:
            batch = random.sample(memory, BATCH_SIZE)

            s = np.array([x[0] for x in batch])
            a = np.array([x[1] for x in batch])
            r = np.array([x[2] for x in batch])
            ns = np.array([x[3] for x in batch])

            q_current = model(s, training=False).numpy()
            q_next = target_model(ns, training=False).numpy()

            targets = q_current.copy()

            targets[np.arange(BATCH_SIZE), a] = (
                r + GAMMA * np.max(q_next, axis=1)
            )

            model.train_on_batch(s, targets)

    EPSILON = max(0.05, EPSILON * 0.9)

    target_model.set_weights(model.get_weights())

    print(
        f"Episode {episode + 1}/5 completed | "
        f"Reward: {total_reward:.2f} | "
        f"Epsilon: {EPSILON:.3f}",
        flush=True
    )

# Save model
model.save("pricepulse_dqn.keras")

# Generate recommendations
results = []

for i, row in df.iterrows():
    q_values = model(
        states[i].reshape(1, 4), training=False
    ).numpy()[0]

    action = ACTIONS[int(np.argmax(q_values))]

    results.append({
        "product_id": row["product_id"],
        "current_price": row["current_price"],
        "recommended_price": round(
            row["current_price"] * (1 + action), 2
        ),
        "DQN_action": action
    })

pd.DataFrame(results).to_csv(
    "dqn_price_recommendations.csv", index=False
)

print("\nDQN training completed!")
print("Model saved: pricepulse_dqn.keras")
print("Recommendations saved: dqn_price_recommendations.csv")
