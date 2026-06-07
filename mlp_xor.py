import numpy as np

# Activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_weights(input_size, hidden_size, output_size):
    np.random.seed(0)
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Step 1: Hidden layer output
# x = sum(xi × wi) + bias
# Oj = sigmoid(x)


def hidden_layer_output(X, W1, b1):
    x = X @ W1 + b1
    Oj = sigmoid(x)
    return Oj

# Step 2: Output layer output
# x = sum(xj × wj) + bias
# Ok = sigmoid(x)


def output_layer_output(Oj, W2, b2):
    x = Oj @ W2 + b2
    Ok = sigmoid(x)
    return Ok

# Step 3: Output error (delta)
# delta k = Ok × (1 - Ok) × (t - Ok)


def output_error(Ok, t):
    delta_k = Ok * (1 - Ok) * (t - Ok)
    return delta_k

# Step 4: Update hidden-output weights
# weight change = learning rate × Oj × delta k
# w(jk)(t+1) = w(jk)(t) + weight change(jk)


def update_hidden_output_weights(W2, b2, Oj, delta_k, lr):
    delta_W2 = Oj.T @ delta_k
    delta_b2 = delta_k.sum(axis=0, keepdims=True)
    W2 = W2 + lr * delta_W2
    b2 = b2 + lr * delta_b2
    return W2, b2

# Step 5: Hidden error (delta)
# deltaj = Oj × (1 - Oj) × sum(delta k × w(jk))


def hidden_error(Oj, delta_k, W2):
    delta_j = Oj * (1 - Oj) * (delta_k @ W2.T)
    return delta_j

# Step 6: Update input-hidden weights
# weight change(ij) = learning rate × xi × deltaj
# w(ij)(t+1) = w(ij)(t) + weight change(ij)


def update_input_hidden_weights(W1, b1, X, delta_j, lr):
    delta_W1 = X.T @ delta_j
    delta_b1 = delta_j.sum(axis=0, keepdims=True)
    W1 = W1 + lr * delta_W1
    b1 = b1 + lr * delta_b1
    return W1, b1


def compute_loss(t, Ok):
    return np.mean((t - Ok) ** 2)


def train(X, t, W1, b1, W2, b2, epochs=100000, lr=0.5):
    for epoch in range(epochs):
        # Step 1 & 2: Forward pass
        Oj = hidden_layer_output(X, W1, b1)
        Ok = output_layer_output(Oj, W2, b2)

        loss = compute_loss(t, Ok)

        # Step 3: Output error
        delta_k = output_error(Ok, t)

        # Step 4: Update hidden → output weights
        W2, b2 = update_hidden_output_weights(W2, b2, Oj, delta_k, lr)

        # Step 5: Hidden error
        delta_j = hidden_error(Oj, delta_k, W2)

        # Step 6: Update input → hidden weights
        W1, b1 = update_input_hidden_weights(W1, b1, X, delta_j, lr)

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss:.6f}")

    return W1, b1, W2, b2


def test(X, t, W1, b1, W2, b2):
    Oj = hidden_layer_output(X, W1, b1)
    Ok = output_layer_output(Oj, W2, b2)

    print("\n--- XOR Test Results ---")
    print(f"{'Input':<10} {'Target':<10} {'Output':<10} {'Rounded'}")
    print("-" * 42)
    for i in range(len(X)):
        pred = Ok[i, 0]
        print(f"{str(X[i]):<10} {t[i, 0]:<10} {pred:<10.4f} {round(pred)}")


def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    t = np.array([[0], [1], [1], [0]])   # t = target

    # MLP 2 input neurons → 4 hidden neurons → 1 output neuron
    W1, b1, W2, b2 = init_weights(input_size=2, hidden_size=4, output_size=1)
    W1, b1, W2, b2 = train(X, t, W1, b1, W2, b2)
    test(X, t, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
