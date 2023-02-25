import numpy as np


def mean_squared_error(hypothesis: np.ndarray, y: np.ndarray) -> float:
    assert hypothesis.shape == y.shape, ("As dimensões deveriam ser iguais."
                                         f"({hypothesis.shape=}; {y.shape=})")
    
    return ((hypothesis - y) ** 2).sum() / (2 * hypothesis.shape[0])


def feature_normalization(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    averages: np.ndarray = X.mean(axis=0)
    standard_deviations: np.ndarray = X.std(axis=0)
    
    normalized_X = (X - averages) / standard_deviations
    
    return (normalized_X, averages, standard_deviations)


def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, epochs: int) -> tuple[np.ndarray, np.ndarray]:
    # Número de amostras
    m = X.shape[0]
    J_history = np.zeros(shape=(epochs,))
    
    for epoch in range(epochs):
        hypothesis: np.ndarray = X.dot(theta)
        
        for i in range(theta.size):
            # Derivada da função Mean Squared Error
            MSE_derivative = (1/m) * ((hypothesis - y) * X[:, i]).sum()
            theta[i] = theta[i] - (alpha * MSE_derivative)
        
        J_history[epoch] = mean_squared_error(X.dot(theta), y)
    
    return (theta, J_history)


def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # pinv(X' * X) * X' * y
    X_transposed = X.T
    return np.linalg.pinv(X_transposed.dot(X)).dot(X_transposed).dot(y)