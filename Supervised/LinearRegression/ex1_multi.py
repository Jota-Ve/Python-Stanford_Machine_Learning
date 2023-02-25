
import numpy as np
import matplotlib.pyplot as plt
import utils


def loading_data(path: str):
    print('################## Loading data ##################\n')

    ## Load Data
    data = np.loadtxt(path, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = len(y)

    # Print out some data points
    print('Showing 10 first examples from the dataset: \n')
    print(f' x = {X[:10]}\n\ny = {y[:10]} \n')
    return (X, y, m)


########################## Gradient Descent ##########################
def running_gradient_descent(X :np.ndarray, y: np.ndarray, alpha: float = 0.0009, num_iters: int = 5000):
    #region Instructions: We have provided you with the following starter
    #               code that runs gradient descent with a particular
    #               learning rate [alpha]. 
    #
    #               Your task is to first make sure that your functions - 
    #               computeCost and gradientDescent already work with 
    #               this starter code and support multiple variables.
    #
    #               After that, try running gradient descent with 
    #               different values of alpha and see which one gives
    #               you the best result.
    #
    #               Finally, you should complete the code at the end
    #               to predict the price of a 1650 sq-ft, 3 br house.
    #
    # Hint: By using the 'hold on' command, you can plot multiple
    #       graphs on the same figure.
    #
    # Hint: At prediction, make sure you do the same feature normalization.
    #endregion

    print('Running gradient descent ...\n')
    # Init Theta and Run Gradient Descent 
    theta = np.zeros((3,))
    theta, J_history = utils.gradient_descent(X=X, y=y, theta=theta, alpha=alpha, epochs=num_iters)

    # Plot the convergence graph
    plt.figure()
    plt.plot(range(len(J_history)), J_history, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')

    # Display gradient descent's result
    print(f'Theta computed from gradient descent: \n{theta}\n\n')
    plt.show()
    
    return theta


def predict(mu, sigma, theta):
    # Estimate the price of a 1650 sq-ft, 3 br house
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.
    price =  np.append([1], (np.array([1650, 3]) - mu)/sigma).dot(theta)
    print(f'Predicted price of a 1650 sq-ft, 3 br house [using gradient descent]:\n {price}\n')
    return price


def gradient_descent_example():
    X, y, m = loading_data('ex1data2.txt')

    # Scale features and set them to zero mean
    print('Normalizing Features ...\n')
    X, mu, sigma = utils.feature_normalization(X)

    # Add intercept term to X
    X = np.append(np.ones((m, 1)), X, axis=1)
    # Training
    theta = running_gradient_descent(X, y)
    #Predicting
    price = predict(mu, sigma, theta)



########################## Normal Equation ##########################
def running_normal_equation(X, y):
    # Calculate the parameters from the normal equation
    print('Solving with normal equations...\n')
    theta = utils.normal_equation(X, y)

    # Display normal equation's result
    print(f'Theta computed from the normal equations: \n{theta} \n\n')
    return theta


def predict_with_normal_equation(theta): 
    #region Instructions: The following code computes the closed form 
    #               solution for linear regression using the normal
    #               equations. You should complete the code in 
    #               normalEqn.m
    #
    #               After doing so, you should complete this code 
    #               to predict the price of a 1650 sq-ft, 3 br house.
    #endregion

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.append([1], [1650, 3]).dot(theta)
    print(f'Predicted price of a 1650 sq-ft, 3 br house [using normal equations]:\n {price}\n')
    

def normal_equation_example():
    ## Load Data
    X, y, m = loading_data('ex1data2.txt')

    # Add intercept term to X
    X = np.append(np.ones((m, 1)), X, axis=1)

    theta = running_normal_equation(X, y)

    predict_with_normal_equation(theta)
    


def main():
    gradient_descent_example()
    normal_equation_example()
    
if __name__ == '__main__':
    main()

