% read the training data

load('bbb.mat');

% initialize the matrices and variables
X = e;
y = f;
m = length(y); % no. of training examples
theta = zeros(2,1); % initial weights(parameters)
iterations = 15; % iterations needed for gradient descent
alpha = 0.1; % learning rate

% plot the data
plot(X, y, 'rx', 'MarkerSize', 10);
title('Training exmaples');
xlabel('Humidity');
ylabel('Temperatures');

% compute the cost function

% adding ones column to X
X = [ones(m,1), X];
J = ComputeCost(X, y, theta);

% running gradient descent
[theta, J_history] = GradientDescent(X, y, theta, alpha, iterations);

% plotting linear regression line
hold on;
plot(X(:, 2), X * theta, '-');
legend('Trainging Data', 'Linear Regression');
hold off;