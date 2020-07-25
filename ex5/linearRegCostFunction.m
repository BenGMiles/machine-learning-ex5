function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Get hypothis matrix
hypothesis = X * theta;

% Compute cost J
sqrErrors = (hypothesis - y) .^ 2;
costJ = 1/(2 * m) * sum(sqrErrors);

% Compute regularized J
% We don't need to regularize theta(0)
regTheta = [0;theta(2:end, :);]

J = costJ + (lambda / (2 * m)) * (regTheta' * regTheta);
grad = (1/m) * X' * (hypothesis - y) + (lambda / m) * regTheta;
grad(1, 1) = (1/m) * X'(1, :) * (hypothesis - y);




% =========================================================================

grad = grad(:);

end
