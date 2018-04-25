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

%printf("theta = %i %i\n",size(theta)(1), size(theta)(2));
%printf("X = %i %i\n",size(X)(1), size(X)(2));
%printf("y = %i %i\n",size(y)(1), size(y)(2));

r = (X * theta - y); %12x1

J = (sum((r.^2)) / (2 * m)) + ((lambda / (2*m)) * sum(theta(2:end,:) .^2));
grad = (r' * X) / m;
normalizacion = zeros(size(theta));
normalizacion = theta * lambda / m;
normalizacion(1) = 0;
%printf("normalizacion %f %f\n",normalizacion(1), normalizacion(2));
%printf("grad %d %d\n",size(grad)(1), size(grad)(2));
grad = grad + normalizacion';
% =========================================================================
grad = grad(:);

end
