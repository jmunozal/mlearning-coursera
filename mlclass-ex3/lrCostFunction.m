function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% sin regularizar
%J = ((-y .* arrayfun(@log, (arrayfun (@sigmoid, theta'*X')))) - ((-y+1) .* arrayfun(@log, 1-(arrayfun (@sigmoid, theta'*X'))))); 
%grad = X' * (arrayfun (@sigmoid, theta'*X') .- y)
%grad = grad ./ m;

%[w r]=size(X);
%[t1 t2]=size(theta);
%[y1 y2]=size(y);
%printf("X = [%i %i]\n", w, r);
%printf("theta = [%i %i]\n", t1, t2);
%printf("y = [%i %i]\n", y1, y2);
afun = arrayfun(@log, (arrayfun (@sigmoid, X*theta)));
afun2 = arrayfun(@log, 1-(arrayfun (@sigmoid, X*theta)));
%printf("afun = [%i %i]\n", size(afun)(1), size(afun)(2));
%printf("afun2 = [%i %i]\n", size(afun2)(1), size(afun2)(2));
J = (-y .* afun) - ((1-y) .* afun2); 
J = sum(J(:));
J = J / m;
J = J + ((lambda / (2 * m)) * (sum(theta(2:end).^2)));

grad = X' * ((arrayfun (@sigmoid, X*theta)) - y);
%printf("grad = [%i %i]\n", size(grad)(1), size(grad)(2));
grad = grad / m;
temp = theta;
temp(1) = 0;
temp = temp .* (lambda / m);
grad = grad + temp;

% =============================================================

grad = grad(:);

end
