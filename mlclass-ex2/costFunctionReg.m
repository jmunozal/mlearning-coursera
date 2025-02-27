function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
R = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% COST

for i = 1:m
	J = J + -y(i) * log(sigmoid(((theta)'*X(i,:)'))) - ((1-y(i))* log(1-sigmoid((theta)'*X(i,:)')));
endfor

for k = 2:size(theta)
	R = R + theta(k)^2;
endfor

R = (lambda * R) / (2 * m);
J = (J / m) + R;

% GRADIENT

for j = 1:size(theta)
	for i = 1:m
		grad(j) = grad(j) + (sigmoid((theta)'*X(i,:)') - y(i)) * X(i,j);
	endfor
endfor
	
grad = grad ./ m;

for k = 2:size(theta)
	grad(k) = grad(k) + (lambda * theta(k) / m);
endfor

% =============================================================

end
