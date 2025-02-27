function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

for i = 1:m
	%printf("Size (theta) = %i, %i\n",size(theta));
	%printf("Size (X(I)) = %i, %i\n",size(X(i,:)));
	J = J + -y(i) * log(sigmoid(((theta)'*X(i,:)'))) - ((1-y(i))* log(1-sigmoid((theta)'*X(i,:)')));

endfor

J = J/m;

for j = 1:size(theta)
	for i = 1:m
		%printf("grad = %i, %i, %i\n",grad);
		grad(j) = grad(j) + (sigmoid((theta)'*X(i,:)') - y(i)) * X(i,j);
		%printf("grad = %i, %i, %i\n",grad);
	endfor
endfor
	
grad = grad ./ m;

% =============================================================

end
