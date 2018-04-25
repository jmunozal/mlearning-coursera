function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%printf("X[%i,%i]\n",size(X)(1),size(X)(2));
%printf("Theta2[%i] [%i]\n",size(Theta2)(1),size(Theta2)(2));

for i=1:m
    aux = [1 X(i,:)];
    r1 = arrayfun(@sigmoid, aux * Theta1');
    aux2 = [1 r1];
%    printf("aux2[%i,%i]\n",size(aux2)(1),size(aux2)(2));
    [r ix] = max(arrayfun(@sigmoid, aux2*Theta2'),[],2);
    p(i) = ix;
endfor


% =========================================================================


end
