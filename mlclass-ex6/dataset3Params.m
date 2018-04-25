function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

matC = [.01 .03 .1 .3 1 3 10 30];
matSigma = [.01 .03 .1 .3 1 3 10 30];

C_aux = 0;
sigma_aux = 0;

C = 0.0001;
sigma = 0.0001;

minError = 999999999999999;

for k = 1:size(matC)(2)
    for l = 1:size(matSigma)(2)
        %model = svmTrain(X, y, matC(1,k), @(x1, x2) gaussianKernel(Xval(:,1), Xval(:,2), matSigma(1,l)));
        model = svmTrain(X, y, matC(1,k), @(x1, x2) gaussianKernel(x1, x2, matSigma(1,l)));
        error = mean(double(svmPredict(model, Xval) ~= yval));
        if error < minError 
            minError = error;
            C_aux = matC(1,k);
            sigma_aux = matSigma(1,l);
            printf('error min C = %f, sigma = %f, error = %f\n', C_aux, sigma_aux, error);
        end
    endfor
endfor    

C = C_aux;
sigma = sigma_aux;

printf('error min C = %f, sigma = %f', C, sigma);

% =========================================================================

end
