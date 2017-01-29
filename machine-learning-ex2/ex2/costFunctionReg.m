function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

sig = sigmoid(X*theta);
%disp ("The value of size(x, theta, sig are:"), disp (size(X)) , disp (size(theta)), disp (size(sig))

thetaCost = 0;

for i = 2:rows(theta);
  thetaCost += theta(i)^2;
endfor
thetaCost = lambda*thetaCost/(2*m);

J = (-y' * log(sig) - (1-y')*log(1-sig))/m + thetaCost;
grad = (X'*(sig - y))/m;

for i = 2:rows(grad);
  grad(i) += lambda*theta(i)/m;
endfor


% =============================================================

end
