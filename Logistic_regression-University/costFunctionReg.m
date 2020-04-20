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
q = X * theta ; 
q  = sigmoid(q) ; 
p = log(q) ; 
r = 1 - q ; 
r = log(r) ; 
r = (1-y) .* r ; 
p = y .* p ; 
s = p + r ;
s = -s ; 
s = s/m ; 
B = theta .^2 ; 
l = lambda / m ; 
B = B * (l/2) ;  
B(1) = 0 ;
J = sum(s) + sum(B) ; 

A= (X' * (q - y ))/m ;
[g,h] = size(theta)  ; 

C = zeros(g,1) ; 
for u = 2 : g
	C(u)  = l * theta(u) ; 
end
grad = A + C ; 





% =============================================================

end
