function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    hx = theta(1) + theta(2)*X(:,2);
    
    double temp0;
    double temp1;
    
    temp0 = (theta(1) - alpha * (hx - y)' * X(:,1) / m);
    temp1 = (theta(2) - alpha * (hx - y)' * X(:,2) / m);
    
    theta(1) = temp0(1);
    theta(2) = temp1(1);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
end
    
end
