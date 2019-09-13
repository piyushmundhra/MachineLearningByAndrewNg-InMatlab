function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
mY = size(y, 1);
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Adding bias unit to X
X = [ones(m,1) X];

%Feedforward
z2 = (X * Theta1');                         %Activation values layer 2
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];    %Adding bias unit to layer 2 activation values
a3 = sigmoid(a2 * Theta2');                 %Calculating output values (layer 3)
    
%Changing format of y
recoY = zeros(num_labels,m);
for i = 1:mY
    tempY = y(i);
    recoY(tempY, i) = 1;
end
recoY = recoY';
%Computing cost without regularization
J = (-1/m) * (sum(sum(recoY .* log(a3) + (1-recoY) .* log(1-a3))));

%Regularizing cost
J = J + (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2)));
% -------------------------------------------------------------

for t = 1:m
    %Performing feedforward for example i
        a1NN = X(t,:);                      %Layer 1 (input layer) -> layer 2 (hidden layer)   
        a1NN = a1NN';
            z2NN = Theta1 * a1NN;
            a2NN = sigmoid(z2NN); 
        a2NN = [1; a2NN];                   %Layer 2 (hidden layer) -> layer 3 (output layer)
            z3NN = Theta2 * a2NN;
            a3NN = sigmoid(z3NN);
    %Calculating delta3's (map from layer 2 to layer 3)
        delta3 = a3NN - recoY(t,:)';
    %Calculating delta2's (map from layer 1 to layer 2)
        delta2 = (Theta2' * delta3) .* sigmoidGradient([1; z2NN]);
    %Accumulating raw gradients  
        Theta1_grad = Theta1_grad + delta2(2:end) * a1NN';
        Theta2_grad = Theta2_grad + delta3 * a2NN';
end
    %Calculating unregularized gradients
        Theta1_grad = (1/m) * Theta1_grad;
        Theta2_grad = (1/m) * Theta2_grad;
% =========================================================================
% Regularizing gradients
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end));
% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
