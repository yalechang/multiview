function [w valuevector]= optimizeW(s, sigma, w, W, alphamax, c1, c2, converge, valuevector)

% This function optimizes the objective function w.r.t vector w

% Parameters
% ----------
% s:    struct('A',[],'c',[]),
% sigma:    kernel width in Gaussian kernel,
% w:    w is the new vector that need to be optimized
% W:    W is the matrix consisting of existing vectors,note that w should
%       be orthogonal to the column vectors in W 
% alphamax: maximum value of alpha in line search
% c1:   constant for first wolfe condition,
% c2:   constant for second wolfe condition,
% converge: threshold for convergence.

% Returns
% -------
% w:    updated value of the vector optimizer
% valuevector:  store the values of objective function in each step 

% make w orthogonal to all the column vectors in matrix W and have unit
% length
w = orthw(w,W);
value = 1e10;

% Compute the contribution of vector w to the value of the objective function
newvalue = getValue(s, sigma, w);

while (abs(value - newvalue) > converge)
    value = newvalue;
    gradw = getGradW(s, sigma, w);
    gradw = -1*gradw;
    gradw = orthw(gradw,[W w]);
    % Determine the step size of line search
    alphas = linesearch2(alphamax, w, gradw, s, sigma, c1, c2);
    w = alphas*gradw + (1-alphas^2)^(1/2)*w;
    newvalue = getValue(s, sigma, w);
    valuevector = [valuevector newvalue];
    disp('newvalue:');
    disp(newvalue);
end
