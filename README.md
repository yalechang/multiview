##############################################################################
function W = overoptimize(s, sigma, dim1, dim2, alphamax, c1, c2, converge)

% This is the main function used to optimize the objective function
% 
% Parameters
% ----------
% s:    struct('A',[],'c',[]),
% sigma:    kernel width in Gaussian kernel,
% dim1: original dimension of data,
% dim2: reduced dimension of data,
% alphamax: the maximum alpha,
% c1:   constant for first wolfe condition,
% c2:   constant for second wolfe condition,
% converge: the threshold for convergence.

% Returns
% -------
% W: projection matrix with dimensions dim1 x dim2


For the struct s, s(index).c is the parameter in objective function for xi and 
xj. s(index).A is the d-by-d rank one matrix defined as (xi-xj)T(xi-xj).

##############################################################################

function [w valuevector]= optimizeW(s, sigma, w, W, alphamax, c1, c2,...
converge,valuevector)

% This function optimize the objective function w.r.t vector w

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


