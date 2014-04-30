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

% Initialize w using a random vector with unit length
w = rand(dim1,1);
w = w / norm(w);
valuevector  = [];

% Optimize the objective function w.r.t vector w
[w valuevector]= optimizeW(s, sigma, w, [], alphamax, c1, c2, converge, valuevector);

% Store w as the first column vector in projection W, which will be used as
% the output
W(:,1) = w;

% Update 'c' in the struct
s = updatec(s, sigma, w);

% Crate wait bar in order to look at convergence time
h = waitbar(0,'waiting...');
for i = 2:dim2
    w = rand(dim1,1);
    [w valuevector]= optimizeW(s, sigma, w, W, alphamax, c1, c2, converge, valuevector);
    s = updatec(s, sigma, w);
    W = [W w];
    waitbar(i/dim2);
end
close(h);
plot(valuevector);

