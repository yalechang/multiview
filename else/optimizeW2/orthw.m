%% This function makes input vector w orthogonal to all the column vectors in input matrix W  

%%
function wout = orthw(w, W)
[n m] = size(W);

% If w is the first vector, then just normalize w
if (isempty(W))
    wout = w/norm(w);

% If there're existing vectors in matrix W, make w to be orthogonal to all
% the column vectors in W and then normalize w
else
    for i = 1:m
        w = w - w'*W(:,i)* W(:,i);
    end
    wout = w/norm(w);
end