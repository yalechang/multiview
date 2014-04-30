%% This function compute the contribution of vector w to the value of the objective function
% Note Gaussian kernel is applied in this case

%%
function value = getValue(s,sigma,w)
l = length(s);
value = 0;
for i = 1:l
    value = value + s(i).c*exp(-1*trace(w'*s(i).A*w)/(2*sigma^2));
end
