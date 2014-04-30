%% This script is used to test the effectiveness of the proposed optimization algorithm.
%% Nonlinear optimization with orthogonality constraint.
%%
clear all,close all, clc;

l = 10;
MU = (rand(4,l)-0.5)*5;
SIGMA = eye(l)*0.05;
Y(1:25,:) = mvnrnd(MU(1,:),SIGMA,25);
Y(26:50,:) = mvnrnd(MU(2,:),SIGMA,25);
Y(51:75,:) = mvnrnd(MU(3,:),SIGMA,25);
Y(76:100,:) = mvnrnd(MU(4,:),SIGMA,25);

data = [Y randn(100,10)];
label = [[ones(25,1); zeros(75,1)] [zeros(25,1); ones(25,1); zeros(50,1)] [zeros(50,1); ones(25,1); zeros(25,1)] [zeros(75,1); ones(25,1)]];
label = label * label';
data = data * orth(rand(20));
s = struct('A',[],'c',[]);
ind = 1;
for i = 1:99
    for j = i+1:100
       if label(i,j) == 0
            s(ind).c = 1;
            l = data(i,:)-data(j,:);
            s(ind).A = l' * l;
            ind = ind + 1;
        end
    end
end

sigma = 5;
tic
W = overoptimize(s, sigma, 20, 10, 0.1, 1e-4, 0.9, 1e-4);
toc