clear all, close all, clc;

x = textread('/Users/changyale/dataset/mfeat/mfeat-fou');
x_mean = repmat(mean(x),2000,1);
x_std = repmat(std(x),2000,1);
X = (x-x_mean)./x_std;

n = 2000;
d = 76;
G = sum((X.*X),2);
Q = repmat(G,1,n);
Rt = repmat(G',n,1);
dists = Q + Rt - 2*(X*X');
dists = dists-tril(dists);
dists = reshape(dists,n^2,1);
sig2 = sqrt(median(dists(dists>0)));
sig2^2
