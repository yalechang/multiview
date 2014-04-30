function dx = SC(w,cluster)
% Input
%       w : N x N affinity matrix 
% cluster : desired number of clusters
% Output
%      dx : clustering result    
    
    %%% compute Laplacian matrix %%%
    D=diag(sum(w,1));
    L=D-w;
    %%% eigen decomposition %%%
    OPTS.disp = 0;
    [f, D_] = eigs((L+L')/2, D, cluster, 'SA', OPTS);%generalized eigenproblem
    dx = kmeans(f,cluster,'EmptyAction','drop','Replicates',50);
    clear D_;
    clear D;
    clear L;
    clear f;
