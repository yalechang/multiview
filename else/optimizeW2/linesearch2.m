%% This function find step size that satisfies Wolfe conditions in the line search

%%
function alphas = linesearch2(alphamax, w2, w1, s, sigma, c1, c2)
alpha(1) = 0;
alpha(2) = alphamax/2;
i = 2;
while(1)
    if ( getValue( s, sigma, (1-alpha(i)^2)^(1/2)*w2 + alpha(i)*w1) > getValue(s, sigma, w2) + c1*alpha(i)...
           *getgraalpha(s, sigma, w1, w2, 0) || ( getValue( s, sigma, (1-alpha(i)^2)^(1/2)*w2 + alpha(i)*w1)...
           > getValue( s, sigma, (1-alpha(i-1)^2)^(1/2)*w2 + alpha(i-1)*w1) && i>2 ))
       alphas = zoom2(alpha(i-1),alpha(i), w2, w1, s, sigma, c1, c2);
       return;
    end
    if ( abs( getgraalpha(s, sigma, w1, w2, alpha(i)) ) <= -1*c2* getgraalpha(s, sigma, w1, w2, 0))
       alphas = alpha(i);
       return;
    end
    if ( getgraalpha(s, sigma, w1, w2, alpha(i)) >= 0)
        alphas = zoom2(alpha(i),alpha(i-1), w2, w1, s, sigma, c1, c2);
        return;
    end
    alpha(i+1) = (alpha(i) + alphamax)/2;
    i = i+1;
    if ( abs(alphamax - alpha(i)) < 1e-10)
         alphas = alpha(i);
         return;
    end
end