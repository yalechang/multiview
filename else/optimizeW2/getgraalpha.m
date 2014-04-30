function derialpha = getgraalpha(s, sigma, w1, w2, alpha)
derivw = getGradW(s, sigma, (1-alpha^2)^(1/2)*w2 + alpha*w1);
derialpha = derivw'*w1 + derivw' * w2 * (-1 * alpha) * (1- alpha^2)^(-1/2);