function alphas = zoom2(alphalo, alphahi, w2, w1, s, sigma, c1, c2)
while (1)    
    alphaj = (alphalo + alphahi)/2;
    if ( getValue( s, sigma, (1-alphaj^2)^(1/2)*w2 + alphaj*w1) > getValue(s, sigma, w2) + c1*alphaj...
           *getgraalpha(s, sigma, w1, w2, 0) ||  getValue(s, sigma, (1-alphaj^2)^(1/2)*w2 + alphaj*w1)...
           > getValue(s, sigma, (1-alphalo^2)^(1/2)*w2 + alphalo*w1) )
        alphahi = alphaj;
        if (abs(alphahi-alphalo) < 1e-10)
           alphas = alphalo;
           return;
        end
    else
        if (abs(getgraalpha(s, sigma, w1, w2, alphaj)) <= -1*c2* getgraalpha(s, sigma, w1, w2, 0))
        alphas = alphaj;
        return;
        end
        if (getgraalpha(s, sigma, w1, w2, alphaj)*(alphahi - alphalo) >= 0)
            alphahi = alphalo;
        end
        alphalo = alphaj;
    end
end
