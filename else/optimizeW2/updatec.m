function sout = updatec(s, sigma, w)
l = length(s);
sout = s;
for i = 1:l
    sout(i).c = sout(i).c* exp(-1*trace(w'*sout(i).A*w)/(2*sigma^2));
end