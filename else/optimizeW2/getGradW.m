function gradw = getGradW(s, sigma, w)
gradw = zeros(length(w),1);
l = length(s);
for i = 1:l
       gradw = gradw - 1/sigma^2*s(i).c*exp(-1*w'*s(i).A*w/(2*sigma^2))*s(i).A*w;
end
