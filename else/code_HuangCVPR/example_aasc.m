clc;
clear all;
threshold=1e-5;
repeat_num=50;


%read data
data=load('dataset/data.txt');
ground_truth=load('dataset/label.txt');
ground_truth=ground_truth+1;
n=hist(ground_truth,min(ground_truth):1:max(ground_truth));
cluster=size(find(n),2);
%normalize data
data=data_normalize(data,'var');
data=normr(data);
%compute kvalue
kvalue=compute_kvalue(data);
number=size(kvalue,1);
sizeK=size(kvalue,3);
equal_k=zeros(number,number);
for i=1:sizeK
    equal_k=equal_k+kvalue(:,:,i)/sizeK;
end

result_SC=zeros(sizeK,repeat_num);
result_EASC=zeros(1,repeat_num);
result_AASC=zeros(1,repeat_num);
for iter=1:repeat_num
    fprintf('iter %d\n',iter);    
    %single sc
    for i=1:size(kvalue,3)
        dx = SC(kvalue(:,:,i),cluster);
        result_SC(i,iter)=nmi(dx,ground_truth);
    end
    %equal weight
    dx = SC(equal_k,cluster);
    result_EASC(1,iter)=nmi(dx,ground_truth);
    %aasc
    [dx weight]= AASC(kvalue,cluster);
    result_AASC(1,iter)=nmi(dx,ground_truth);
end
for i=1:size(kvalue,3)
    fprintf('SC%d : %4f\n',i,mean(result_SC(i,:)));
end
fprintf('EASC : %4f\n',mean(result_EASC));
fprintf('AASC : %4f\n',mean(result_AASC));
fprintf('weight : %4f\n',weight);
clear all;