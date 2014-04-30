%% Extract low-frequency coefficients of Fourier transformation

%%
clear all, close all, clc;

I = imread('an2i_left_angry_open.pgm');
img = fftshift(I);
F = fftshift(fft2(img));

% magnitude = mat2gray(100*log(1+abs(F)));    % Magnitude spectrum
% phase = mat2gray( (angle(F)) );             % Phase spectrum

[M N K] = size(F);

L = 10;

fsub(M,N,K)=0;
fsub(M/2-L:M/2+L,N/2-L:N/2+L,1:K) = F(M/2-L:M/2+L,N/2-L:N/2+L,1:K);


I2 = uint8(real(ifftshift(ifft2(ifftshift(fsub)))));

figure
subplot(121)
imshow(I)

subplot(122)
imshow(I2)