%% Mean Filter for an image
%%
clear all, close all, clc;

img = imread('an2i_left_angry_open.pgm');
img_2 = medfilt2(img);

figure
subplot(121);
imshow(img);

subplot(122);
imshow(img_2);