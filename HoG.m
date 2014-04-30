clear all, close all, clc;

% This script extract HoG features from images
basepath = '/Users/changyale/dataset/faces_4/an2i/an2i_left_angry_open_4.pgm';
im = imread(basepath);

cellSize = 2;
hog = vl_hog(im2single(im),cellSize,'verbose');

imhog = vl_hog('render',hog,'verbose');
clf; imagesc(imhog); colormap gray;
%imshow(im);
