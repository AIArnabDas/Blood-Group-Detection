clc;
clear all;
close all;
disp('Process start....');

% Image Reading (Fixed File Path)
FilePath = 'E:\Blood Group Detection using Fingerprint\Blood Group Detection\niREP.png';
disp('The Image File Location is:');
disp(FilePath);
[DataImg, map] = imread(FilePath);
figure, imshow(DataImg, map);
title('Original Image'); 

% Calculating Height and Width of the Image
[ImH, ImW, Cdata] = size(DataImg);
if (Cdata == 3)
    DataImg = rgb2gray(DataImg);  % Fixed rgb2gray function
end
disp('Height:');
disp(ImH);
disp('Width:');
disp(ImW);

% Image Preprocessing Stage
I = DataImg;

% Canny Edge Detection
[~, threshold] = edge(I, 'canny');
fudgeFactor = 0.5;
Cedge = edge(I, 'Canny', threshold * fudgeFactor);
figure, imshow(Cedge);
title('Edge Detection (Canny)');

% Dilation
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
Cedgedil = imdilate(Cedge, [se90 se0]);
figure, imshow(Cedgedil);
title('Dilated Edge Mask');

% Filling Holes
Cedgefill = imfill(Cedgedil, 'holes');
figure, imshow(Cedgefill);
title('Dilated Edge with Filled Holes');

% Extracting the Largest Connected Component
B = strel('square', 15);
A = Cedgefill;
CC = bwconncomp(A);
numPixels = cellfun(@numel, CC.PixelIdxList);
[biggest, idx] = max(numPixels);
I(CC.PixelIdxList{idx}) = 0;

% Thresholding
Dfinal = zeros(ImH, ImW);
for i = 1:ImH
    for j = 1:ImW
        if (I(i, j) == 0)
            Dfinal(i, j) = DataImg(i, j);
        else 
            Dfinal(i, j) = 0;
        end
    end
end

figure, imshow(uint8(Dfinal));
title('Largest Connected Component');

% Labeling Connected Components
[Ilabel, num] = bwlabel(Dfinal);
Iprops = regionprops(Ilabel, 'BoundingBox');
Ibox = Iprops.BoundingBox;
Icrop = imcrop(DataImg, Ibox);
Icropsize = imresize(Icrop, [ImH, ImW]);
figure, imshow(uint8(Icropsize));
title('Cropped and Resized Image');

% Noise Reduction
FilteredImg = medfilt2(Icropsize);
figure, imshow(uint8(FilteredImg));
title('After Median Filtering');

% Histogram Equalization
EnhancedImg = histeq(FilteredImg);
figure, imshow(uint8(EnhancedImg));
title('After Histogram Equalization');

% ========= Coherence Computation =========
M = 3;
R = M - 1;
Input_Im = double(EnhancedImg);
row_max = size(Input_Im, 1) - M + 1;
col_max = size(Input_Im, 2) - M + 1;
Data_Im = zeros(row_max + R, col_max + R);

for i = 1:row_max
    for j = 1:col_max
        % Contrast Saliency Measure
        A = Input_Im(i:i+M-1, j:j+M-1);
        [U, W, V] = svd(A);
        w1 = W(1,1);
        w2 = W(2,2);
        w3 = W(3,3);
        C = w1 - (w2 * w3);
        Data_Im(i, j) = C;
    end
end

figure, imshow(uint8(Data_Im));
title('Coherence Computation');

% ========= Local Coherence Pattern (LCP) =========
C = round(M / 2);
Coh_Im = double(Data_Im);
LCP_Im = zeros(row_max, col_max);

for i = 1:row_max
    for j = 1:col_max
        % LCP Calculation
        Al = Coh_Im(i:i+M-1, j:j+M-1);
        Cdata = Al(C,C);
        Cdata1 = Al(1,2);
        Cdata2 = Al(1,3);
        Cdata3 = Al(2,3);
        Cdata4 = Al(3,3);
        Cdata5 = Al(3,2);
        Cdata6 = Al(3,1);
        Cdata7 = Al(2,1);
        Cdata8 = Al(1,1);
        
        B1 = Cdata > Cdata1;
        B2 = Cdata > Cdata2;
        B3 = Cdata > Cdata3;
        B4 = Cdata > Cdata4;
        B5 = Cdata > Cdata5;
        B6 = Cdata > Cdata6;
        B7 = Cdata > Cdata7;
        B8 = Cdata > Cdata8;
        
        LCP_Im(i, j) = B1 + B2 * 2 + B3 * 4 + B4 * 8 + B5 * 16 + B6 * 32 + B7 * 64 + B8 * 128;
    end
end

figure, imshow(uint8(LCP_Im));
title('Local Coherence Pattern');

% Histogram Calculation
bins = 2^8;
histLCP_Im = hist(LCP_Im(:), 0:(bins-1));
figure, stem(histLCP_Im);
title('Feature Vector - LBP');
xlabel('No. of Features');
ylabel('Feature Vector');
