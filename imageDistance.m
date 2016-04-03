function imdist = imageDistance(imFile1, imFile2)
% compute a distance between two image files (of same dimension). imFile1
% and imFile2 are image files in uint8 format.
imdist = mean(mean(mean(((double(imFile1 - imFile2)).^2),3),2),1);