%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Homework 3 miniproject, ML, Spring 2016      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This is the script that I used to generate the fragmented 
% image files. - Herbert

%%% set the sidelength of tiles
% setting is l = 20 for the  difficult, 
% and l = 25 for the simple dataset.
l = 25;

%%% read image file 


rgb = imread('riddleImage.jpg');



%%% cut image to largest sidelengths which are multiples of l
xPixels = size(rgb,1); % number of pixels in first array dimension 
yPixels = size(rgb,2);
tilesX = floor(xPixels / l); % number of tiles in first array dimension
tilesY = floor(yPixels / l);
cutRGB = rgb(1:l*tilesX, 1:l*tilesY, :); 

%%% display 
figure(1); clf;
image(cutRGB); 
axis image; % this command sets the aspect ratio correct for image files

disp([num2str((tilesX-2)*(tilesY-2)) ' / ' num2str(tilesX*tilesY)])

%%% partition the cutRGB image into tiles of sidelength l and sort into an
%%% array smallRGBtiles
% note: zeros creates an array of type double, we need to convert that to
% the 8-bit integer format 
RGBtiles = uint8(zeros(l, l, 3, tilesX * tilesY)); 
tilecounter = 0;
tilesToShuffle = [];
edgeTiles = [];         % indexes of tiles that lie on the image edge
for xTile = 1:tilesX
  for yTile = 1:tilesY
    tilecounter = tilecounter + 1;
    RGBtiles(:,:,:, tilecounter) = ...
      cutRGB((xTile - 1)*l+1:xTile*l,(yTile - 1)*l+1:yTile*l, : );
    if xTile > 1 && xTile < tilesX && yTile > 1 && yTile < tilesY
        % tile is not on the edge and should be shuffled
        tilesToShuffle = [tilesToShuffle tilecounter];
    else
        edgeTiles = [edgeTiles tilecounter];
    end
  end
end

%%% shuffle the tiles. This is the array given out to course participants.
%RGBtilesShuffled = RGBtiles(:,:,:,randperm(tilesX * tilesY));
RGBtilesShuffled = RGBtiles;
RGBtilesShuffled(:,:,:,tilesToShuffle) = ...
    RGBtilesShuffled(:,:,:,tilesToShuffle(randperm(length(tilesToShuffle))));

%%% to visualize the mess, rearrange to original canvas size and display
%%% again
RGBrearranged = uint8(zeros(l * tilesX, l * tilesY, 3));
tilecounter = 0;
for xTile = 1:tilesX
  for yTile = 1:tilesY
    tilecounter = tilecounter + 1;
    RGBrearranged((xTile - 1)*l+1:xTile*l,(yTile - 1)*l+1:yTile*l, : ) = ...
      RGBtilesShuffled(:,:,:,tilecounter);
  end
end
figure(2); clf;
image(RGBrearranged);
axis image;

imwrite(RGBrearranged, 'img3.png');

disp(sprintf('shuffled image has distance %g from original', ...
  imageDistance(RGBrearranged, cutRGB)));

%save shuffledImageEasy RGBrearranged RGBtilesShuffled tilesX tilesY l edgeTiles;
