% SLIC Simple Linear Iterative Clustering SuperPixels
%
% Implementation of Achanta, Shaji, Smith, Lucchi, Fua and Susstrunk's
% SLIC Superpixels
%
% Usage:   [l, Am, Sp, d] = slic(im, k, m, seRadius, colopt, mw)
%
% Arguments:  im - Image to be segmented.
%              k - Number of desired superpixels. Note that this is nominal
%                  the actual number of superpixels generated will generally
%                  be a bit larger, espiecially if parameter m is small.
%              m - Weighting factor between colour and spatial
%                  differences. Values from about 5 to 40 are useful.  Use a
%                  large value to enforce superpixels with more regular and
%                  smoother shapes. Try a value of 10 to start with.
%       seRadius - Regions morphologically smaller than this are merged with
%                  adjacent regions. Try a value of 1 or 1.5.  Use 0 to
%                  disable.
%         colopt - String 'mean' or 'median' indicating how the cluster
%                  colour centre should be computed. Defaults to 'mean'
%             mw - Optional median filtering window size.  Image compression
%                  can result in noticeable artifacts in the a*b* components
%                  of the image.  Median filtering can reduce this. mw can be
%                  a single value in which case the same median filtering is
%                  applied to each L* a* and b* components.  Alternatively it
%                  can be a 2-vector where mw(1) specifies the median
%                  filtering window to be applied to L* and mw(2) is the
%                  median filtering window to be applied to a* and b*.
%
% Returns:     l - Labeled image of superpixels. Labels range from 1 to k.
%             Am - Adjacency matrix of segments.  Am(i, j) indicates whether
%                  segments labeled i and j are connected/adjacent
%             Sp - Superpixel attribute structure array with fields:
%                   L  - Mean L* value
%                   a  - Mean a* value
%                   b  - Mean b* value
%                   r  - Mean row value
%                   c  - Mean column value
%                   stdL  - Standard deviation of L* 
%                   stda  - Standard deviation of a* 
%                   stdb  - Standard deviation of b* 
%                   N - Number of pixels
%                   edges - List of edge numbers that bound each
%                           superpixel. This field is allocated, but not set,
%                           by SLIC. Use SPEDGES for this.
%              d - Distance image giving the distance each pixel is from its
%                  associated superpixel centre.
%
% It is suggested that use of this function is followed by SPDBSCAN to perform a
% DBSCAN clustering of superpixels.  This results in a simple and fast
% segmentation of an image.
%
% Minor variations from the original algorithm as defined in Achanta et al's
% paper:
%
% - SuperPixel centres are initialised on a hexagonal grid rather than a square
%   one. This results in a segmentation that will be nominally 6-connected
%   which hopefully facilitates any subsequent post-processing that seeks to
%   merge superpixels.
% - Initial cluster positions are not shifted to point of lowest gradient
%   within a 3x3 neighbourhood because this will be rendered irrelevant the
%   first time cluster centres are updated.
%
% Reference: R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua and
% S. Susstrunk. "SLIC Superpixels Compared to State-of-the-Art Superpixel
% Methods"  PAMI. Vol 34 No 11.  November 2012. pp 2274-2281.
%
% See also: SPDBSCAN, MCLEANUPREGIONS, REGIONADJACENCY, DRAWREGIONBOUNDARIES, RGB2LAB

% Copyright (c) 2013 Peter Kovesi
% Centre for Exploration Targeting
% School of Earth and Environment
% The University of Western Australia
% peter.kovesi at uwa edu au
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% Feb  2013
% July 2013 Super pixel attributes returned as a structure array

% Note that most of the computation time is not in the clustering, but rather
% in the region cleanup process.


function [l, Am, Sp, d] = slic(im, k, m, seRadius, colopt, mw, nItr, eim, We)
    
    if ~exist('colopt','var') || isempty(colopt), colopt = 'mean'; end
    if ~exist('mw','var')     || isempty(mw),         mw = 0;      end
    if ~exist('nItr','var')   || isempty(nItr),     nItr = 10;     end
    
    % grayscale variable is false in default
    %if ~exist('grayscale','var') || isempty(grayscale), grayscale = false; end
    
    if exist('eim', 'var'), USEDIST = 0; else, USEDIST = 1; end    
    
    MEANCENTRE = 1;
    MEDIANCENTRE = 2;
    
    if strcmp(colopt, 'mean')
        centre = MEANCENTRE;
    elseif strcmp(colopt, 'median')
        centre = MEDIANCENTRE;        
    else
        error('Invalid colour centre computation option');
    end
    
    % Obtain rows, cols and depths from the input image stack
    [rows, cols, depths] = size(im);
    disp("input image stack: ")
    disp(size(im));
    % Convert image to L*a*b* colourspace.  This gives us a colourspace that is
    % nominally perceptually uniform. This allows us to use the euclidean
    % distance between colour coordinates to measure differences between
    % colours.  Note the image becomes double after conversion.  We may want to
    % go to signed shorts to save memory.
    % im = rgb2lab(im); 
    
    % Apply median filtering to colour components if mw has been supplied
    % and/or non-zero
    if mw
        if length(mw) == 1
            mw(2) = mw(1);  % Use same filtering for L and chrominance
        end
        for n = 1:3
            % J = medfilt2(I,[m n]) performs median filtering, where each output pixel contains the median value in the m-by-n neighborhood around the corresponding pixel in the input image.
            im(:,:,n) = medfilt2(im(:,:,n), [mw(1) mw(1)]);
        end
    end
    
    % Nominal spacing between grid elements assuming cube grid
    S = (rows * cols * depths / k) .^ (1/3);
    
    nodeRows = floor(cols / S);
    nodeCols = floor(rows / S);
    nodeDepths = floor(depths / S);
    
    % Allocate memory and initialise clusters, labels and distances.
    % All cluster centers are stored in a matrix named C.
    C = zeros(7,k);          % Cluster centre data 1:3 is mean Lab value,
                             % 4:6 is row, col and depth of centre, 7 is No
                             % of voxels
    l = -ones(rows, cols, depths);   % Voxel labels.
    
    % Voxel distance to cluster centers is initialized as infinity 
    d = inf(rows, cols, depths);     
    
    S = round(S);
    
    % Initialze cluster centers 
    for ri = 1:nodeRows
        x = ri + S/2;
        for ci = 1:nodeCols
            y = ci + S/2;
            for di = 1:nodeDepths
                z = di + S/2;
                C(4:6,k) = [x,y,z];
                k = k + 1;
            end
        end
    end
    
    
    for n = 1:nItr
       for kk = 1:k  % for each cluster kk

           % Get subimage around cluster
           rmin = max(C(5,kk)-S, 1);   rmax = min(C(5,kk)+S, rows); 
           cmin = max(C(4,kk)-S, 1);   cmax = min(C(4,kk)+S, cols);
           dmin = max(C(6,kk)-S, 1);   dmax = min(C(6,kk)+S, depths);
           subim = im(rmin:rmax, cmin:cmax, dmin:dmax); 
           disp(size(subim));
           assert(numel(subim) > 0)
           
           % Compute distances D between C(:,kk) and subimage
           if USEDIST
               D = dist(C(:, kk), subim, rmin, cmin, dmin, S, m);
           else
               D = dist2(C(:, kk), subim, rmin, cmin, S, m, eim, We);
           end

           % If any voxel distance from the cluster centre is less than its
           % previous value update its distance and label
           subd =  d(rmin:rmax, cmin:cmax, dmin:dmax);
           subl =  l(rmin:rmax, cmin:cmax, dmin:dmax);
           updateMask = D < subd;
           subd(updateMask) = D(updateMask);
           subl(updateMask) = kk;
           
           d(rmin:rmax, cmin:cmax, dmin:dmax) = subd;
           l(rmin:rmax, cmin:cmax, dmin:dmax) = subl;           
       end
       
       % Update cluster centres with mean values
       C(:) = 0;
       for r = 1:rows
           for c = 1:cols
               for z = 1:depths
                   tmp = [im(r,c,1); im(r,c,2); im(r,c,3); c; r; 1];
                   C(:, l(r,c)) = C(:, l(r,c)) + tmp;
               end
           end
       end
       
       % Divide by number of pixels in each superpixel to get mean values
       for kk = 1:k 
           C(1:5,kk) = round(C(1:5,kk)/C(6,kk)); 
       end
       
       % Note the residual error, E, is not calculated because we are using a
       % fixed number of iterations 
    end
    
    % Cleanup small orphaned regions and 'spurs' on each region using
    % morphological opening on each labeled region.  The cleaned up regions are
    % assigned to the nearest cluster. The regions are renumbered and the
    % adjacency matrix regenerated.  This is needed because the cleanup is
    % likely to change the number of labeled regions.
%     [l, Am] = mcleanupregions(l, seRadius);
    Am = l;

    % Recompute the final superpixel attributes and write information into
    % the Sp struct array.
    N = length(Am);
    Sp = struct('L', cell(1,N), 'a', cell(1,N), 'b', cell(1,N), ...
                'stdL', cell(1,N), 'stda', cell(1,N), 'stdb', cell(1,N), ...
                'r', cell(1,N), 'c', cell(1,N), 'N', cell(1,N));
    [X,Y] = meshgrid(1:cols, 1:rows);
    L = im(:,:,1);    
    A = im(:,:,2);    
    B = im(:,:,3);    
    for n = 1:N
        mask = l==n;
        nm = sum(mask(:));
        if centre == MEANCENTRE     
            Sp(n).L = sum(L(mask))/nm;
            Sp(n).a = sum(A(mask))/nm;
            Sp(n).b = sum(B(mask))/nm;
            
        elseif centre == MEDIANCENTRE
            Sp(n).L = median(L(mask));
            Sp(n).a = median(A(mask));
            Sp(n).b = median(B(mask));
        end
        
        Sp(n).r = sum(Y(mask))/nm;
        Sp(n).c = sum(X(mask))/nm;
        
        % Compute standard deviations of the colour components of each super
        % pixel. This can be used by code seeking to merge superpixels into
        % image segments.  Note these are calculated relative to the mean colour
        % component irrespective of the centre being calculated from the mean or
        % median colour component values.
        Sp(n).stdL = std(L(mask));
        Sp(n).stda = std(A(mask));
        Sp(n).stdb = std(B(mask));

        Sp(n).N = nm;  % Record number of pixels in superpixel too.
    end
    
%-- dist -------------------------------------------
%
% Usage:  D = dist(C, im, r1, c1, S, m)
% 
% Arguments:   C - Cluster being considered
%             im - sub-image surrounding cluster centre
%         r1, c1 - row and column of top left corner of sub image within the
%                  overall image.
%              S - grid spacing
%              m - weighting factor between colour and spatial differences.
%
% Returns:     D - Distance image giving distance of every pixel in the
%                  subimage from the cluster centre
%
% Distance = sqrt( dc^2 + (ds/S)^2*m^2 )
% where:
% dc = sqrt(dl^2 + da^2 + db^2)  % Colour distance
% ds = sqrt(dx^2 + dy^2)         % Spatial distance
%
% m is a weighting factor representing the nominal maximum colour distance
% expected so that one can rank colour similarity relative to distance
% similarity.  try m in the range [1-40] for L*a*b* space
%
% ?? Might be worth trying the Geometric Mean instead ??
%  Distance = sqrt(dc * ds)
% but having a factor 'm' to play with is probably handy

% This code could be more efficient

function D = dist(C, im, r1, c1, d1, S, m)

    % Squared spatial distance
    %    ds is a fixed 'image' we should be able to exploit this
    %    and use a fixed meshgrid for much of the time somehow...
    [rows, cols, depths] = size(im);
    [x,y,z] = meshgrid(c1:(c1+cols-1), r1:(r1+rows-1), d1:(d1+depths-1));
    x = x-C(4);  % x and y dist from cluster centre
    y = y-C(5);
    z = z-C(6);
    ds2 = x.^2 + y.^2 + z.^2;
    
    % Squared colour difference
    for n = 1:3
        im(:,:,n) = (im(:,:,n)-C(n)).^2;
    end
    dc2 = sum(im,3);
    
    D = sqrt(dc2 + ds2/S^2*m^2);
    
    
    
%--- dist2 ------------------------------------------
%
% Usage:  D = dist2(C, im, r1, c1, S, m, eim)
% 
% Arguments:   C - Cluster being considered
%             im - sub-image surrounding cluster centre
%         r1, c1 - row and column of top left corner of sub image within the
%                  overall image.
%              S - grid spacing
%              m - weighting factor between colour and spatial differences.
%            eim - Edge strength sub-image corresponding to im
%
% Returns:     D - Distance image giving distance of every pixel in the
%                  subimage from the cluster centre
%
% Distance = sqrt( dc^2 + (ds/S)^2*m^2 )
% where:
% dc = sqrt(dl^2 + da^2 + db^2)  % Colour distance
% ds = sqrt(dx^2 + dy^2)         % Spatial distance
%
% m is a weighting factor representing the nominal maximum colour distance
% expected so that one can rank colour similarity relative to distance
% similarity.  try m in the range [1-40] for L*a*b* space
%

function D = dist2(C, im, r1, c1, S, m, eim, We)

    % Squared spatial distance
    %    ds is a fixed 'image' we should be able to exploit this
    %    and use a fixed meshgrid for much of the time somehow...
    [rows, cols, chan] = size(im);
    [x,y] = meshgrid(c1:(c1+cols-1), r1:(r1+rows-1));
    x = x-C(4);
    y = y-C(5);
    ds2 = x.^2 + y.^2;
    
    % Squared colour difference
    for n = 1:3
        im(:,:,n) = (im(:,:,n)-C(n)).^2;
    end
    dc2 = sum(im,3);
    
    % Combine colour and spatial distance measure
    D = sqrt(dc2 + ds2/S^2*m^2);
    
    % for every pixel in the subimage call improfile to the cluster centre
    % and use the largest value as the 'edge distance'
    rCentre = C(5)-r1;   % Cluster centre coords relative to this sub-image
    cCentre = C(4)-c1;
    de = zeros(rows,cols);
    for r = 1:rows
        for c = 1:cols
            v = improfile(eim,[c cCentre], [r rCentre]);
            de(r,c) = max(v);
        end
    end

    % Combine edge distance with weight, We with total Distance.
    D = D + We * de;
    