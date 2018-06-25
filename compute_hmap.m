function Hmap = compute_hmap(datafile, maskfile, outname, nchunks)
%
%   compute_hmap - Compute whole-brain Hurst exponent map
%   
%   INPUT
%
%       datafile = full filename to 4D time-series data
%       maskfile = whole brain binary mask of the same size as datafile
%       outname  = full filename of the resulting H map to save to disk
%       nchunks  = number to indicate how many chunks of voxels you want to
%                  break up the processing into. An advisable number for
%                  this is around 400. It is not advisable to use numbers
%                  like 50, as this will significantly lead to slower
%                  processing. High numbers like 500 or larger also lead to
%                  slow downs.
%
%   OUTPUT
%
%       Hmap = 3D map whereby each brain voxel contains H
%
%   Example usage:
%
%       datafile = '/Users/mvlombardo/data/Erest_pp.nii.gz';
%       maskfile = '/Users/mvlombardo/data/mask.nii.gz';
%       outname = '/Users/mvlombardo/data/Erest_H.nii.gz';
%       nchunks = 400;
%       Hmap = compute_hmap(datafile, maskfile, outname, nchunks);
%
%   Dependencies
%
%   Requires the nonfractal MATLAB toolbox to compute the Hurst exponent.
%   Requires FSL MATLAB functions (read_avw, save_avw) for reading and 
%   writing *nii.gz files.
%
%   written by mvlombardo - 19.06.2018
%   

%% parameters used by bfn_mfin_ml to compute Hurst exponent
Hfilter = 'haar';
lb_param = [-0.5,0];
ub_param = [1.5,10];


%% Read in data and mask and transform 4D data into 2D matrix
[data] = read_avw(datafile);
[mask, dims, scales] = read_avw(maskfile); mask = logical(mask);

% number of timepoints
ntimepoints = size(data,4);

% number of brain voxels in mask
nbrainvoxels = sum(mask(:));

% initialize a 2D data matrix in memory [ntimepoints, nbrainvoxels]
data_mat = zeros(ntimepoints,nbrainvoxels);

% reshape 4D data into 2D data_mat
for i = 1:ntimepoints
    
    % extract current volume
    tmp_vol = data(:,:,:,i);
    
    % grab all brain voxels
    data_mat(i,:) = tmp_vol(mask);

end % for i

%% Loop over chunks of voxels and compute Hurst exponent

% find subset indices to loop over
chunk_size = floor(nbrainvoxels/nchunks); % size of each chunk of voxels
start_num = 1:chunk_size:nbrainvoxels; % the starting index for each time through the loop
end_num = [start_num(2:end)-1 nbrainvoxels]; % the end index for each time through the loop
start_end_idx_mat = [start_num' end_num']; % make start and end index 2 columns in a matrix

% initialize empty H vector to save results into
H = zeros(1,nbrainvoxels);
for i = 1:size(start_end_idx_mat,1)
    
    disp(sprintf('Working on chunk %d',i));
    
    % get indices to use
    idx2use = start_end_idx_mat(i,1):start_end_idx_mat(i,2);
    
    % grab subset of data to use
    data2use = data_mat(:,idx2use);
    
    % compute Hurst
    [Htmp] = bfn_mfin_ml(data2use, ...
        'filter',Hfilter,'lb',lb_param,'ub',ub_param);
    
    % save H values into final H vector
    H(1,idx2use) = Htmp;

end % for i

%% find voxels with no data in the time-series and re-fill H with NaN
% mask of voxels with flat-line 0 values in the time-series
nots_voxel_mask = sum(data_mat,1)==0;
H(1,nots_voxel_mask) = NaN;


%% reshape H vector into an image and write out to disk
% initial 3D Hmap in memory
Hmap = zeros(size(mask));

% find indices of brain voxels
brain_idx = find(mask);

% find x, y, z subscripts for each brain voxel
[x,y,z] = ind2sub(size(mask),brain_idx);

% loop over brain voxels
for i = 1:length(brain_idx)

    % take each brain voxel and place it into the 3D coordinate for Hmap
    Hmap(x(i),y(i),z(i)) = H(i);

end % for i

%% save file to disk
if ~isempty(outname)

    disp(sprintf('Writing %s to disk',outname));
    vsize = scales; vtype = 'f';
    save_avw(Hmap, outname, vtype, vsize);

end % if

disp('Done');

end % function compute_hmap