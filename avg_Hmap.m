function result = avg_Hmap(Hmap_file, parcel_file, mask_file, out_file)
%
%   avg_Hmap
%
%   Take a voxel-wise H map and compute the mean H per each region in a
%   parcellation.
%
%   INPUT
%       Hmap_file = full filepath and filename to H map to use
%       parcel_file = full filepath and filename to parcel file to use
%       mask_file = full filepath and filename to mask file to use
%       out_file = full filepath and filename of file to write out
%
%   The expectation of this script is that Hmap_file, parcel_file, and
%   mask_file all have the same dimensions and voxel sizes. Also the
%   out_file should be a .csv file.
%

% read in images
Hmap = read_avw(Hmap_file);
parcel_map = read_avw(parcel_file);
mask = read_avw(maks_file); mask = logical(mask);

% get vector of parcel labels
parcels2use = unique(parcel_map);
parcels2use(parcels2use==0) = [];

% loop over each parcel and compute the mean from brain voxels that overlap
% with voxels within the parcel, ignoring NaNs
result = zeros(1,parcels2use);
for i = 1:length(parcels2use)
    mask2use = ismember(parcel_map,parcels2use(i)) & mask;
    result(:,i) = nanmean(Hmap(mask2use));
end % for i

% write file to disk
tab2write = array2table(result);
fname2save = out_file;
writetable(tab2write, fname2save, 'FileType','text','delimiter',',')

end % function avg_Hmap