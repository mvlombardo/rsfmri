function result = msent(imgfile, maskfile, outname, m, r, nscales, ...
                        atlasfile, VERBOSE)
% msent - compute multiscale entropy on 4d time-series data
%
%   INPUT
%       imgfile = name of input 4D .nii.gz
%       maskfile = name of mask .nii.gz
%       outname = prefix name of output
%       m = m parameter
%       r = r parameter
%       nscales = number of scales
%       atlasfile = name of parc .nii.gz, if '' then it runs voxel-wise
%       VERBOSE = set to 1 if you want to see progress while running
%
%   OUTPUT
%       result = usually a 4d matrix for voxel-wise processing, otherwise
%                it is a 2d matrix for when you process with a parcellation
%

if isempty(atlasfile)
    result = mse_img(imgfile, maskfile, outname, atlasfile, m, r, nscales, ...
                 VERBOSE);
elseif ~isempty(atlasfile)
    result = mse_parc(imgfile, maskfile, outname, atlasfile, m, r, nscales, ...
                 VERBOSE);
end

end % msent


%% 
function result = load_data(imgfile, maskfile, atlasfile)

[result.imgdata, dims, result.vsize, bpp, endian] = read_avw(imgfile);
result.maskdata = logical(read_avw(maskfile));
result.nvoxels = sum(result.maskdata(:));

if ~isempty(atlasfile)
    result.atlasdata = read_avw(atlasfile);
    result.nregions = length(unique(result.atlasdata(:)))-1;
else
    result.atlasdata = [];
    result.nregions = result.nvoxels;
end

result.numtps = size(result.imgdata,4);
end % load_data

%%
function result = parcellate(atlasfile, datafile, fname2save, MEANCENTER, nreg)
%   parcellate - Extract mean time series from parcellation
%
%   Depends on read_avw.m function from FSL to read in *nii.gz files
%
%   INPUT
%       atlasfile = parcellation to use
%       datafile = data to use
%       fname2save = prefix to use for saving file
%       mean_center = set to 1 to remove mean, otherwise set to 0
%       nreg = number of regions
%
%   OUTPUT
%       result = table [nvols, nparcels]
%
%   Example usage:
%
%   atlasfile = 'MMP_in_MNI_symmetrical_1_resamp.nii.gz';
%   datafile = 'P01_1_T1c_medn_nlw.nii.gz';
%   fname2save = 'parc_hcpsymm_medn.csv';
%   MEANCENTER = 1;
%   result = parcellate(atlasfile,datafile,fname2save,MEANCENTER);
%


%% read in parcellation
[atlas, dims,scales,bpp,endian] = read_avw(atlasfile);
% find unique parcel numbers, and remove parcel 0
% parc_num = unique(atlas); parc_num(parc_num==0) = [];
parc_num = 1:nreg;

%% read in data
[data, dims,scales,bpp,endian] = read_avw(datafile);

%% compute mean within parcel for each time point
for i = 1:length(parc_num)
    % make binary ROI mask for specific parcel
    mask = ismember(atlas,parc_num(i));

    if sum(mask(:))==0
        result(:,i) = NaN;
    else
        % loop over timepoints
        for ivol = 1:size(data,4)
            % grab specific timepoint
            tmp_vol = data(:,:,:,ivol);
            % compute mean within parcel mask
            result(ivol,i) = nanmean(tmp_vol(mask));
        end % for i
    end % if
end % for ivol


%% format column names
for i = 1:length(parc_num)
    var_names{i} = sprintf('parcel_%03d',i);
end

%% mean center data
if MEANCENTER
    % loop over each parcel
    for i = 1:size(result,2)
        % subtract out the mean
        result(:,i) = result(:,i) - nanmean(result(:,i));
    end % for i
end % if MEANCENTER


%% write out result to file
if ~isempty(fname2save)
    % format table to write out
    result = cell2table(num2cell(result),'VariableNames',var_names);

    % write to a file
    writetable(result,fname2save,'FileType','text','delimiter',',');
end % if ~isempty(fname2save)

end % function parcellate(atlasfile, datafile, fname2save, MEANCENTER)


%%
function e = msentropy(input, m, r, factor)
%   msentropy - calculate multi-scale entropy
%
%   INPUT
%       input = time-series data as input
%       m = m parameter (e.g., 2)
%       r = r parameter (e.g., 0.15)
%       factor = scale factor (e.g., 5)
%
%   Example usage
%
%   m = 2; r = 0.15; factor = 5;
%   e = msentropy(parcel_timeseries, m, r, factor);
%

y = input;
y = y - mean(y);
y =y/std(y);

for i = 1:factor
   s = coarsegraining(y,i);
   sampe = sampenc(s,m+1,r);
   e(i) = sampe(m+1);   
end
e = e';

end % function msentropy

%% coarsegraining function
function output = coarsegraining(input, factor)

n = length(input);
blk = fix(n/factor);
for i = 1:blk
   s(i) = 0; 
   for j = 1:factor
      s(i) = s(i) + input(j + (i - 1)*factor);
   end    
   s(i) = s(i)/factor;
end
output = s';

end % function coarsegraining

%% sampenc function
function [e, A, B] = sampenc(y, M, r)
%function [e, A, B] = sampenc(y, M, r);
%
%   Input
%
%       y input data
%       M maximum template length
%       r matching tolerance
%
%   Output
%
%       e sample entropy estimates for m=0,1,...,M-1
%       A number of matches for m=1,...,M
%       B number of matches for m=0,...,M-1 excluding last point

n = length(y);
lastrun = zeros(1,n);
run = zeros(1,n);
A = zeros(M,1);
B = zeros(M,1);
p = zeros(M,1);
e = zeros(M,1);
for i = 1:(n-1)
   nj = n-i;
   y1 = y(i);
   for jj = 1:nj
      j = jj+i;      
      if abs(y(j)-y1)<r
         run(jj) = lastrun(jj)+1;
         M1 = min(M,run(jj));
         for m = 1:M1           
            A(m) = A(m)+1;
            if j<n
               B(m) = B(m)+1;
            end % if j<n
         end % for m = 1:M1
      else
         run(jj) = 0;
      end % if abs(y(j)-y1)<r
   end % for jj = 1:nj
   for j = 1:nj
      lastrun(j) = run(j);
   end % for j = 1:nj
end % for i = 1:(n-1)

N = n*(n-1)/2;
B = [N;B(1:(M-1))];
p = A./B;
e = -log(p);

end % function sampenc

%%
function result = mse_img(imgfile, maskfile, outname, atlasfile, m, r, nscales, VERBOSE)

data = load_data(imgfile, maskfile, atlasfile);

indices = find(data.maskdata);
[x, y, z] = ind2sub(size(data.maskdata), indices);

result = zeros(size(data.maskdata,1), size(data.maskdata,2), size(data.maskdata,3), nscales);

for iscale = 1:nscales
    for ivox = 1:data.nvoxels
        if VERBOSE
            disp(sprintf('Working on scale %d, voxel %d of %d',iscale, ivox, data.nvoxels));
        end % if VERBOSE
        
        ts = squeeze(data.imgdata(x(ivox), y(ivox), z(ivox), :));
        
        try
            tmp_result = msentropy(ts, m, r, iscale);
            result(x(ivox), y(ivox), z(ivox), iscale) = tmp_result(iscale);
        catch
            result(x(ivox), y(ivox), z(ivox), iscale) = NaN;
        end % try
    end % for ivox
end % for iscale = 1:nscales

% save to file
fname2save = sprintf('%s_mse.nii.gz',outname);
vtype = 'f';
vsize = data.vsize';
save_avw(result, fname2save, vtype, vsize);

end % mse_img


%% 
function result = mse_parc(imgfile, maskfile, outname, atlasfile, m, r, nscales, VERBOSE)

data = load_data(imgfile, maskfile, atlasfile);
MEANCENTER = 0;
parc_data = parcellate(atlasfile, imgfile, '', MEANCENTER, data.nregions);
parc_data = parc_data';

indices = find(data.maskdata);
[x, y, z] = ind2sub(size(data.maskdata), indices);

result_img = zeros(size(data.maskdata,1), size(data.maskdata,2), size(data.maskdata,3), nscales);
result = zeros(data.nregions,nscales);
tmp_res = zeros(size(data.maskdata));

scale_labels = cell(1,nscales);
for ireg = 1:data.nregions
    region_labels{ireg,1} = sprintf('region_%03d',ireg);
    
    for iscale = 1:nscales
        scale_labels{iscale} = sprintf('scale_%d',iscale);
        if VERBOSE
            disp(sprintf('Working on scale %d, region %d of %d',iscale, ireg, data.nregions));
        end % if VERBOSE
        
        ts = parc_data(ireg,:);
        
        mask_roi = ismember(data.maskdata,ireg);
        tmp_res = zeros(size(data.maskdata));

        try
            tmp_result = msentropy(ts, m, r, iscale);
            result(ireg, iscale) = tmp_result(iscale);
            
            tmp_res(mask_roi) = tmp_result(iscale); 
            result_img(:,:,:,iscale) = tmp_res;
        catch 
            result(ireg, iscale) = NaN;

            tmp_res(mask_roi) = NaN; 
            result_img(:,:,:,iscale) = tmp_res;
        end % try
        
    end % for iscale
end % for ireg


% save to file
fname2save = sprintf('%s_mse_parc.nii.gz',outname);
vtype = 'f';
vsize = data.vsize';
save_avw(result_img, fname2save, vtype, vsize);
 
tab2write = cell2table([region_labels num2cell(result)], ...
                        'VariableNames',[{'region'}, scale_labels]);
fname2save = sprintf('%s_mse_parc.csv',outname);
writetable(tab2write,fname2save,'FileType','text','delimiter',',');
result = tab2write;

end % mse_parc