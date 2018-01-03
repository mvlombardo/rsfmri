function result = parcellate(atlasfile, datafile, fname2save, MEANCENTER)
%   parcellate - Extract mean time series from parcellation
%
%   INPUT
%       atlasfile = parcellation to use
%       datafile = data to use
%       fname2save = prefix to use for saving file
%       mean_center = set to 1 to remove mean, otherwise set to 0
%
%   OUTPUT
%       result = table [nvols x nparcels]        
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
parc_num = unique(atlas); parc_num(parc_num==0) = [];

%% read in data
[data, dims,scales,bpp,endian] = read_avw(datafile);

%% compute mean within parcel for each time point
for i = 1:length(parc_num)
    mask = ismember(atlas,parc_num(i));
    
    for ivol = 1:size(data,4)
        tmp_vol = data(:,:,:,ivol);
        result(ivol,i) = nanmean(tmp_vol(mask));
    end % for i

end % for ivol


%% format column names
for i = 1:length(parc_num)
    var_names{i} = sprintf('parcel_%03d',i);
end

%% mean center data
if MEANCENTER
    for i = 1:size(result,2)
        result(:,i) = result(:,i) - nanmean(result(:,i));
    end % for i
end % if MEANCENTER


%% write out result to file
result = cell2table(num2cell(result),'VariableNames',var_names);
writetable(result,fname2save,'FileType','text','delimiter',',');


end % function parcellate 