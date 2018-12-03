function Results = compute_H_graphmetrics(datafile, nmodules2find, corr_type, save_path, outfile_prefix)
%
%   INPUT
%       datafile = text file with parcellated time-series data
%       nmodules2find = optional argument to specify number of modules to find. Default is 5
%       corr_type = 'Spearman' or 'Pearson'. Defaults to 'Spearman'
%       save_path = path you want to save results into
%       outfile_prefix = prefix you want to put on saved output file
%

%% add path and check input arguments
addpath /Users/mvlombardo/Dropbox/matlab/BCT/2017_01_15_BCT;

% check if nmodules2find is specified
if ~exist('nmodules2find','var')
    nmodules2find = 5;
end % if

% check if corr_type is specified
if ~exist('corr_type','var')
    corr_type = 'Spearman';
end % if

%% Get graph theory metrics
Results = parcel2graphmetrics(datafile, nmodules2find);

%% read in parcellated data
data = readtable(datafile);
parcel_names = data.Properties.VariableNames;
data = table2array(data);

% find bad columns
mask = data==0 | isnan(data);
bad_reg = sort(find(mask(1,:)));
if ~isempty(bad_reg)
    bad_reg_mask = zeros(1,size(data,2));
    bad_reg_mask(bad_reg) = 1;
    bad_reg_mask = logical(bad_reg_mask);
    data2use = data;
    data2use(:,find(mask(1,:))) = [];
else
    data2use = data;
end

%% compute Hurst exponent
[H] = bfn_mfin_ml(data2use, 'filter','haar','lb',[-0.5,0],'ub',[1.5,10]);
H = reshape(H,1,length(H));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        H = insertAt(H,NaN,bad_reg(i));
    end
end

%% compute correlation between H and Graph Theory Metrics
GraphMeasures = table2array(Results.mainOutputTable(:,6:end));

Hcorr = corr(H',GraphMeasures,'rows','pairwise','type',corr_type);
graph_vars = Results.mainOutputTable.Properties.VariableNames(6:end);

%% Add H and correlations with graph metrics to Results
Results.HurstExponent = H;
Results.H_correlationsWith_GraphMetrics = cell2table(num2cell(Hcorr),'VariableNames',graph_vars);

tab2use = [Results.mainOutputTable, cell2table(num2cell(H'), 'VariableNames',{'H'})];
Results.mainOutputTable = tab2use;

%% Save output to file
fname2save = fullfile(save_path,sprintf('%s_H_GraphMetrics.csv',outfile_prefix));
writetable(Results.mainOutputTable, fname2save, ...
    'FileType', 'text', 'delimiter', ',');

fname2save = fullfile(save_path,sprintf('%s_H_correlationsWith_GraphMetrics.csv',outfile_prefix));
writetable(Results.H_correlationsWith_GraphMetrics, fname2save, ...
    'FileType', 'text', 'delimiter', ',');

fname2save = fullfile(save_path, sprintf('%s_H_GraphMetrics.mat', outfile_prefix));
save(fname2save,'Results');

end % function compute_H_graphmetrics


%% Main function to compute graph metrics
function Results = parcel2graphmetrics(datafile, nmodules2find)
%   
%   parcel2graphmetrics
%
%   Computes several different graph theory metrices from a file with
%   parcellated time-series
%
%   INPUT
%       datafile = text file with parcellated time-series data
%       nmodules2find = number to indicate how many modules to find. Default set to 5.
%
%   OUTPUT
%       Results = structure with all metrics saved within it
%
%
%   Dependencies
%   
%   Brain Connectivity Toolbox (version 2017-15-01): https://sites.google.com/site/bctnet/
%   Weighted Stochastic Block Model:  http://tuvalu.santafe.edu/~aaronc/wsbm/
%
%   Example usage:
% 
%   datafile = '/Users/mvlombardo/Desktop/parc_ts.csv';
%   nmodules2find = 5;
%   Results = parcel2graphmetrics(datafile, nmodules2find);
%

%% Check arguments

if ~exist('datafile','var')
    error('Need parcellated time-series data as input file!');
end

if ~exist('nmodules2find','var')
    nmodules2find = 5;
end % if

%% read in parcellated data
data = readtable(datafile);
parcel_names = data.Properties.VariableNames;
data = table2array(data);

% find bad columns
mask = data==0 | isnan(data);
bad_reg = sort(find(mask(1,:)));
if ~isempty(bad_reg)
    bad_reg_mask = zeros(1,size(data,2));
    bad_reg_mask(bad_reg) = 1;
    bad_reg_mask = logical(bad_reg_mask);
    data2use = data;
    data2use(:,find(mask(1,:))) = [];
else
    data2use = data;
end

%% compute Adjacency matrix (Pearson correlation)
adjmat2use = corr(data2use,'rows','pairwise');

%% binarize and normalize weights of connectivity matrix

% transform adjacency matrices
Wautofix = weight_conversion(adjmat2use,'autofix');
Wnorm = weight_conversion(Wautofix,'normalize');
Wlength = weight_conversion(adjmat2use,'lengths');

%% loop over range of connection costs and compute correlation between H and Degree
costs = [0.01:0.005:0.5];
Degree_overcosts = zeros(size(Wnorm,1),length(costs));
BC_overcosts = zeros(size(Degree_overcosts));
coreness_overcosts = zeros(size(Degree_overcosts));
for i = 1:length(costs)
    
    % threshold at specific cost
    cost_threshold = costs(i);
%     Wprop = threshold_proportional(adjmat2use,cost_threshold);
%     Wbin = weight_conversion(Wprop,'binarize');
    Wbin = mst_threshold(Wnorm,cost_threshold,0);
    
    % compute degree
    Degree_overcosts(:,i) = degrees_und(Wbin);

    % compute betweeness centrality
    BC_overcosts(:,i) = betweenness_bin(Wbin);

    % compute k-coreness centrality
    coreness_overcosts(:,i) = kcoreness_centrality_bu(Wbin);

end % for i = 1:length(costs)


%% Compute Degree at a cost of 5% (binary)
optimal_cost = 0.05;
% Wprop = threshold_proportional(adjmat2use,optimal_cost);
% Wbin = weight_conversion(Wprop,'binarize');
Wbin = mst_threshold(Wnorm,cost_threshold,0);

% compute Degree
Degree = degrees_und(Wbin);
Degree = reshape(Degree,1,length(Degree));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        Degree = insertAt(Degree,NaN,bad_reg(i));
    end
end

%% Betweeness Centrality at a cost of 5% (binary)
BC = betweenness_bin(Wbin);
BC = reshape(BC,1,length(BC));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        BC = insertAt(BC,NaN,bad_reg(i));
    end
end

%% Rich-Club K-coreness centrality at a cost of 5% (binary)
[coreness, kn] = kcoreness_centrality_bu(Wbin);
coreness = reshape(coreness,1,length(coreness));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        coreness = insertAt(coreness,NaN,bad_reg(i));
    end
end

%% Connection Strength (weighted)
Strength = strengths_und_sign(Wnorm);
Strength = reshape(Strength,1,length(Strength));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        Strength = insertAt(Strength,NaN,bad_reg(i));
    end
end

%% Clustering coefficient (weighted)
CC = clustering_coef_wu_sign(Wautofix);
CC = reshape(CC,1,length(CC));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        CC = insertAt(CC,NaN,bad_reg(i));
    end
end

%% Modularity (Louvain and WSBM method), within-module degree Z-score, participation, gateway, diversity coefficient (weighted)
% Modularity Louvain
gamma = 0.5; 
nmodules = 0;
% change gamma parameter until number of modules equals nmodules2find
while nmodules~=nmodules2find
    [M, Q] = community_louvain(Wautofix,gamma,[],'negative_asym');
    nmodules = length(unique(M)); %disp(nmodules);
    if nmodules>nmodules2find
        break
    else
        gamma = gamma + 0.001;
    end % if
end % while
M = reshape(M,1,length(M));

% Modularity WSBM
wsbm_n_blocks = length(unique(M)); 
edge_list = Adj2Edg(Wautofix);
[M_wsbm, Model_wsbm] = wsbm(edge_list,wsbm_n_blocks);
M_wsbm = reshape(M_wsbm,1,length(M_wsbm));

% metrics Louvain
MZ = module_degree_zscore(Wnorm,M,0); 
MZ = reshape(MZ,1,length(MZ));
[PC] = participation_coef(Wnorm,M); 
PC = reshape(PC,1,length(PC));
[PCP, PCN] = participation_coef_sign(Wnorm,M); 
PCP = reshape(PCP,1,length(PCP)); PCN = reshape(PCN,1,length(PCN));
[Gpos,Gneg] = gateway_coef_sign(Wnorm,M,1);
Gpos = reshape(Gpos,1,length(Gpos)); Gneg = reshape(Gneg,1,length(Gneg));
[DVpos DVneg] = diversity_coef_sign(Wnorm,M);
DVpos = reshape(DVpos,1,length(DVpos)); DVneg = reshape(DVneg,1,length(DVneg));

% metrics WSBM
MZ_wsbm = module_degree_zscore(Wnorm,M_wsbm,0);
MZ_wsbm = reshape(MZ_wsbm,1,length(MZ_wsbm));
[PC_wsbm] = participation_coef(Wnorm,M); 
PC_wsbm = reshape(PC_wsbm,1,length(PC_wsbm));
[PCP_wsbm, PCN_wsbm] = participation_coef_sign(Wnorm,M_wsbm);
PCP_wsbm = reshape(PCP_wsbm,1,length(PCP_wsbm)); PCN_wsbm = reshape(PCN_wsbm,1,length(PCN_wsbm));
[Gpos_wsbm,Gneg_wsbm] = gateway_coef_sign(Wnorm,M_wsbm,1);
Gpos_wsbm = reshape(Gpos_wsbm,1,length(Gpos_wsbm)); Gneg_wsbm = reshape(Gneg_wsbm,1,length(Gneg_wsbm));
[DVpos_wsbm, DVneg_wsbm] = diversity_coef_sign(Wnorm,M_wsbm);
DVpos_wsbm = reshape(DVpos_wsbm,1,length(DVpos_wsbm)); DVneg_wsbm = reshape(DVneg_wsbm,1,length(DVneg_wsbm));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        M = insertAt(M,NaN,bad_reg(i));
        M_wsbm = insertAt(M_wsbm,NaN,bad_reg(i));
        MZ = insertAt(MZ,NaN,bad_reg(i));
        PC = insertAt(PC,NaN,bad_reg(i));
        PCP = insertAt(PCP,NaN,bad_reg(i));
        PCN = insertAt(PCN,NaN,bad_reg(i));
        Gpos = insertAt(Gpos,NaN,bad_reg(i));
        Gneg = insertAt(Gneg,NaN,bad_reg(i));
        DVpos = insertAt(DVpos,NaN,bad_reg(i));
        DVneg = insertAt(DVneg,NaN,bad_reg(i));
        MZ_wsbm = insertAt(MZ_wsbm,NaN,bad_reg(i));
        PC_wsbm = insertAt(PC_wsbm,NaN,bad_reg(i));
        PCP_wsbm = insertAt(PCP_wsbm,NaN,bad_reg(i));
        PCN_wsbm = insertAt(PCN_wsbm,NaN,bad_reg(i));
        Gpos_wsbm = insertAt(Gpos_wsbm,NaN,bad_reg(i));
        Gneg_wsbm = insertAt(Gneg_wsbm,NaN,bad_reg(i));
        DVpos_wsbm = insertAt(DVpos_wsbm,NaN,bad_reg(i));
        DVneg_wsbm = insertAt(DVneg_wsbm,NaN,bad_reg(i));
    end
end

%% Core-Periphery Partition (weighted)
Core1Periphery0_weighted = core_periphery_dir(Wautofix);
Core1Periphery0_weighted = reshape(Core1Periphery0_weighted,1,length(Core1Periphery0_weighted));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        Core1Periphery0_weighted = insertAt(Core1Periphery0_weighted,NaN,bad_reg(i));
    end
end

%% Core-Periphery Partition (binary)
Core1Periphery0_binary = core_periphery_dir(Wbin);
Core1Periphery0_binary = reshape(Core1Periphery0_binary,1,length(Core1Periphery0_binary));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        Core1Periphery0_binary= insertAt(Core1Periphery0_binary,NaN,bad_reg(i));
    end
end


%% Eigenvector Centrality (weighted)
EIGC = eigenvector_centrality_und(Wnorm);
EIGC = reshape(EIGC,1,length(EIGC));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        EIGC = insertAt(EIGC,NaN,bad_reg(i));
    end
end

%% Local Efficiency (weighted)
Eloc = efficiency_wei(Wautofix,2);
Eloc = reshape(Eloc,1,length(Eloc));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        Eloc = insertAt(Eloc,NaN,bad_reg(i));
    end
end

%% Delta Efficiency (weighted)
% Global Efficiency
Eglob = efficiency_wei(abs(Wnorm));

for i = 1:size(Wnorm,1)
    tmp_data = data;
    tmp_data(:,i) = [];
    tmp_r = abs(corr(tmp_data));
    Eglob_attack(i) = efficiency_wei(tmp_r);
    DeltaEfficiency(1,i) = (Eglob - Eglob_attack(i))/Eglob;
end
DeltaEfficiency = reshape(DeltaEfficiency,1,length(DeltaEfficiency));

if ~isempty(bad_reg)
    for i = 1:length(bad_reg)
        % correct values for bad regions
        DeltaEfficiency = insertAt(DeltaEfficiency,NaN,bad_reg(i));
    end
end

%% Pack into Results structure
dat4export = [M', M_wsbm', Core1Periphery0_weighted', Core1Periphery0_binary', ...
    Degree', Strength', CC', ...
    real(Eloc)', DeltaEfficiency', BC', ...
    coreness', MZ', MZ_wsbm', EIGC', ...
    PC', PCP',PCN',Gpos',Gneg',DVpos',DVneg', ...
    PC_wsbm',PCP_wsbm',PCN_wsbm',Gpos_wsbm',Gneg_wsbm',DVpos_wsbm',DVneg_wsbm'];

col_names = {'ModuleLabels_Louvain','ModuleLabels_WSBM', ...
    'CorePeripheryLabels_Weighted', 'CorePeripheryLabels_Binary',...
    'Degree','Strength','ClusteringCoefficient', ...
    'LocalEfficiency','DeltaEfficiency','BetweenessCentrality', ...
    'KCorenessCentrality','WithinModuleDegreeZscore_Louvain','WithinModuleDegreeZscore_WSBM','EigenvectorCentrality', ...
    'ParticipationCoeff_Louvain','ParticipationCoeffPos_Louvain','ParticipationCoeffNeg_Louvain', ...
    'GatewayCoeffPos_Louvain','GatewayCoeffNeg_Louvain', ...
    'DiversityCoeffPos_Louvain','DiversityCoeffNeg_Louvain', ...
    'ParticipationCoeff_WSBM','ParticipationCoeffPos_WSBM','ParticipationCoeffNeg_WSBM', ...
    'GatewayCoeffPos_WSBM','GatewayCoeffNeg_WSBM', ...
    'DiversityCoeffPos_WSBM','DiversityCoeffNeg_WSBM'};

tab2export = cell2table([parcel_names',num2cell(dat4export)],'VariableNames',[{'region'},col_names]);

Results.data = data;                                % full parcel dataset
Results.bad_reg = bad_reg;                          % regions flagged up as bad (e.g., 0)
Results.adjmat2use = adjmat2use;                    % adjacency matrix used for graph computations
Results.CostUsedForBinaryThreshold = optimal_cost;  % cost threshold used for binary thresholding
Results.BinaryAdjMat = Wbin;                        % thresholded binary connection matrix
Results.WeightedAdjMatNorm = Wnorm;                 % normalized weighted connection matrix
Results.WeightedAdjMatLength = Wlength;             % length weighted connection matrix
Results.GlobalEfficiency = Eglob;                   % global efficiency calculated from weighted connection matrix
Results.costs = costs;                              % cost range to loop over
Results.Degree_overcosts = Degree_overcosts;        % nodal Degree over range of costs
Results.BetweenessCentrality_overcosts = BC_overcosts; % nodal Betweenenss Centrality over range of costs
Results.KcoreCentrality_overcosts = coreness_overcosts; % nodal K-coreness Centrality over range of costs
Results.mainOutputTable = tab2export;               % main output table with nodal measures    

disp('Done');

end % function parcel2graphmetrics


%% function to insert NaN for bad regions
function arrOut = insertAt(arr,val,index)
assert( index<= numel(arr)+1);
assert( index>=1);
if index == numel(arr)+1
    arrOut = [arr val];
else
    arrOut = [arr(1:index-1) val arr(index:end)];
end
end % function insertAt


%% function for thresholding minimum spanning tree at a specific cost
function [A] = mst_threshold(Co,MyCost,bin)
% input
% Co = weighted matrix
% MyCost = cost (in range [0,1])
% bin = "binary" = optional flag: T if binary (default), F if weighted
%
% output
% A = thresholded matrix
%
% author: Petra Vertes
% modified by Frantisek Vasa
%
% 24.10.18 - added graphminspantree function from MATLAB to do MST rather 
% than using kruskal_mst - mvlombardo
%

if nargin < 3
    bin = true;
end

n=size(Co,1);       % N nodes
Co=(Co+Co')/2;      % force symmetrize matrix
Co(find(Co<0))=0;   % set negative correlations to 0
Co(1:n+1:n*n)=1;    % set diagonal to ones

% create MST (the minimum spanning tree of the network)
D=ones(n,n)./Co;
% MST=kruskal_mst(sparse(D));
[MST] = graphminspantree(sparse(D));

% order C according to decreasing wieghts in the correlation matrix
Co=triu(Co,1);
ind = find(triu(ones(n,n),1));
Clist = Co(ind);
Cnonz = length(Clist);
[ClistSort, IX] = sort(Clist,'descend');
[row col]=ind2sub([n,n],ind(IX));
dd = length(Clist);

% store Initial MST in the adjacency matrix A that defines the output network
A = double(logical(full(MST)));

% grow the network according to weights in Co matrix
t=1;
enum=n-1;
% add edges in correct order until all possible edges exist
while (enum < MyCost*n*(n-1)/2)
    % if edge wasn't initially included in MST
    if A(row(t),col(t)) == 0
        if bin
            %add edge
            A(row(t),col(t)) = 1; %Co(row(t),col(t)); % binary version
            A(col(t),row(t)) = 1; %Co(col(t),row(t)); % binary version
            enum=enum+1;
        else
            A(row(t),col(t)) = Co(row(t),col(t)); % weighted version
            A(col(t),row(t)) = Co(row(t),col(t)); % weighted version
            enum=enum+1;
        end
    end
    t=t+1;
end

end % function mst_threshold
