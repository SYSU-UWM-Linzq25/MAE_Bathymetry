function B003b_Check_Bathy3DEP_Diff_ForRiver(river, varargin)
%% ============================================================
%  B003_Check_Bathy3DEP_Diff_ForRiver.m
%
%  Purpose:
%    Diagnose abs(bathy - 3DEP) distribution for one river.
%
%  It reports:
%    1) diff quantiles for all valid bathy/3DEP pairs
%    2) diff quantiles inside LCC
%    3) diff quantiles outside LCC
%    4) pixel counts above different thresholds
%    5) bathy NoData / 3DEP NoData relationship
%
%  Example:
%    B003_Check_Bathy3DEP_Diff_ForRiver('KewaFix2Null','res',1)
% ============================================================

p = inputParser;

addRequired(p, 'river', @(x) ischar(x) || isstring(x));

addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));

addParameter(p, 'res', 1, @isnumeric);
addParameter(p, 'bathyND', -999999, @isnumeric);
addParameter(p, 'demND', -999999, @isnumeric);
addParameter(p, 'thresholds_m', [0.02 0.05 0.10 0.25 0.50 1.00 2.00], @isnumeric);
addParameter(p, 'doPathSetup', true, @islogical);

parse(p, river, varargin{:});

river = char(p.Results.river);
rootPR = char(p.Results.rootPR);
res = p.Results.res;
bathyND = p.Results.bathyND;
demND = p.Results.demND;
thresholds_m = p.Results.thresholds_m;
doPathSetup = p.Results.doPathSetup;

if doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

outDir = fullfile(rootPR, 'Z008_Bathy3DEP_Diff_Diagnostics', river, sprintf('%dm', res));
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

bathy_vrt = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res), ...
    river, sprintf('Bathy_%dm.vrt', res));

lcc_vrt = fullfile(rootPR, sprintf('LCC_%dm', res), ...
    river, sprintf('ESA_WorldCover_Resampleandclip_%dm.vrt', res));

if res == 1
    dem_vrt = fullfile(rootPR, '3DEP_1m_ResampleClip', ...
        river, 'DEM_3DEP_1m_ResampleandClip.vrt');
else
    dem_vrt = fullfile(rootPR, sprintf('PredictionMask_%dm', res), ...
        river, '_tmp_3DEP_to_bathy_grid', ...
        sprintf('DEM_3DEP_%dm_to_bathygrid.vrt', res));
end

if exist(bathy_vrt, 'file') ~= 2
    error('Missing bathy: %s', bathy_vrt);
end

if exist(dem_vrt, 'file') ~= 2
    error('Missing 3DEP: %s', dem_vrt);
end

if exist(lcc_vrt, 'file') ~= 2
    error('Missing LCC: %s', lcc_vrt);
end

[~, rowsB, colsB, geoB, projB, dataTypeB, ~] = RasterInfo(bathy_vrt);
[~, rowsD, colsD, geoD, ~, ~, ~] = RasterInfo(dem_vrt);
[~, rowsL, colsL, geoL, ~, dataTypeL, ~] = RasterInfo(lcc_vrt);

fprintf('\n============================================================\n');
fprintf('Bathy-3DEP diff diagnostic\n');
fprintf('River: %s\n', river);
fprintf('Resolution: %dm\n', res);
fprintf('Bathy: %s\n', bathy_vrt);
fprintf('3DEP : %s\n', dem_vrt);
fprintf('LCC  : %s\n', lcc_vrt);
fprintf('============================================================\n');

fprintf('Bathy rows/cols = %d / %d\n', rowsB, colsB);
fprintf('3DEP  rows/cols = %d / %d\n', rowsD, colsD);
fprintf('LCC   rows/cols = %d / %d\n', rowsL, colsL);

if rowsB ~= rowsD || colsB ~= colsD || rowsB ~= rowsL || colsB ~= colsL
    error('Grid size mismatch.');
end

fprintf('maxGeoDiff Bathy-3DEP = %.12g\n', max(abs(geoB(:) - geoD(:))));
fprintf('maxGeoDiff Bathy-LCC  = %.12g\n', max(abs(geoB(:) - geoL(:))));

% Kewa is small enough to read full raster.
% For very large OR_MKRC 1m, do not use this full-read version.
B = double(ReadRaster(bathy_vrt));
D = double(ReadRaster(dem_vrt));
L = double(ReadRaster(lcc_vrt));

bathy_valid = isfinite(B) & ~isnan(B) & (B ~= bathyND) & (B > -1e20);
dem_valid   = isfinite(D) & ~isnan(D) & (D ~= demND) & (D > -1e20);
lcc_candidate = isfinite(L) & ~isnan(L) & (L == 1);

valid_pair = bathy_valid & dem_valid;

diffBD = abs(B - D);

diff_all = diffBD(valid_pair);
diff_lcc = diffBD(valid_pair & lcc_candidate);
diff_out = diffBD(valid_pair & ~lcc_candidate);

qs = [0 1 5 10 25 50 75 90 95 99 100];

Q_all = prctile(diff_all, qs);
Q_lcc = prctile(diff_lcc, qs);
Q_out = prctile(diff_out, qs);

Tq = table(qs(:), Q_all(:), Q_lcc(:), Q_out(:), ...
    'VariableNames', {'Percentile', 'Diff_All_m', 'Diff_InsideLCC_m', 'Diff_OutsideLCC_m'});

disp(' ');
disp('===== Diff quantiles =====');
disp(Tq);

outQuantileCSV = fullfile(outDir, sprintf('%s_diff_quantiles_%dm.csv', river, res));
writetable(Tq, outQuantileCSV);

% Threshold sensitivity table
rows = {};
header = { ...
    'Threshold_m', ...
    'N_all_valid_pair', ...
    'N_diff_gt_threshold_all', ...
    'Frac_diff_gt_threshold_all', ...
    'N_inside_LCC_valid_pair', ...
    'N_diff_gt_threshold_inside_LCC', ...
    'Frac_diff_gt_threshold_inside_LCC', ...
    'N_outside_LCC_valid_pair', ...
    'N_diff_gt_threshold_outside_LCC', ...
    'Frac_diff_gt_threshold_outside_LCC'};

rows(1,:) = header;

N_all = nnz(valid_pair);
N_in  = nnz(valid_pair & lcc_candidate);
N_out = nnz(valid_pair & ~lcc_candidate);

for i = 1:numel(thresholds_m)
    th = thresholds_m(i);

    N_gt_all = nnz(valid_pair & diffBD > th);
    N_gt_in  = nnz(valid_pair & lcc_candidate & diffBD > th);
    N_gt_out = nnz(valid_pair & ~lcc_candidate & diffBD > th);

    rows(end+1,:) = { ...
        th, ...
        N_all, ...
        N_gt_all, ...
        N_gt_all / max(N_all, 1), ...
        N_in, ...
        N_gt_in, ...
        N_gt_in / max(N_in, 1), ...
        N_out, ...
        N_gt_out, ...
        N_gt_out / max(N_out, 1)};
end

outThresholdCSV = fullfile(outDir, sprintf('%s_diff_threshold_sensitivity_%dm.csv', river, res));
writecell(rows, outThresholdCSV);

disp(' ');
disp('===== Threshold sensitivity =====');
disp(cell2table(rows(2:end,:), 'VariableNames', rows(1,:)));

% Basic validity counts
N_total = numel(B);
N_bathy_valid = nnz(bathy_valid);
N_dem_valid = nnz(dem_valid);
N_lcc = nnz(lcc_candidate);
N_valid_pair = nnz(valid_pair);

N_bathy_valid_dem_invalid_lcc = nnz(bathy_valid & ~dem_valid & lcc_candidate);
N_bathy_nodata_dem_valid_lcc = nnz(~bathy_valid & dem_valid & lcc_candidate);
N_lcc_bathy_nodata = nnz(lcc_candidate & ~bathy_valid);

fprintf('\n===== Validity counts =====\n');
fprintf('N_total                         = %d\n', N_total);
fprintf('N_bathy_valid                   = %d (%.4f)\n', N_bathy_valid, N_bathy_valid/N_total);
fprintf('N_dem_valid                     = %d (%.4f)\n', N_dem_valid, N_dem_valid/N_total);
fprintf('N_LCC_candidate                 = %d (%.4f)\n', N_lcc, N_lcc/N_total);
fprintf('N_bathy_dem_valid_pair          = %d (%.4f)\n', N_valid_pair, N_valid_pair/N_total);
fprintf('N_bathy_valid_dem_invalid_LCC   = %d\n', N_bathy_valid_dem_invalid_lcc);
fprintf('N_bathy_NoData_dem_valid_LCC    = %d\n', N_bathy_nodata_dem_valid_lcc);
fprintf('N_LCC_bathy_NoData_removed      = %d\n', N_lcc_bathy_nodata);

% Write key visual layers for this small river/res.
% For OR_MKRC 1m this would be huge; use only for Kewa/small rivers.
diffTif = fullfile(outDir, sprintf('%s_Diff_m_%dm.tif', river, res));
diffVis = diffBD;
diffVis(~valid_pair) = bathyND;

WriteRaster(diffTif, diffVis, geoB, projB, dataTypeB, 'GTiff', bathyND);

for i = 1:numel(thresholds_m)
    th = thresholds_m(i);
    thStr = strrep(sprintf('%.2f', th), '.', 'p');

    maskTif = fullfile(outDir, sprintf('%s_Diff_gt_%sm_%dm.tif', river, thStr, res));
    mask_u8 = uint8(valid_pair & diffBD > th);

    WriteRaster(maskTif, double(mask_u8), geoB, projB, dataTypeL, 'GTiff', 255);
end

fprintf('\nQuantile CSV:\n%s\n', outQuantileCSV);
fprintf('Threshold CSV:\n%s\n', outThresholdCSV);
fprintf('Diff raster:\n%s\n', diffTif);
fprintf('Output folder:\n%s\n', outDir);
fprintf('============================================================\n');

end