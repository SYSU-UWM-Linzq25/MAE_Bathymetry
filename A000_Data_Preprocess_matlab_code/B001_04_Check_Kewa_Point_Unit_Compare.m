%% ============================================================
%  B000_Check_Kewa_Point_Unit_Compare.m
%
%  Purpose:
%    Check whether KewaFix2Null bathy elevation is likely in feet.
%
%  Logic:
%    Sample common valid pixels from:
%      Bathy_1m_FixND/KewaFix2Null/Bathy_1m.vrt
%      3DEP_1m_ResampleClip/KewaFix2Null/DEM_3DEP_1m_ResampleandClip.vrt
%
%    Compare:
%      diff_raw_m_assumed = bathy_raw - dem
%      diff_ft_to_m       = bathy_raw * 0.3048 - dem
%
%    If abs(diff_ft_to_m) is much smaller than abs(diff_raw_m_assumed),
%    then bathy_raw is likely in feet.
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

river = 'KewaFix2Null';

bathy_vrt = fullfile(rootPR, 'Bathy_1m_FixND', river, 'Bathy_1m.vrt');
dem_vrt   = fullfile(rootPR, '3DEP_1m_ResampleClip', river, 'DEM_3DEP_1m_ResampleandClip.vrt');

outDir = fullfile(rootPR, 'Z003_Check_Bathy_3DEP_Unit', river);
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

outCSV = fullfile(outDir, 'Kewa_point_unit_compare.csv');

globalND = -999999;
feetToMeter = 0.3048;

if exist(bathy_vrt, 'file') ~= 2
    error('Missing bathy: %s', bathy_vrt);
end

if exist(dem_vrt, 'file') ~= 2
    error('Missing resampled 3DEP: %s', dem_vrt);
end

[~, rowsB, colsB, geoTransB, projB, ~, ~] = RasterInfo(bathy_vrt);
[~, rowsD, colsD, geoTransD, projD, ~, ~] = RasterInfo(dem_vrt);

fprintf('Bathy size: rows=%d cols=%d\n', rowsB, colsB);
fprintf('3DEP  size: rows=%d cols=%d\n', rowsD, colsD);
fprintf('Bathy pixel: %.6f %.6f\n', geoTransB(2), geoTransB(6));
fprintf('3DEP  pixel: %.6f %.6f\n', geoTransD(2), geoTransD(6));

if rowsB ~= rowsD || colsB ~= colsD
    error('Bathy and 3DEP are not on the same grid.');
end

if max(abs(geoTransB(:) - geoTransD(:))) > 1e-6
    warning('GeoTransform differs between bathy and 3DEP. Check alignment carefully.');
end

%% Read whole Kewa raster
% Kewa is small enough: about 2023 x 1733.
B = double(ReadRaster(bathy_vrt));
D = double(ReadRaster(dem_vrt));

validB = isfinite(B) & ~isnan(B) & (B ~= globalND) & (B > -1e20);
validD = isfinite(D) & ~isnan(D) & (D ~= globalND) & (D > -1e20);

valid = validB & validD;

nValid = nnz(valid);
fprintf('Common valid pixels: %d\n', nValid);

if nValid == 0
    error('No common valid pixels found.');
end

%% Candidate differences
B_if_meter = B;
B_if_feet_to_meter = B * feetToMeter;

diff_raw = B_if_meter - D;
diff_ftm = B_if_feet_to_meter - D;

abs_raw = abs(diff_raw(valid));
abs_ftm = abs(diff_ftm(valid));

fprintf('\nGlobal common-valid comparison:\n');

medAbsRaw = median(abs_raw, 'omitnan');
medAbsFtm = median(abs_ftm, 'omitnan');
meanAbsRaw = mean(abs_raw, 'omitnan');
meanAbsFtm = mean(abs_ftm, 'omitnan');

fprintf('  Median abs(B_raw - DEM)        = %.3f m\n', medAbsRaw);
fprintf('  Median abs(B_raw*0.3048 - DEM) = %.3f m\n', medAbsFtm);
fprintf('  Mean   abs(B_raw - DEM)        = %.3f m\n', meanAbsRaw);
fprintf('  Mean   abs(B_raw*0.3048 - DEM) = %.3f m\n', meanAbsFtm);

improveRatio = medAbsRaw / medAbsFtm;
fprintf('  Improvement ratio = %.3f\n', improveRatio);

if improveRatio > 2
    fprintf('\nDIAGNOSIS: Kewa bathy is very likely in feet.\n');
else
    fprintf('\nDIAGNOSIS: Not enough evidence from point comparison alone.\n');
end

%% Random sample points
rng(20260603);

idx = find(valid);
nSample = min(50, numel(idx));
sampleIdx = idx(randperm(numel(idx), nSample));

[rowSample, colSample] = ind2sub(size(B), sampleIdx);

River = repmat({river}, nSample, 1);

% Do not use variable names Row / Col.
% MATLAB table may conflict with dimension name "Row".
PixRow = rowSample(:);
PixCol = colSample(:);

X = nan(nSample, 1);
Y = nan(nSample, 1);

for k = 1:nSample
    [y, x] = RowCol2Proj(geoTransB, PixRow(k), PixCol(k));
    X(k) = x;
    Y(k) = y;
end

Bathy_raw = B(sampleIdx);
Bathy_raw_times_03048 = Bathy_raw * feetToMeter;
DEM_3DEP = D(sampleIdx);

Diff_raw_minus_DEM = Bathy_raw - DEM_3DEP;
Diff_ftm_minus_DEM = Bathy_raw_times_03048 - DEM_3DEP;

AbsDiff_raw = abs(Diff_raw_minus_DEM);
AbsDiff_ftm = abs(Diff_ftm_minus_DEM);

T = table( ...
    River, PixRow, PixCol, X, Y, ...
    Bathy_raw, Bathy_raw_times_03048, DEM_3DEP, ...
    Diff_raw_minus_DEM, Diff_ftm_minus_DEM, ...
    AbsDiff_raw, AbsDiff_ftm);

writetable(T, outCSV);

fprintf('\nSample point comparison written to:\n%s\n', outCSV);

%% Print first 15 sample points
disp(T(1:min(15, height(T)), :));