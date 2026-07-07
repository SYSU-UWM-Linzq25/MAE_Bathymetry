function D002_Verify_LossMask_NoBathyNoData_ForRiver(riverArg, varargin)
% D002_Verify_LossMask_NoBathyNoData_ForRiver
%
% Verify that final extracted MAE Loss_Mask_Pixel tiles do NOT contain loss=1
% where the original Bathy_<res>m_FixND raster is NoData / invalid.
%
% This is a QA-only script. It does not modify training tiles.
%
% Main check:
%   violation = (Loss_Mask_Pixel == 1) AND (Bathy_<res>m is invalid)
%
% Default roots are consistent with the MAE bathymetry project.
%
% Example:
%   D002_Verify_LossMask_NoBathyNoData_ForRiver('Kletzch_Combined_UpMax3Null');
%   D002_Verify_LossMask_NoBathyNoData_ForRiver('ALL', 'resolution', 1);
%   D002_Verify_LossMask_NoBathyNoData_ForRiver('ALL', 'maxTilesPerRiver', 50);
%
% Z. Lin project helper script.

% -----------------------------
% Defaults
% -----------------------------
cfg.processedRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';
cfg.tileRoot      = fullfile(cfg.processedRoot, 'Tiles_for_MAE_v2');
cfg.resolution    = 1;
cfg.noDataValue   = -999999;
cfg.invalidBelow  = -9999;
cfg.maxTilesPerRiver = inf;
cfg.progressEvery = 200;
cfg.cleanTemp     = true;
cfg.overwrite     = true;
cfg.verbose       = true;

% Fixed selected river list
allRivers = { ...
    'BadgerFinNull', ...
    'Estabrook_Combined', ...
    'KewaFix2Null', ...
    'Kletzch_Combined_UpMax3Null', ...
    'CA_KlamathRiver_TopoBathy_2018_D18', ...
    'CO_UpperColorado_Topobathy_1_2020', ...
    'MD_PotomacRiver_Bathy_2019', ...
    'NE_Niobrara_Topobathy_2018', ...
    'OR_MKRC_Topobathy_2021', ...
    'OR_SantiamRiverTB_Topobathy_1_D23', ...
    'WA_ChehalisRiverTB_Topobathy_1_D23', ...
    'WA_Nisqually_Bathymetric_2020'};

% -----------------------------
% Parse inputs
% -----------------------------
if nargin < 1 || isempty(riverArg)
    riverArg = 'LIST';
end

if mod(numel(varargin),2) ~= 0
    error('Optional arguments must be name/value pairs.');
end
for k = 1:2:numel(varargin)
    key = varargin{k};
    val = varargin{k+1};
    if ~isfield(cfg, key)
        error('Unknown option: %s', key);
    end
    cfg.(key) = val;
end

if strcmpi(riverArg, 'LIST')
    fprintf('Available rivers:\n');
    for i = 1:numel(allRivers)
        fprintf('  %2d. %s\n', i, allRivers{i});
    end
    return;
elseif strcmpi(riverArg, 'ALL')
    rivers = allRivers;
else
    rivers = {riverArg};
end

fprintf('\n============================================================\n');
fprintf('D002 Verify Loss_Mask_Pixel does not overlap Bathy NoData\n');
fprintf('Tile root      : %s\n', cfg.tileRoot);
fprintf('Processed root : %s\n', cfg.processedRoot);
fprintf('Resolution     : %dm\n', cfg.resolution);
if isfinite(cfg.maxTilesPerRiver)
    fprintf('Max tiles/river: %d\n', cfg.maxTilesPerRiver);
else
    fprintf('Max tiles/river: ALL\n');
end
fprintf('NoData         : %g, invalidBelow=%g\n', cfg.noDataValue, cfg.invalidBelow);

summaryRows = struct([]);

for ir = 1:numel(rivers)
    river = rivers{ir};
    try
        row = processOneRiver(river, cfg);
        summaryRows = appendStruct(summaryRows, row);
    catch ME
        fprintf('\n[ERROR] %s\n%s\n', river, ME.message);
        rethrow(ME);
    end
end

% Write global summary
qaRoot = fullfile(cfg.tileRoot, 'QA');
if ~exist(qaRoot, 'dir'); mkdir(qaRoot); end
summaryCsv = fullfile(qaRoot, sprintf('D002_verify_loss_no_bathy_nodata_summary_%dm.csv', cfg.resolution));
if ~isempty(summaryRows)
    Tsum = struct2table(summaryRows);
    writetable(Tsum, summaryCsv);
    fprintf('\nSummary CSV: %s\n', summaryCsv);
end

fprintf('\n============================================================\n');
fprintf('D002 finished.\n');
fprintf('============================================================\n');

end

% ========================================================================
function row = processOneRiver(river, cfg)
res = cfg.resolution;
resTag = sprintf('%dm', res);

fprintf('\n============================================================\n');
fprintf('[D002] River: %s\n', river);
fprintf('============================================================\n');

lossDir = fullfile(cfg.tileRoot, sprintf('Tiles_%s', resTag), 'Loss_Mask_Pixel');
bathyVrt = fullfile(cfg.processedRoot, sprintf('Bathy_%dm_FixND', res), river, sprintf('Bathy_%dm.vrt', res));
qaDir = fullfile(cfg.tileRoot, 'QA', river);
tmpDir = fullfile(qaDir, sprintf('_tmp_D002_bathy_crop_%s', resTag));

assert(exist(lossDir, 'dir') == 7, 'Loss_Mask_Pixel directory not found: %s', lossDir);
assert(exist(bathyVrt, 'file') == 2, 'Bathy VRT not found: %s', bathyVrt);
if ~exist(qaDir, 'dir'); mkdir(qaDir); end
if ~exist(tmpDir, 'dir'); mkdir(tmpDir); end

filesAll = dir(fullfile(lossDir, '*_LossMaskPixel.tif'));
if isempty(filesAll)
    filesAll = dir(fullfile(lossDir, '*LossMask*.tif'));
end
names = {filesAll.name};
keep = contains(names, river);
files = filesAll(keep);

if isempty(files)
    warning('No Loss_Mask_Pixel tiles found for river %s in %s', river, lossDir);
end

% Stable order by name
[~,ord] = sort({files.name});
files = files(ord);

nTotal = numel(files);
nCheck = min(nTotal, cfg.maxTilesPerRiver);
if isfinite(cfg.maxTilesPerRiver)
    files = files(1:nCheck);
end

fprintf('Loss mask tiles found : %d\n', nTotal);
fprintf('Loss mask tiles check : %d\n', numel(files));
fprintf('Bathy source          : %s\n', bathyVrt);

records = repmat(emptyRecord(), numel(files), 1);
tic;

for i = 1:numel(files)
    lossFile = fullfile(files(i).folder, files(i).name);
    rec = checkOneTile(lossFile, bathyVrt, tmpDir, cfg);
    records(i) = rec;

    if cfg.verbose && (mod(i, cfg.progressEvery) == 0 || i == numel(files))
        elapsed = toc;
        rate = i / max(elapsed, eps);
        eta = (numel(files)-i) / max(rate, eps) / 60;
        nViolTiles = sum([records(1:i).violation_count] > 0);
        nViolPix   = sum([records(1:i).violation_count]);
        fprintf('  checked %d/%d (%.1f%%), violation_tiles=%d, violation_pixels=%d, rate=%.2f tile/s, ETA=%.1f min\n', ...
            i, numel(files), 100*i/max(numel(files),1), nViolTiles, nViolPix, rate, eta);
    end
end

T = struct2table(records);
outCsv = fullfile(qaDir, sprintf('D002_verify_loss_no_bathy_nodata_%s_%s.csv', resTag, river));
writetable(T, outCsv);

nViolationTiles = sum(T.violation_count > 0);
nViolationPixels = sum(T.violation_count);
nLossBadValueTiles = sum(T.loss_bad_value_count > 0);
nLossNoDataTiles = sum(T.loss_nodata_count > 0);

fprintf('Checked tiles         : %d\n', height(T));
fprintf('Violation tiles       : %d\n', nViolationTiles);
fprintf('Violation pixels      : %d\n', nViolationPixels);
fprintf('Loss bad-value tiles  : %d\n', nLossBadValueTiles);
fprintf('Loss 255-value tiles  : %d\n', nLossNoDataTiles);
fprintf('QA CSV                : %s\n', outCsv);

if nViolationPixels == 0
    fprintf('[OK] No loss=1 pixels overlap Bathy NoData/invalid for %s.\n', river);
else
    fprintf('[WARNING] Found loss=1 pixels on Bathy NoData/invalid for %s. Inspect CSV.\n', river);
end

if cfg.cleanTemp
    try
        rmdir(tmpDir, 's');
    catch
    end
end

row = struct();
row.river = river;
row.resolution_m = res;
row.tiles_found = nTotal;
row.tiles_checked = height(T);
row.violation_tiles = nViolationTiles;
row.violation_pixels = nViolationPixels;
row.loss_bad_value_tiles = nLossBadValueTiles;
row.loss_255_value_tiles = nLossNoDataTiles;
row.csv = outCsv;

end

% ========================================================================
function rec = checkOneTile(lossFile, bathyVrt, tmpDir, cfg)
[~,base,~] = fileparts(lossFile);
rec = emptyRecord();
rec.tile = string(base);
rec.loss_file = string(lossFile);

% Read loss mask tile
loss = double(imread(lossFile));
[rows, cols] = size(loss);
rec.rows = rows;
rec.cols = cols;
rec.total_pixels = rows * cols;
rec.loss_pixel_count = sum(loss(:) == 1);
rec.loss_nodata_count = sum(loss(:) == 255);
rec.loss_bad_value_count = sum(~(loss(:) == 0 | loss(:) == 1 | loss(:) == 255));

% Get target extent from the loss tile itself
meta = gdalinfoJson(lossFile);
gt = meta.geoTransform;
if iscell(gt); gt = cellfun(@double, gt); end
x0 = gt(1); pxW = gt(2); y0 = gt(4); pxH = gt(6);
x1 = x0 + cols * pxW;
y1 = y0 + rows * pxH;
xmin = min(x0, x1); xmax = max(x0, x1);
ymin = min(y0, y1); ymax = max(y0, y1);

tmpBathy = fullfile(tmpDir, [base '_bathy_tmp.tif']);
if exist(tmpBathy, 'file') == 2 && cfg.overwrite
    delete(tmpBathy);
end

cmd = sprintf(['gdalwarp -overwrite -of GTiff -te %.15g %.15g %.15g %.15g -ts %d %d ' ...
               '-r near -srcnodata %.15g -dstnodata %.15g -ot Float32 ' ...
               '-co TILED=YES -co COMPRESS=LZW -co BIGTIFF=YES %s %s'], ...
               xmin, ymin, xmax, ymax, cols, rows, cfg.noDataValue, cfg.noDataValue, q(bathyVrt), q(tmpBathy));
[status, msg] = system(cmd);
if status ~= 0
    error('gdalwarp bathy crop failed for %s\nCommand: %s\nMessage: %s', lossFile, cmd, msg);
end

bathy = double(imread(tmpBathy));
if ~isequal(size(bathy), size(loss))
    error('Bathy crop size mismatch for %s. Loss=%dx%d, Bathy=%dx%d', base, rows, cols, size(bathy,1), size(bathy,2));
end

invalid = ~isfinite(bathy) | bathy == cfg.noDataValue | bathy <= cfg.invalidBelow;
violation = (loss == 1) & invalid;

rec.bathy_invalid_pixel_count = sum(invalid(:));
rec.violation_count = sum(violation(:));
if rec.loss_pixel_count > 0
    rec.violation_ratio_in_loss = rec.violation_count / rec.loss_pixel_count;
else
    rec.violation_ratio_in_loss = 0;
end

if rec.violation_count > 0
    vals = bathy(violation);
    rec.min_violation_bathy = min(vals(:));
    rec.max_violation_bathy = max(vals(:));
else
    rec.min_violation_bathy = NaN;
    rec.max_violation_bathy = NaN;
end

end

% ========================================================================
function meta = gdalinfoJson(fname)
cmd = sprintf('gdalinfo -json %s', q(fname));
[status, txt] = system(cmd);
if status ~= 0
    error('gdalinfo -json failed for %s\n%s', fname, txt);
end
meta = jsondecode(txt);
end

% ========================================================================
function s = q(p)
s = ['"' strrep(p, '"', '\"') '"'];
end

% ========================================================================
function rec = emptyRecord()
rec = struct();
rec.tile = string('');
rec.loss_file = string('');
rec.rows = NaN;
rec.cols = NaN;
rec.total_pixels = NaN;
rec.loss_pixel_count = NaN;
rec.loss_nodata_count = NaN;
rec.loss_bad_value_count = NaN;
rec.bathy_invalid_pixel_count = NaN;
rec.violation_count = NaN;
rec.violation_ratio_in_loss = NaN;
rec.min_violation_bathy = NaN;
rec.max_violation_bathy = NaN;
end

% ========================================================================
function A = appendStruct(A, row)
if isempty(A)
    A = row;
else
    A(end+1) = row; %#ok<AGROW>
end
end
