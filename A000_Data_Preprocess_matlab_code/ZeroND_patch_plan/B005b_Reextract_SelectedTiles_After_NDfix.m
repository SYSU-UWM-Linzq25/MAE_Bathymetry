function auditCsv = B005b_Reextract_SelectedTiles_After_NDfix(river, varargin)
%% Re-extract tiles from FIXED merged data using existing selected center points.
%
% This does NOT redo random sampling or change PointID. It reads the existing
% <river>_Select_CenterPoints_1m.shp and writes a staged NDfix tile set.

p = inputParser;
addRequired(p, 'river', @(x) ischar(x) || isstring(x));
addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'res', 1, @isnumeric);
addParameter(p, 'winH', 336, @isnumeric);
addParameter(p, 'winW', 336, @isnumeric);
addParameter(p, 'globalND', -999999, @isnumeric);
addParameter(p, 'zeroTol', 1e-8, @isnumeric);
addParameter(p, 'assertNoZero', true, @islogical);
addParameter(p, 'selectedShp', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'finalMixVrt', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'maskVrt', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'outRoot', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);
parse(p, river, varargin{:});

river = char(p.Results.river);
rootPR = char(p.Results.rootPR);
res = p.Results.res;
winH = p.Results.winH;
winW = p.Results.winW;
globalND = double(p.Results.globalND);
zeroTol = double(p.Results.zeroTol);

if p.Results.doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

selectedShp = char(p.Results.selectedShp);
if isempty(selectedShp)
    selectedShp = fullfile(rootPR, 'Tiles_for_MAE', ...
        sprintf('%s_Select_CenterPoints_%dm.shp', river, res));
end
finalMixVrt = char(p.Results.finalMixVrt);
if isempty(finalMixVrt)
    finalMixVrt = fullfile(rootPR, sprintf('Bathy3DEP_Merged_Tiff_%dm_NDfix', res), ...
        river, sprintf('Combined_Bathy_Priority_%dm.vrt', res));
end
maskVrt = char(p.Results.maskVrt);
if isempty(maskVrt)
    maskVrt = fullfile(rootPR, sprintf('PredictionMask_LCCBathyValid_%dm_NDfix', res), ...
        river, sprintf('MAE_PredictionMask_%dm.vrt', res));
end
outRoot = char(p.Results.outRoot);
if isempty(outRoot)
    outRoot = fullfile(rootPR, 'Tiles_for_MAE_NDfix');
end

if exist(selectedShp, 'file') ~= 2; error('Missing selected point shp: %s', selectedShp); end
if exist(finalMixVrt, 'file') ~= 2; error('Missing fixed merged VRT: %s', finalMixVrt); end
if exist(maskVrt, 'file') ~= 2; error('Missing fixed mask VRT: %s', maskVrt); end

GT = shaperead(selectedShp);
if isempty(GT); error('No selected points in %s', selectedShp); end

% Keep only requested resolution if BestRes is present.
if isfield(GT, 'BestRes')
    keep = arrayfun(@(s) double(s.BestRes) == res, GT);
    GT = GT(keep);
end

[~, rowsF, colsF, geoF, projF, ~, ~] = RasterInfo(finalMixVrt);
[~, rowsM, colsM, geoM, ~, ~, ~] = RasterInfo(maskVrt);
if rowsF ~= rowsM || colsF ~= colsM
    error('Grid size mismatch: fixed merge=%d/%d mask=%d/%d', rowsF, colsF, rowsM, colsM);
end
if max(abs(geoF(:) - geoM(:))) > 1e-8
    error('GeoTransform mismatch between fixed merge and fixed mask.');
end

baseOut = fullfile(outRoot, sprintf('Tiles_%dm', res));
trainDir = fullfile(baseOut, 'Train_tile');
maskDir = fullfile(baseOut, 'LCC_Mask');
outRiverDir = fullfile(baseOut, 'TileOutRiver');
for folder = {trainDir, maskDir, outRiverDir}
    if exist(folder{1}, 'dir') ~= 7; mkdir(folder{1}); end
end

auditCsv = fullfile(baseOut, sprintf('%s_reextract_audit.csv', river));

hr = floor(winH / 2);
hc = floor(winW / 2);
outDataType = 6; % Float32

N = numel(GT);
PointID_col = nan(N,1);
N_zero_col = nan(N,1);
N_nodata_col = nan(N,1);
N_valid_col = nan(N,1);
ValidMin_col = nan(N,1);
ValidMax_col = nan(N,1);
Status_col = strings(N,1);

for k = 1:N
    if ~isfield(GT, 'PointID')
        error('Selected shapefile has no PointID field: %s', selectedShp);
    end

    PointID = double(GT(k).PointID);
    X = double(GT(k).X(1));
    Y = double(GT(k).Y(1));
    [row0, col0] = Proj2RowCol(geoF, Y, X);

    r1 = row0 - hr;
    c1 = col0 - hc;
    if r1 < 1 || c1 < 1 || (r1 + winH - 1) > rowsF || (c1 + winW - 1) > colsF
        error('Tile out of bounds: river=%s PointID=%d r1=%d c1=%d', ...
            river, PointID, r1, c1);
    end

    tileData = double(ReadRaster(finalMixVrt, r1, c1, winH, winW));
    tileMaskRaw = double(ReadRaster(maskVrt, r1, c1, winH, winW));
    tileMask = uint8(tileMaskRaw == 1);

    invalid = ~isfinite(tileData) | isnan(tileData) | (tileData == globalND);
    tileData(invalid) = globalND;
    zeroMask = isfinite(tileData) & (tileData ~= globalND) & abs(tileData) <= zeroTol;

    if p.Results.assertNoZero && any(zeroMask(:))
        error(['Unexpected zero remains after upstream ND fix: river=%s PointID=%d ' ...
               'N_zero=%d. Do not write this tile.'], ...
               river, PointID, nnz(zeroMask));
    end

    valid = isfinite(tileData) & ~isnan(tileData) & (tileData ~= globalND);
    tileOutRiver = tileData;
    tileOutRiver(tileMask == 1) = globalND; % Never write NaN and hope it becomes ND.

    subgeoTrans = subTranscoef(geoF, r1, c1);
    trainFile = fullfile(trainDir, sprintf('Select_tile_Basin_%dm_%s_ID%d.tif', ...
        res, river, PointID));
    maskFile = fullfile(maskDir, sprintf('Select_tile_%dm_%s_ID%d_LCC_Mask.tif', ...
        res, river, PointID));
    outRiverFile = fullfile(outRiverDir, sprintf('Select_tileOutRiver_%dm_%s_ID%d.tif', ...
        res, river, PointID));

    if ~p.Results.overwrite && (exist(trainFile,'file') == 2 || exist(maskFile,'file') == 2)
        error('Output exists and overwrite=false for PointID=%d', PointID);
    end

    WriteRaster(trainFile, tileData, subgeoTrans, projF, outDataType, 'GTiff', globalND);
    WriteRaster(maskFile, double(tileMask), subgeoTrans, projF, 1, 'GTiff', 255);
    WriteRaster(outRiverFile, tileOutRiver, subgeoTrans, projF, outDataType, 'GTiff', globalND);

    PointID_col(k) = PointID;
    N_zero_col(k) = nnz(zeroMask);
    N_nodata_col(k) = nnz(~valid);
    N_valid_col(k) = nnz(valid);
    if any(valid(:))
        ValidMin_col(k) = min(tileData(valid));
        ValidMax_col(k) = max(tileData(valid));
    end
    Status_col(k) = "PASS";

    fprintf('\r  Re-extract progress: %6.2f%% (%d/%d)', 100*k/N, k, N);
end
fprintf('\n');

T = table(PointID_col, N_zero_col, N_nodata_col, N_valid_col, ...
    ValidMin_col, ValidMax_col, Status_col, ...
    'VariableNames', {'PointID','N_zero','N_nodata','N_valid', ...
    'Valid_min','Valid_max','Status'});
writetable(T, auditCsv);

fprintf('Re-extracted tiles : %d\n', N);
fprintf('Train folder       : %s\n', trainDir);
fprintf('Mask folder        : %s\n', maskDir);
fprintf('Audit CSV          : %s\n', auditCsv);
end
