function outVrt = B001_12_Rebuild_Bathy3DEP_Merge_ForRiver(river, varargin)
%% Rebuild a NoData-safe bathy-priority + 3DEP merge for one river.
%
% valid bathy -> bathy
% invalid bathy + valid 3DEP -> 3DEP
% both invalid -> -999999
%
% The output array is initialized to -999999 rather than copied from bathy.
% This prevents holes/NaN from being silently written as zero.

p = inputParser;
addRequired(p, 'river', @(x) ischar(x) || isstring(x));
addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'res', 1, @isnumeric);
addParameter(p, 'bathyRootPattern', 'Bathy_%dm_CanonicalND', @(x) ischar(x) || isstring(x));
addParameter(p, 'demRootPattern', '3DEP_%dm_ResampleClip', @(x) ischar(x) || isstring(x));
addParameter(p, 'outRootPattern', 'Bathy3DEP_Merged_Tiff_%dm_CanonicalND', @(x) ischar(x) || isstring(x));
addParameter(p, 'globalND', -999999, @isnumeric);
addParameter(p, 'zeroTol', 1e-8, @isnumeric);
addParameter(p, 'forbidZeroOutput', false, @islogical);
addParameter(p, 'tileSize', 2048, @isnumeric);
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);
parse(p, river, varargin{:});

river = char(p.Results.river);
rootPR = char(p.Results.rootPR);
res = p.Results.res;
globalND = double(p.Results.globalND);
zeroTol = double(p.Results.zeroTol);
tileSize = p.Results.tileSize;

if p.Results.doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

bathyVrt = fullfile(rootPR, sprintf(char(p.Results.bathyRootPattern), res), ...
    river, sprintf('Bathy_%dm.vrt', res));
demVrt = fullfile(rootPR, sprintf(char(p.Results.demRootPattern), res), ...
    river, sprintf('DEM_3DEP_%dm_ResampleandClip.vrt', res));
if res == 1 && exist(demVrt, 'file') ~= 2
    demVrt = fullfile(rootPR, '3DEP_1m_ResampleClip', river, ...
        'DEM_3DEP_1m_ResampleandClip.vrt');
end

outSub = fullfile(rootPR, sprintf(char(p.Results.outRootPattern), res), river);
tilesDir = fullfile(outSub, '_tiles');
outVrt = fullfile(outSub, sprintf('Combined_Bathy_Priority_%dm.vrt', res));
listTxt = fullfile(outSub, 'tile_list.txt');
summaryCsv = fullfile(outSub, 'merge_summary.csv');

if exist(bathyVrt, 'file') ~= 2; error('Missing canonical bathy: %s', bathyVrt); end
if exist(demVrt, 'file') ~= 2; error('Missing resampled 3DEP: %s', demVrt); end
if exist(outSub, 'dir') ~= 7; mkdir(outSub); end
if exist(tilesDir, 'dir') ~= 7; mkdir(tilesDir); end

if p.Results.overwrite
    cmdClean = sprintf( ...
        'find "%s" -maxdepth 1 -type f -name ''tile_*.tif'' -delete', ...
        tilesDir);

    [statusClean, msgClean] = system(cmdClean);

    if statusClean ~= 0
        error('Failed to clean merged bathy/3DEP tiles: %s', msgClean);
    end
    if exist(outVrt, 'file') == 2; delete(outVrt); end
    if exist(listTxt, 'file') == 2; delete(listTxt); end
elseif exist(outVrt, 'file') == 2
    fprintf('[SKIP] Output exists: %s\n', outVrt);
    return;
end

[~, rowsB, colsB, geoB, projB, ~, ~] = RasterInfo(bathyVrt);
[~, rowsD, colsD, geoD, projD, ~, ~] = RasterInfo(demVrt);
if rowsB ~= rowsD || colsB ~= colsD
    error(['Grid size mismatch for %s: canonical bathy=%d/%d, 3DEP=%d/%d. ' ...
           'Rebuild 3DEP_1m_ResampleClip after the unit/grid correction.'], ...
           river, rowsB, colsB, rowsD, colsD);
end
if max(abs(geoB(:) - geoD(:))) > 1e-8
    error(['GeoTransform mismatch for %s. Rebuild the resampled 3DEP against ' ...
           'the corrected Bathy_1m_FixND grid.'], river);
end
if ~strcmp(projB, projD)
    warning('[%s] Projection text differs although size/GeoTransform match.', river);
end

outDataType = 6; % Float32
N_total = 0;
N_bathy_valid = 0;
N_dem_fill = 0;
N_both_invalid = 0;
N_zero_from_bathy = 0;
N_zero_from_dem = 0;
N_zero_output = 0;
validMin = inf;
validMax = -inf;

totalTiles = ceil(rowsB / tileSize) * ceil(colsB / tileSize);
tileCount = 0;

for rLocal = 1:tileSize:rowsB
    rr = min(tileSize, rowsB - rLocal + 1);
    for cLocal = 1:tileSize:colsB
        cc = min(tileSize, colsB - cLocal + 1);

        B = double(ReadRaster(bathyVrt, rLocal, cLocal, rr, cc));
        D = double(ReadRaster(demVrt, rLocal, cLocal, rr, cc));

        validB = isfinite(B) & ~isnan(B) & (B ~= globalND);
        validD = isfinite(D) & ~isnan(D) & (D ~= globalND);

        C = ones(rr, cc) * globalND;
        C(validB) = B(validB);
        fillMask = ~validB & validD;
        C(fillMask) = D(fillMask);
        C(~isfinite(C) | isnan(C)) = globalND;

        zeroFromB = validB & abs(B) <= zeroTol;
        zeroFromD = fillMask & abs(D) <= zeroTol;
        zeroOutput = isfinite(C) & (C ~= globalND) & abs(C) <= zeroTol;

        if p.Results.forbidZeroOutput && any(zeroOutput(:))
            error(['Unexpected valid zero in merge: river=%s row=%d col=%d N=%d. ' ...
                   'Inspect the source before continuing.'], ...
                   river, rLocal, cLocal, nnz(zeroOutput));
        end

        validC = isfinite(C) & ~isnan(C) & (C ~= globalND);
        if any(validC(:))
            validMin = min(validMin, min(C(validC)));
            validMax = max(validMax, max(C(validC)));
        end

        N_total = N_total + numel(C);
        N_bathy_valid = N_bathy_valid + nnz(validB);
        N_dem_fill = N_dem_fill + nnz(fillMask);
        N_both_invalid = N_both_invalid + nnz(~validB & ~validD);
        N_zero_from_bathy = N_zero_from_bathy + nnz(zeroFromB);
        N_zero_from_dem = N_zero_from_dem + nnz(zeroFromD);
        N_zero_output = N_zero_output + nnz(zeroOutput);

        subgeoTrans = subTranscoef(geoB, rLocal, cLocal);
        tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));
        WriteRaster(tileTif, C, subgeoTrans, projB, outDataType, 'GTiff', globalND);

        tileCount = tileCount + 1;
        fprintf('\r  Merge progress: %6.2f%% (%d/%d)', ...
            100 * tileCount / totalTiles, tileCount, totalTiles);
    end
end
fprintf('\n');

cmdList = sprintf('find "%s" -maxdepth 1 -type f -name "tile_*.tif" | sort > "%s"', ...
    tilesDir, listTxt);
assert(system(cmdList) == 0, 'Failed to build merged tile list.');
cmdVrt = sprintf(['gdalbuildvrt -overwrite -srcnodata %.17g -vrtnodata %.17g ' ...
                  '-input_file_list "%s" "%s"'], ...
                  globalND, globalND, listTxt, outVrt);
assert(system(cmdVrt) == 0, 'gdalbuildvrt failed: %s', cmdVrt);

T = table(string(river), res, rowsB, colsB, string(bathyVrt), string(demVrt), ...
    N_total, N_bathy_valid, N_dem_fill, N_both_invalid, ...
    N_zero_from_bathy, N_zero_from_dem, N_zero_output, ...
    validMin, validMax, globalND, ...
    'VariableNames', {'River','Resolution_m','Rows','Cols','Bathy_source', ...
    'DEM_source','N_total','N_bathy_valid','N_filled_from_3DEP', ...
    'N_both_invalid','N_zero_from_bathy','N_zero_from_3DEP', ...
    'N_zero_output','Valid_min','Valid_max','Unified_NoData'});
writetable(T, summaryCsv);

fprintf('Bathy-valid pixels    : %d\n', N_bathy_valid);
fprintf('Filled from 3DEP      : %d\n', N_dem_fill);
fprintf('Both invalid          : %d\n', N_both_invalid);
fprintf('Valid zero output     : %d\n', N_zero_output);
fprintf('Valid output range    : %.8f to %.8f\n', validMin, validMax);
fprintf('Output VRT            : %s\n', outVrt);
end
