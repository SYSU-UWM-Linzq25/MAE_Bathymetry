function outVrt = B003s_10_Build_SimpleFinalMask_FromCanonicalBathy(river, varargin)
%% Build final_mask = (LCC == 1) & canonical_bathy_valid.
%
% Zero is not globally treated as NoData. The canonical bathy stage already
% resolves source metadata and optional river-specific zero-fill policy.

p = inputParser;
addRequired(p, 'river', @(x) ischar(x) || isstring(x));
addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'res', 1, @isnumeric);
addParameter(p, 'bathyRootPattern', 'Bathy_%dm_CanonicalND', @(x) ischar(x) || isstring(x));
addParameter(p, 'lccRootPattern', 'LCC_%dm', @(x) ischar(x) || isstring(x));
addParameter(p, 'outRootPattern', 'PredictionMask_LCCBathyValid_%dm_CanonicalND', @(x) ischar(x) || isstring(x));
addParameter(p, 'globalND', -999999, @isnumeric);
addParameter(p, 'maskND', 255, @isnumeric);
addParameter(p, 'zeroIsNoDataFallback', false, @islogical);
addParameter(p, 'zeroTol', 1e-8, @isnumeric);
addParameter(p, 'tileSize', 2048, @isnumeric);
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);
parse(p, river, varargin{:});

river = char(p.Results.river);
rootPR = char(p.Results.rootPR);
res = p.Results.res;
globalND = double(p.Results.globalND);
maskND = double(p.Results.maskND);
zeroTol = double(p.Results.zeroTol);
tileSize = p.Results.tileSize;

if p.Results.doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

bathyVrt = fullfile(rootPR, sprintf(char(p.Results.bathyRootPattern), res), ...
    river, sprintf('Bathy_%dm.vrt', res));
lccVrt = fullfile(rootPR, sprintf(char(p.Results.lccRootPattern), res), ...
    river, sprintf('ESA_WorldCover_Resampleandclip_%dm.vrt', res));
outSub = fullfile(rootPR, sprintf(char(p.Results.outRootPattern), res), river);
tilesDir = fullfile(outSub, '_tiles');
outVrt = fullfile(outSub, sprintf('MAE_PredictionMask_%dm.vrt', res));
listTxt = fullfile(outSub, 'tile_list.txt');
summaryCsv = fullfile(outSub, 'mask_summary.csv');

if exist(bathyVrt, 'file') ~= 2; error('Missing canonical bathy: %s', bathyVrt); end
if exist(lccVrt, 'file') ~= 2; error('Missing LCC: %s', lccVrt); end
if exist(outSub, 'dir') ~= 7; mkdir(outSub); end
if exist(tilesDir, 'dir') ~= 7; mkdir(tilesDir); end

if p.Results.overwrite
    cmdClean = sprintf( ...
        'find "%s" -maxdepth 1 -type f -name ''tile_*.tif'' -delete', ...
        tilesDir);

    [statusClean, msgClean] = system(cmdClean);

    if statusClean ~= 0
        error('Failed to clean final-mask tiles: %s', msgClean);
    end
    if exist(outVrt, 'file') == 2; delete(outVrt); end
    if exist(listTxt, 'file') == 2; delete(listTxt); end
elseif exist(outVrt, 'file') == 2
    fprintf('[SKIP] Output exists: %s\n', outVrt);
    return;
end

[~, rowsB, colsB, geoB, projB, ~, ~] = RasterInfo(bathyVrt);
[~, rowsL, colsL, geoL, ~, ~, ~] = RasterInfo(lccVrt);
if rowsB ~= rowsL || colsB ~= colsL
    error('Grid size mismatch: canonical bathy=%d/%d LCC=%d/%d', ...
        rowsB, colsB, rowsL, colsL);
end
if max(abs(geoB(:) - geoL(:))) > 1e-8
    error('GeoTransform mismatch between canonical bathy and LCC.');
end

N_total = rowsB * colsB;
N_lcc = 0;
N_bathy_valid = 0;
N_final = 0;
N_zero_fallback_removed = 0;

totalTiles = ceil(rowsB / tileSize) * ceil(colsB / tileSize);
tileCount = 0;

for rLocal = 1:tileSize:rowsB
    rr = min(tileSize, rowsB - rLocal + 1);
    for cLocal = 1:tileSize:colsB
        cc = min(tileSize, colsB - cLocal + 1);

        B = double(ReadRaster(bathyVrt, rLocal, cLocal, rr, cc));
        L = double(ReadRaster(lccVrt, rLocal, cLocal, rr, cc));

        bathyValid = isfinite(B) & ~isnan(B) & (B ~= globalND);
        zeroFallback = bathyValid & abs(B) <= zeroTol;
        if p.Results.zeroIsNoDataFallback
            bathyValid(zeroFallback) = false;
        else
            zeroFallback(:) = false;
        end

        lccCandidate = isfinite(L) & ~isnan(L) & (L == 1);
        finalMask = lccCandidate & bathyValid;

        N_lcc = N_lcc + nnz(lccCandidate);
        N_bathy_valid = N_bathy_valid + nnz(bathyValid);
        N_final = N_final + nnz(finalMask);
        N_zero_fallback_removed = N_zero_fallback_removed + nnz(zeroFallback);

        subgeoTrans = subTranscoef(geoB, rLocal, cLocal);
        tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));
        WriteRaster(tileTif, double(uint8(finalMask)), subgeoTrans, projB, ...
            1, 'GTiff', maskND);

        tileCount = tileCount + 1;
        fprintf('\r  Mask progress: %6.2f%% (%d/%d)', ...
            100 * tileCount / totalTiles, tileCount, totalTiles);
    end
end
fprintf('\n');

cmdList = sprintf('find "%s" -maxdepth 1 -type f -name "tile_*.tif" | sort > "%s"', ...
    tilesDir, listTxt);
assert(system(cmdList) == 0, 'Failed to build mask tile list.');
cmdVrt = sprintf(['gdalbuildvrt -overwrite -srcnodata %.17g -vrtnodata %.17g ' ...
                  '-input_file_list "%s" "%s"'], ...
                  maskND, maskND, listTxt, outVrt);
assert(system(cmdVrt) == 0, 'gdalbuildvrt failed: %s', cmdVrt);

T = table(string(river), res, rowsB, colsB, N_total, N_lcc, ...
    N_bathy_valid, N_final, N_zero_fallback_removed, ...
    'VariableNames', {'River','Resolution_m','Rows','Cols','N_total', ...
    'N_LCC_candidate','N_bathy_valid','N_final_mask', ...
    'N_zero_fallback_removed'});
writetable(T, summaryCsv);

fprintf('LCC candidate pixels  : %d\n', N_lcc);
fprintf('Bathy valid pixels    : %d\n', N_bathy_valid);
fprintf('Final mask pixels     : %d\n', N_final);
fprintf('Output mask VRT       : %s\n', outVrt);
end
