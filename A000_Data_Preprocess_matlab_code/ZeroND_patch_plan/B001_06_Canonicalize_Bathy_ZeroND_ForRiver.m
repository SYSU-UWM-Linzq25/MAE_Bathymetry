function outVrt = B001_06_Canonicalize_Bathy_ZeroND_ForRiver(river, varargin)
%% Canonicalize one river bathymetry so confirmed zero-fill pixels become -999999.
%
% This function is intentionally river-specific. It MUST NOT be used globally
% unless zero has been confirmed to be a fill value for that river.
%
% Example:
%   B001_06_Canonicalize_Bathy_ZeroND_ForRiver( ...
%       'OR_SantiamRiverTB_Topobathy_1_D23', ...
%       'zeroIsNoData', true)

p = inputParser;
addRequired(p, 'river', @(x) ischar(x) || isstring(x));
addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'res', 1, @isnumeric);
addParameter(p, 'srcRootPattern', 'Bathy_%dm_FixND', @(x) ischar(x) || isstring(x));
addParameter(p, 'outRootPattern', 'Bathy_%dm_FixND_ZeroFixed', @(x) ischar(x) || isstring(x));
addParameter(p, 'globalND', -999999, @isnumeric);
addParameter(p, 'zeroIsNoData', false, @islogical);
addParameter(p, 'zeroTol', 1e-8, @isnumeric);
addParameter(p, 'tileSize', 2048, @isnumeric);
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);
parse(p, river, varargin{:});

river = char(p.Results.river);
rootPR = char(p.Results.rootPR);
res = p.Results.res;
globalND = double(p.Results.globalND);
zeroIsNoData = p.Results.zeroIsNoData;
zeroTol = double(p.Results.zeroTol);
tileSize = p.Results.tileSize;
overwrite = p.Results.overwrite;

if ~zeroIsNoData
    error(['zeroIsNoData=false. This function is for explicitly confirmed ' ...
           'zero-fill rivers only. Refusing to continue.']);
end

if p.Results.doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

srcRoot = fullfile(rootPR, sprintf(char(p.Results.srcRootPattern), res));
outRoot = fullfile(rootPR, sprintf(char(p.Results.outRootPattern), res));
srcVrt = fullfile(srcRoot, river, sprintf('Bathy_%dm.vrt', res));
outSub = fullfile(outRoot, river);
tilesDir = fullfile(outSub, '_tiles');
outVrt = fullfile(outSub, sprintf('Bathy_%dm.vrt', res));
listTxt = fullfile(outSub, 'tile_list.txt');
summaryCsv = fullfile(outSub, 'zero_nodata_canonicalize_summary.csv');

if exist(srcVrt, 'file') ~= 2
    error('Missing source bathy VRT: %s', srcVrt);
end
if exist(outSub, 'dir') ~= 7; mkdir(outSub); end
if exist(tilesDir, 'dir') ~= 7; mkdir(tilesDir); end

if overwrite
    system(sprintf('rm -f "%s"/tile_*.tif', tilesDir));
    if exist(outVrt, 'file') == 2; delete(outVrt); end
    if exist(listTxt, 'file') == 2; delete(listTxt); end
elseif exist(outVrt, 'file') == 2
    fprintf('[SKIP] Output exists and overwrite=false: %s\n', outVrt);
    return;
end

[~, rows, cols, geoTrans, proj, ~, srcND] = RasterInfo(srcVrt);
outDataType = 6; % GDAL Float32, required because -999999 must be preserved.

fprintf('\n============================================================\n');
fprintf('Canonicalize bathy zero-fill to unified NoData\n');
fprintf('River       : %s\n', river);
fprintf('Source      : %s\n', srcVrt);
fprintf('Output      : %s\n', outVrt);
fprintf('Source ND   : %.17g\n', double(srcND));
fprintf('Unified ND  : %.17g\n', globalND);
fprintf('Zero tol    : %.17g\n', zeroTol);
fprintf('Rows/Cols   : %d / %d\n', rows, cols);
fprintf('============================================================\n');

N_total = 0;
N_zero_reclassified = 0;
N_other_invalid = 0;
N_valid = 0;
validMin = inf;
validMax = -inf;

totalTiles = ceil(rows / tileSize) * ceil(cols / tileSize);
tileCount = 0;

for rLocal = 1:tileSize:rows
    rr = min(tileSize, rows - rLocal + 1);
    for cLocal = 1:tileSize:cols
        cc = min(tileSize, cols - cLocal + 1);

        B = double(ReadRaster(srcVrt, rLocal, cLocal, rr, cc));

        isFinite = isfinite(B) & ~isnan(B);
        isZeroFill = isFinite & abs(B) <= zeroTol;
        isDeclaredND = isFinite & (B == globalND);
        if isfinite(srcND)
            isDeclaredND = isDeclaredND | (B == double(srcND));
        end
        isOtherInvalid = ~isFinite | isDeclaredND;
        isInvalid = isOtherInvalid | isZeroFill;

        Bfix = B;
        Bfix(isInvalid) = globalND;

        valid = isfinite(Bfix) & ~isnan(Bfix) & (Bfix ~= globalND);
        if any(valid(:))
            validMin = min(validMin, min(Bfix(valid)));
            validMax = max(validMax, max(Bfix(valid)));
        end

        N_total = N_total + numel(B);
        N_zero_reclassified = N_zero_reclassified + nnz(isZeroFill);
        N_other_invalid = N_other_invalid + nnz(isOtherInvalid & ~isZeroFill);
        N_valid = N_valid + nnz(valid);

        subgeoTrans = subTranscoef(geoTrans, rLocal, cLocal);
        tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));
        WriteRaster(tileTif, Bfix, subgeoTrans, proj, outDataType, 'GTiff', globalND);

        tileCount = tileCount + 1;
        fprintf('\r  Progress: %6.2f%% (%d/%d)', ...
            100 * tileCount / totalTiles, tileCount, totalTiles);
    end
end
fprintf('\n');

cmdList = sprintf('find "%s" -maxdepth 1 -type f -name "tile_*.tif" | sort > "%s"', ...
    tilesDir, listTxt);
assert(system(cmdList) == 0, 'Failed to build tile list.');

cmdVrt = sprintf(['gdalbuildvrt -overwrite -srcnodata %.17g -vrtnodata %.17g ' ...
                  '-input_file_list "%s" "%s"'], ...
                  globalND, globalND, listTxt, outVrt);
assert(system(cmdVrt) == 0, 'gdalbuildvrt failed: %s', cmdVrt);

T = table(string(river), res, rows, cols, N_total, N_zero_reclassified, ...
    N_other_invalid, N_valid, validMin, validMax, globalND, ...
    'VariableNames', {'River','Resolution_m','Rows','Cols','N_total', ...
    'N_zero_reclassified','N_other_invalid','N_valid','Valid_min', ...
    'Valid_max','Unified_NoData'});
writetable(T, summaryCsv);

fprintf('Zero pixels reclassified : %d\n', N_zero_reclassified);
fprintf('Other invalid pixels     : %d\n', N_other_invalid);
fprintf('Valid range              : %.8f to %.8f\n', validMin, validMax);
fprintf('Output VRT               : %s\n', outVrt);
fprintf('Summary                  : %s\n', summaryCsv);
end
