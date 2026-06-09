function outVrt = B001_10_Canonicalize_Bathy_ForRiver(river, varargin)
%% Canonicalize one already-unit-corrected bathymetry product.
%
% Source:
%   Bathy_<res>m_FixND/<river>/Bathy_<res>m.vrt
%
% Output:
%   Bathy_<res>m_CanonicalND/<river>/Bathy_<res>m.vrt
%
% This function:
%   - preserves all valid elevation values and the existing grid;
%   - converts NaN, Inf, source-declared NoData, and -999999 to -999999;
%   - optionally converts zero to NoData ONLY for an explicitly confirmed
%     zero-fill source;
%   - always writes Float32 and sets both pixel and VRT NoData to -999999.
%
% It does NOT convert units. OR_MKRC and Kewa therefore retain the output of
% B001_02_fix_OR_MKRC_unit_only.m and
% B001_05_fix_KewaFix2Null_unit_only.m.

p = inputParser;
addRequired(p, 'river', @(x) ischar(x) || isstring(x));
addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));
addParameter(p, 'res', 1, @isnumeric);
addParameter(p, 'srcRootPattern', 'Bathy_%dm_FixND', @(x) ischar(x) || isstring(x));
addParameter(p, 'outRootPattern', 'Bathy_%dm_CanonicalND', @(x) ischar(x) || isstring(x));
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
zeroTol = double(p.Results.zeroTol);
tileSize = p.Results.tileSize;

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
summaryCsv = fullfile(outSub, 'canonicalize_summary.csv');

if exist(srcVrt, 'file') ~= 2
    error('Missing source bathy VRT: %s', srcVrt);
end
if exist(outSub, 'dir') ~= 7; mkdir(outSub); end
if exist(tilesDir, 'dir') ~= 7; mkdir(tilesDir); end

if p.Results.overwrite
    cmdClean = sprintf( ...
        'find "%s" -maxdepth 1 -type f -name ''tile_*.tif'' -delete', ...
        tilesDir);

    [statusClean, msgClean] = system(cmdClean);

    if statusClean ~= 0
        error('Failed to clean canonical bathy tiles: %s', msgClean);
    end
    
    if exist(outVrt, 'file') == 2; delete(outVrt); end
    if exist(listTxt, 'file') == 2; delete(listTxt); end
elseif exist(outVrt, 'file') == 2
    fprintf('[SKIP] Output exists: %s\n', outVrt);
    return;
end

[~, rows, cols, geoTrans, proj, ~, srcND] = RasterInfo(srcVrt);
outDataType = 6; % Float32
srcNDPrint = NaN;
if ~isempty(srcND) && isfinite(srcND)
    srcNDPrint = double(srcND);
end

fprintf('\n============================================================\n');
fprintf('Canonicalize bathymetry: %s\n', river);
fprintf('Unit conversion          : NONE (input is already corrected)\n');
fprintf('Source                   : %s\n', srcVrt);
fprintf('Output                   : %s\n', outVrt);
fprintf('Source metadata NoData   : %.17g\n', srcNDPrint);
fprintf('Unified NoData           : %.17g\n', globalND);
fprintf('zeroIsNoData             : %d\n', p.Results.zeroIsNoData);
fprintf('============================================================\n');

N_total = 0;
N_zero_input = 0;
N_zero_reclassified = 0;
N_source_declared_nd = 0;
N_nonfinite = 0;
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

        finite = isfinite(B) & ~isnan(B);
        zeroInput = finite & abs(B) <= zeroTol;

        declaredND = finite & (B == globalND);
        if ~isempty(srcND) && isfinite(srcND)
            declaredND = declaredND | (B == double(srcND));
        end

        nonfinite = ~finite;
        invalid = nonfinite | declaredND;
        zeroReclassified = false(size(B));
        if p.Results.zeroIsNoData
            zeroReclassified = zeroInput & ~invalid;
            invalid = invalid | zeroReclassified;
        end

        Bfix = B;
        Bfix(invalid) = globalND;

        valid = isfinite(Bfix) & ~isnan(Bfix) & (Bfix ~= globalND);
        if any(valid(:))
            validMin = min(validMin, min(Bfix(valid)));
            validMax = max(validMax, max(Bfix(valid)));
        end

        N_total = N_total + numel(B);
        N_zero_input = N_zero_input + nnz(zeroInput);
        N_zero_reclassified = N_zero_reclassified + nnz(zeroReclassified);
        N_source_declared_nd = N_source_declared_nd + nnz(declaredND);
        N_nonfinite = N_nonfinite + nnz(nonfinite);
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
assert(system(cmdList) == 0, 'Failed to build bathy tile list.');

cmdVrt = sprintf(['gdalbuildvrt -overwrite -srcnodata %.17g -vrtnodata %.17g ' ...
                  '-input_file_list "%s" "%s"'], ...
                  globalND, globalND, listTxt, outVrt);
assert(system(cmdVrt) == 0, 'gdalbuildvrt failed: %s', cmdVrt);

T = table(string(river), res, rows, cols, string(srcVrt), string(outVrt), ...
    string("values_preserved_input_already_unit_corrected"), ...
    p.Results.zeroIsNoData, N_total, N_zero_input, N_zero_reclassified, ...
    N_source_declared_nd, N_nonfinite, N_valid, validMin, validMax, globalND, ...
    'VariableNames', {'River','Resolution_m','Rows','Cols','Source','Output', ...
    'Unit_action','ZeroIsNoData','N_total','N_zero_input', ...
    'N_zero_reclassified','N_source_declared_nodata','N_nonfinite','N_valid', ...
    'Valid_min','Valid_max','Unified_NoData'});
writetable(T, summaryCsv);

fprintf('Input zero pixels       : %d\n', N_zero_input);
fprintf('Zero reclassified       : %d\n', N_zero_reclassified);
fprintf('Declared NoData pixels  : %d\n', N_source_declared_nd);
fprintf('Nonfinite pixels        : %d\n', N_nonfinite);
fprintf('Valid range             : %.8f to %.8f\n', validMin, validMax);
fprintf('Output VRT              : %s\n', outVrt);
end
