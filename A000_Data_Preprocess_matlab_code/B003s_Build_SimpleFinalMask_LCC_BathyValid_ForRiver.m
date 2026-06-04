function B003s_Build_SimpleFinalMask_LCC_BathyValid_ForRiver(river, varargin)
%% ============================================================
%  B003s_Build_SimpleFinalMask_LCC_BathyValid_ForRiver.m
%
%  Purpose:
%    Build a simple and robust MAE final prediction mask for ONE river:
%
%      final_mask = (LCC == 1) & bathy_valid
%
%  Meaning:
%    1 = pixels to be masked / predicted by MAE
%    0 = known pixels / not prediction target
%
%  Why this simplified version:
%    - Avoid bathy NoData pixels entering the mask.
%    - Prevent RMSE inflation from regions without bathy truth.
%    - Keep the first downstream training experiment simple and stable.
%    - More advanced diff/water-probability logic can be added later.
%
%  Inputs:
%    Bathy_<res>m_FixND/<river>/Bathy_<res>m.vrt
%    LCC_<res>m/<river>/ESA_WorldCover_Resampleandclip_<res>m.vrt
%
%  Output:
%    PredictionMask_LCCBathyValid_<res>m/<river>/MAE_PredictionMask_<res>m.vrt
%
%  Example:
%    B003s_Build_SimpleFinalMask_LCC_BathyValid_ForRiver( ...
%        'MD_PotomacRiver_Bathy_2019', ...
%        'targetRes', [1])
% ============================================================

%% -------------------- Parse inputs --------------------
p = inputParser;

addRequired(p, 'river', @(x) ischar(x) || isstring(x));

addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));

addParameter(p, 'targetRes', [1], @isnumeric);

addParameter(p, 'bathyND', -999999, @isnumeric);

% Final prediction mask:
% 0 = known / not predicted
% 1 = predicted
% 255 = mask NoData metadata, but normally we write only 0/1
addParameter(p, 'maskND', 255, @isnumeric);

addParameter(p, 'tile', 2048, @isnumeric);
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'doPathSetup', true, @islogical);

parse(p, river, varargin{:});

river = char(p.Results.river);
rootPR = char(p.Results.rootPR);
targetRes = p.Results.targetRes;

bathyND = p.Results.bathyND;
maskND  = p.Results.maskND;
tile    = p.Results.tile;

overwrite = p.Results.overwrite;
doPathSetup = p.Results.doPathSetup;

%% -------------------- Path setup --------------------
if doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

%% -------------------- Diagnostic folder --------------------
diagDir = fullfile(rootPR, 'Z013_SimpleFinalMask_Diagnostics');
if exist(diagDir, 'dir') ~= 7
    mkdir(diagDir);
end

safeRiver = regexprep(river, '[^\w\d_]+', '_');
diagCSV = fullfile(diagDir, sprintf('%s_SimpleFinalMask_Diagnostics.csv', safeRiver));

diagRows = {};
diagRows(1,:) = { ...
    'River', ...
    'Resolution_m', ...
    'Rows', ...
    'Cols', ...
    'N_total', ...
    'N_LCC_candidate', ...
    'N_bathy_valid', ...
    'N_LCC_and_bathy_valid_final', ...
    'N_LCC_bathy_NoData_removed', ...
    'FinalMask_fraction'};

fprintf('\n============================================================\n');
fprintf('Build SIMPLE final mask for river: %s\n', river);
fprintf('Logic: final_mask = (LCC == 1) & bathy_valid\n');
fprintf('targetRes = %s\n', mat2str(targetRes));
fprintf('============================================================\n');

%% ============================================================
%  Main loop over resolution
% ============================================================
for j = 1:numel(targetRes)

    res = targetRes(j);

    bathy_vrt = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res), ...
        river, sprintf('Bathy_%dm.vrt', res));

    lcc_vrt = fullfile(rootPR, sprintf('LCC_%dm', res), ...
        river, sprintf('ESA_WorldCover_Resampleandclip_%dm.vrt', res));

    if exist(bathy_vrt, 'file') ~= 2
        warning('[%s %dm] Missing bathy: %s', river, res, bathy_vrt);
        continue;
    end

    if exist(lcc_vrt, 'file') ~= 2
        warning('[%s %dm] Missing LCC: %s', river, res, lcc_vrt);
        continue;
    end

    %% -------------------- Grid check --------------------
    [~, rowsB, colsB, geoB, projB, dataTypeB, ~] = RasterInfo(bathy_vrt);
    [~, rowsL, colsL, geoL, ~, dataTypeL, ~] = RasterInfo(lcc_vrt);

    if rowsB ~= rowsL || colsB ~= colsL
        error('[%s %dm] Size mismatch: Bathy=%d/%d, LCC=%d/%d. Fix LCC first.', ...
            river, res, rowsB, colsB, rowsL, colsL);
    end

    maxGeoDiff = max(abs(geoB(:) - geoL(:)));
    if maxGeoDiff > 1e-8
        error('[%s %dm] GeoTransform mismatch: %.12g. Fix LCC first.', ...
            river, res, maxGeoDiff);
    end

    %% -------------------- Output paths --------------------
    outSub = fullfile(rootPR, sprintf('PredictionMask_LCCBathyValid_%dm', res), river);
    if exist(outSub, 'dir') ~= 7
        mkdir(outSub);
    end

    tilesDir = fullfile(outSub, '_tiles');
    if exist(tilesDir, 'dir') ~= 7
        mkdir(tilesDir);
    end

    outVrt = fullfile(outSub, sprintf('MAE_PredictionMask_%dm.vrt', res));
    listTxt = fullfile(outSub, 'tile_list.txt');

    if exist(outVrt, 'file') == 2
        if overwrite
            delete(outVrt);
        else
            warning('[%s %dm] Output exists and overwrite=false. Skip.', river, res);
            continue;
        end
    end

    if exist(listTxt, 'file') == 2
        delete(listTxt);
    end

    system(sprintf('rm -f "%s"/tile_*.tif', tilesDir));

    %% -------------------- Tile loop --------------------
    fprintf('\n[%s %dm] Build simple final mask\n', river, res);
    fprintf('  Bathy: %s\n', bathy_vrt);
    fprintf('  LCC  : %s\n', lcc_vrt);
    fprintf('  rows=%d cols=%d\n', rowsB, colsB);

    totalTiles = ceil(rowsB / tile) * ceil(colsB / tile);
    tileCount = 0;

    N_total = rowsB * colsB;
    N_LCC_candidate = 0;
    N_bathy_valid = 0;
    N_final = 0;
    N_removed_bathyND = 0;

    for rLocal = 1:tile:rowsB

        rr = min(tile, rowsB - rLocal + 1);

        for cLocal = 1:tile:colsB

            cc = min(tile, colsB - cLocal + 1);

            B = double(ReadRaster(bathy_vrt, rLocal, cLocal, rr, cc));
            L = double(ReadRaster(lcc_vrt,   rLocal, cLocal, rr, cc));

            bathy_valid = isfinite(B) & ~isnan(B) & ...
                          (B ~= bathyND) & (B > -1e20);

            lcc_candidate = isfinite(L) & ~isnan(L) & (L == 1);

            final_mask = lcc_candidate & bathy_valid;

            N_LCC_candidate = N_LCC_candidate + nnz(lcc_candidate);
            N_bathy_valid = N_bathy_valid + nnz(bathy_valid);
            N_final = N_final + nnz(final_mask);
            N_removed_bathyND = N_removed_bathyND + nnz(lcc_candidate & ~bathy_valid);

            final_u8 = uint8(final_mask);

            subgeoTrans = subTranscoef(geoB, rLocal, cLocal);
            tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));

            WriteRaster(tileTif, double(final_u8), subgeoTrans, projB, ...
                dataTypeL, 'GTiff', maskND);

            tileCount = tileCount + 1;
            fprintf('\r  Progress: %6.2f%% (%d/%d)', ...
                100 * tileCount / totalTiles, tileCount, totalTiles);

            clear B L bathy_valid lcc_candidate final_mask final_u8
        end
    end

    fprintf('\nTile writing done.\n');

    %% -------------------- Build VRT --------------------
    cmdList = sprintf('find "%s" -maxdepth 1 -type f -name "tile_*.tif" | sort > "%s"', ...
        tilesDir, listTxt);

    statusList = system(cmdList);
    if statusList ~= 0
        error('Failed to build tile list: %s', cmdList);
    end

    info = dir(listTxt);
    if isempty(info) || info.bytes == 0
        error('tile_list.txt is empty: %s', listTxt);
    end

    cmdV = sprintf('gdalbuildvrt -overwrite -vrtnodata %g -input_file_list "%s" "%s"', ...
        maskND, listTxt, outVrt);

    statusV = system(cmdV);
    if statusV ~= 0
        error('gdalbuildvrt failed: %s', cmdV);
    end

    FinalMask_fraction = N_final / N_total;

    fprintf('\n[%s %dm] Summary\n', river, res);
    fprintf('  LCC candidate pixels        : %d\n', N_LCC_candidate);
    fprintf('  Bathy valid pixels          : %d\n', N_bathy_valid);
    fprintf('  LCC & bathy valid final     : %d\n', N_final);
    fprintf('  LCC bathy NoData removed    : %d\n', N_removed_bathyND);
    fprintf('  Final mask fraction         : %.6f\n', FinalMask_fraction);
    fprintf('  Output mask VRT             : %s\n', outVrt);

    diagRows(end+1,:) = { ...
        river, ...
        res, ...
        rowsB, ...
        colsB, ...
        N_total, ...
        N_LCC_candidate, ...
        N_bathy_valid, ...
        N_final, ...
        N_removed_bathyND, ...
        FinalMask_fraction};
end

writecell(diagRows, diagCSV);

fprintf('\n============================================================\n');
fprintf('Simple final mask done for river:\n%s\n', river);
fprintf('Diagnostics written to:\n%s\n', diagCSV);
fprintf('============================================================\n');

end