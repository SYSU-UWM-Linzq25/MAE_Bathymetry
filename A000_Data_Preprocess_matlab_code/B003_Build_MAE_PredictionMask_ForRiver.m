function B003_Build_MAE_PredictionMask_ForRiver(river, varargin)
%% ============================================================
%  B003_Build_MAE_PredictionMask_ForRiver.m
%
%  Purpose:
%    Build final MAE prediction mask for ONE river.
%
%  Core philosophy:
%    We prefer slightly over-masking rather than under-masking.
%    Under-masking may leak bathy information into the model input.
%
%  Final mask meaning:
%    1 = pixels to be masked/predicted by MAE
%    0 = known pixels / not prediction target
%
%  Final logic:
%
%    A) LCC base candidate:
%       lcc_base_candidate =
%           LCC == 1 & bathy_valid
%
%       This is always kept in final_mask.
%       Reason:
%         LCC is treated as a strong river/water prior.
%         If bathy is valid and LCC says river/water, we mask it
%         to avoid leaking bathy into model input.
%
%    B) Diff expansion outside LCC:
%       diff_expand_candidate =
%           LCC == 0 & bathy_valid & dem_valid &
%           abs(bathy - 3DEP) > threshold
%
%       This expands the mask outside LCC when bathy and 3DEP differ enough.
%
%    C) Final mask:
%       final_mask =
%           lcc_base_candidate | diff_expand_candidate
%
%  Hard rule:
%    bathy NoData pixels are NEVER included in final_mask.
%    If bathy is NoData, it cannot be used for loss/RMSE.
%
%  Threshold modes:
%    thresholdMode = 'auto_lcc'
%       Automatically estimate threshold from outside-LCC diff distribution.
%       Default uses outside-LCC p85.
%
%    thresholdMode = 'manual'
%       Use user-provided diffThreshold_m.
%
%  Example auto:
%    B003_Build_MAE_PredictionMask_ForRiver( ...
%        'KewaFix2Null', ...
%        'targetRes', [1], ...
%        'thresholdMode', 'auto_lcc', ...
%        'autoOutsideQuantile', 85)
%
%  Example manual:
%    B003_Build_MAE_PredictionMask_ForRiver( ...
%        'KewaFix2Null', ...
%        'targetRes', [1], ...
%        'thresholdMode', 'manual', ...
%        'diffThreshold_m', 0.50)
%
%  Output:
%    PredictionMask_<res>m/<river>/MAE_PredictionMask_<res>m.vrt
%    PredictionMask_<res>m/<river>/_tiles/*.tif
%
%  Diagnostics:
%    Z006_PredictionMask_Diagnostics/<river>_B003_*.csv
%    Z007_PredictionMask_VisualCheck/<river>/<res>m/*.tif
% ============================================================

%% -------------------- Parse inputs --------------------
p = inputParser;

addRequired(p, 'river', @(x) ischar(x) || isstring(x));

addParameter(p, 'rootPR', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', ...
    @(x) ischar(x) || isstring(x));

addParameter(p, 'targetRes', [1 3 5 10], @isnumeric);

% Manual threshold interface.
% Only used when thresholdMode = 'manual'.
addParameter(p, 'diffThreshold_m', 0.50, @isnumeric);

% Threshold mode:
%   'auto_lcc' : estimate threshold from outside-LCC diff distribution.
%   'manual'   : use diffThreshold_m.
addParameter(p, 'thresholdMode', 'auto_lcc', ...
    @(x) ischar(x) || isstring(x));

% Auto threshold controls.
% Because we prefer over-masking, p85 is a safer default than p90.
% Larger quantile -> stricter threshold -> smaller outside-LCC expansion.
% Smaller quantile -> looser threshold -> larger outside-LCC expansion.
addParameter(p, 'autoOutsideQuantile', 85, @isnumeric);

% Minimum/maximum threshold constraints.
% Use autoMaxThreshold_m = Inf by default so the data controls it.
% You may set autoMaxThreshold_m = 1.0 if you want to avoid very high thresholds.
addParameter(p, 'autoMinThreshold_m', 0.05, @isnumeric);
addParameter(p, 'autoMaxThreshold_m', Inf, @isnumeric);

% Histogram approximation settings for auto threshold.
% This avoids storing huge OR_MKRC diff arrays in memory.
addParameter(p, 'autoHistBinWidth_m', 0.01, @isnumeric);
addParameter(p, 'autoHistMaxDiff_m', 10.0, @isnumeric);

addParameter(p, 'bathyND', -999999, @isnumeric);
addParameter(p, 'demND', -999999, @isnumeric);

% final prediction mask is 0/1.
% Use 255 as mask NoData because 0 is a valid class.
addParameter(p, 'maskND', 255, @isnumeric);

addParameter(p, 'tile', 2048, @isnumeric);

addParameter(p, 'saveVerify', true, @islogical);
addParameter(p, 'nVerifyTilesPerRiverRes', 3, @isnumeric);
addParameter(p, 'minFinalPixForVerify', 100, @isnumeric);
addParameter(p, 'minDiffExpandPixForVerify', 100, @isnumeric);
addParameter(p, 'minRemovedPixForVerify', 100, @isnumeric);

% Set false if you already loaded GDALLoad() and paths in the current session.
addParameter(p, 'doPathSetup', true, @islogical);

parse(p, river, varargin{:});

river = char(p.Results.river);
rootPR = char(p.Results.rootPR);
targetRes = p.Results.targetRes;

thresholdMode = lower(char(p.Results.thresholdMode));
diffThreshold_manual_m = p.Results.diffThreshold_m;

autoOutsideQuantile = p.Results.autoOutsideQuantile;
autoMinThreshold_m = p.Results.autoMinThreshold_m;
autoMaxThreshold_m = p.Results.autoMaxThreshold_m;
autoHistBinWidth_m = p.Results.autoHistBinWidth_m;
autoHistMaxDiff_m = p.Results.autoHistMaxDiff_m;

bathyND = p.Results.bathyND;
demND   = p.Results.demND;
maskND  = p.Results.maskND;

tile = p.Results.tile;

saveVerify = p.Results.saveVerify;
nVerifyTilesPerRiverRes = p.Results.nVerifyTilesPerRiverRes;
minFinalPixForVerify = p.Results.minFinalPixForVerify;
minDiffExpandPixForVerify = p.Results.minDiffExpandPixForVerify;
minRemovedPixForVerify = p.Results.minRemovedPixForVerify;

doPathSetup = p.Results.doPathSetup;

validModes = {'auto_lcc', 'manual'};
if ~ismember(thresholdMode, validModes)
    error('Invalid thresholdMode: %s. Use auto_lcc or manual.', thresholdMode);
end

%% -------------------- Path setup --------------------
if doPathSetup
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();

    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

%% -------------------- Output folders --------------------
outDiagDir = fullfile(rootPR, 'Z006_PredictionMask_Diagnostics');
if exist(outDiagDir, 'dir') ~= 7
    mkdir(outDiagDir);
end

verifyRoot = fullfile(rootPR, 'Z007_PredictionMask_VisualCheck');
if exist(verifyRoot, 'dir') ~= 7
    mkdir(verifyRoot);
end

%% -------------------- Diagnostic table --------------------
diagRows = {};
diagHeader = { ...
    'River', ...
    'Resolution_m', ...
    'Rows', ...
    'Cols', ...
    'N_total', ...
    'N_LCC_candidate', ...
    'N_LCC_base_candidate', ...
    'N_bathy_valid', ...
    'N_dem_valid', ...
    'N_diff_valid_pair', ...
    'N_diff_expand_outside_LCC', ...
    'N_LCC_bathy_nodata_removed', ...
    'N_LCC_small_diff_kept', ...
    'N_LCC_dem_missing_kept', ...
    'N_final_mask', ...
    'FinalMask_fraction', ...
    'ThresholdMode', ...
    'DiffThreshold_m', ...
    'AutoOutsideQuantile', ...
    'AutoOutsideThresholdRaw_m', ...
    'AutoNOutsideValidPair', ...
    'AutoNOutsideAboveThreshold', ...
    'AutoOutsideAboveThresholdFraction'};
diagRows(1,:) = diagHeader;

fprintf('\n============================================================\n');
fprintf('B003 MAE prediction mask for river: %s\n', river);
fprintf('thresholdMode = %s\n', thresholdMode);
if strcmpi(thresholdMode, 'manual')
    fprintf('manual diffThreshold_m = %.3f m\n', diffThreshold_manual_m);
else
    fprintf('autoOutsideQuantile = %.1f\n', autoOutsideQuantile);
    fprintf('autoMinThreshold_m  = %.3f\n', autoMinThreshold_m);
    if isfinite(autoMaxThreshold_m)
        fprintf('autoMaxThreshold_m  = %.3f\n', autoMaxThreshold_m);
    else
        fprintf('autoMaxThreshold_m  = Inf\n');
    end
end
fprintf('targetRes = %s\n', mat2str(targetRes));
fprintf('============================================================\n');

%% ============================================================
%  Main loop over resolutions
% ============================================================
for j = 1:numel(targetRes)

    res = targetRes(j);

    %% -------------------- Input paths --------------------
    bathy_vrt = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res), ...
        river, sprintf('Bathy_%dm.vrt', res));

    lcc_vrt = fullfile(rootPR, sprintf('LCC_%dm', res), ...
        river, sprintf('ESA_WorldCover_Resampleandclip_%dm.vrt', res));

    if exist(bathy_vrt, 'file') ~= 2
        warning('Skip %s %dm: missing bathy %s', river, res, bathy_vrt);
        continue;
    end

    if exist(lcc_vrt, 'file') ~= 2
        warning('Skip %s %dm: missing LCC %s', river, res, lcc_vrt);
        continue;
    end

    %% -------------------- Output paths --------------------
    outSub = fullfile(rootPR, sprintf('PredictionMask_%dm', res), river);
    if exist(outSub, 'dir') ~= 7
        mkdir(outSub);
    end

    tilesDir = fullfile(outSub, '_tiles');
    if exist(tilesDir, 'dir') ~= 7
        mkdir(tilesDir);
    end

    outVrt = fullfile(outSub, sprintf('MAE_PredictionMask_%dm.vrt', res));
    listTxt = fullfile(outSub, 'tile_list.txt');

    % Clean previous outputs for this river/res.
    system(sprintf('rm -f "%s"/tile_*.tif', tilesDir));
    if exist(outVrt, 'file') == 2
        delete(outVrt);
    end
    if exist(listTxt, 'file') == 2
        delete(listTxt);
    end

    verifyDir = fullfile(verifyRoot, river, sprintf('%dm', res));
    if exist(verifyDir, 'dir') ~= 7
        mkdir(verifyDir);
    end

    % Clean old visual check files for this river/res.
    system(sprintf('rm -f "%s"/*.tif', verifyDir));

    %% -------------------- 3DEP source --------------------
    % For 1m, use existing 3DEP resampled-to-bathy grid.
    % For 3m/5m/10m, build a temporary VRT aligned to the current bathy grid.
    if res == 1
        dem_vrt = fullfile(rootPR, '3DEP_1m_ResampleClip', ...
            river, 'DEM_3DEP_1m_ResampleandClip.vrt');

        if exist(dem_vrt, 'file') ~= 2
            warning('Skip %s 1m: missing 3DEP %s', river, dem_vrt);
            continue;
        end
    else
        dem1m_vrt = fullfile(rootPR, '3DEP_1m_ResampleClip', ...
            river, 'DEM_3DEP_1m_ResampleandClip.vrt');

        if exist(dem1m_vrt, 'file') ~= 2
            warning('Skip %s %dm: missing 1m 3DEP %s', river, res, dem1m_vrt);
            continue;
        end

        tmpDemDir = fullfile(outSub, '_tmp_3DEP_to_bathy_grid');
        if exist(tmpDemDir, 'dir') ~= 7
            mkdir(tmpDemDir);
        end

        dem_vrt = fullfile(tmpDemDir, sprintf('DEM_3DEP_%dm_to_bathygrid.vrt', res));

        [~, rowsT, colsT, geoT, projT, ~, ~] = RasterInfo(bathy_vrt);

        xmin = geoT(1);
        xres = geoT(2);
        ymax = geoT(4);
        yres = geoT(6);

        xmax = xmin + colsT * xres;
        ymin = ymax + rowsT * yres;

        proj_arg = sprintf('''%s''', projT);

        cmdDem = sprintf([ ...
            'gdalwarp -overwrite -of VRT ', ...
            '-r average ', ...
            '-t_srs %s -te_srs %s ', ...
            '-te %.10f %.10f %.10f %.10f ', ...
            '-ts %d %d ', ...
            '-srcnodata %g -dstnodata %g ', ...
            '-wo INIT_DEST=NO_DATA -wo SKIP_NOSOURCE=YES ', ...
            '"%s" "%s"' ], ...
            proj_arg, proj_arg, ...
            xmin, ymin, xmax, ymax, ...
            colsT, rowsT, ...
            demND, demND, ...
            dem1m_vrt, dem_vrt);

        fprintf('\nBuild temporary 3DEP %dm VRT:\n%s\n', res, cmdDem);
        statusDem = system(cmdDem);

        if statusDem ~= 0
            warning('Failed to build temporary 3DEP VRT for %s %dm', river, res);
            continue;
        end
    end

    %% -------------------- Grid check --------------------
    [~, rowsB, colsB, geoB, projB, dataTypeB, ~] = RasterInfo(bathy_vrt);
    [~, rowsL, colsL, geoL, projL, dataTypeL, ~] = RasterInfo(lcc_vrt);
    [~, rowsD, colsD, geoD, projD, ~, ~] = RasterInfo(dem_vrt);

    if rowsB ~= rowsL || colsB ~= colsL || rowsB ~= rowsD || colsB ~= colsD
        warning('Grid size mismatch: %s %dm. Skip.', river, res);
        fprintf('  Bathy rows/cols = %d/%d\n', rowsB, colsB);
        fprintf('  LCC   rows/cols = %d/%d\n', rowsL, colsL);
        fprintf('  DEM   rows/cols = %d/%d\n', rowsD, colsD);
        continue;
    end

    maxGeoDiff_L = max(abs(geoB(:) - geoL(:)));
    maxGeoDiff_D = max(abs(geoB(:) - geoD(:)));

    if maxGeoDiff_L > 1e-6 || maxGeoDiff_D > 1e-6
        warning('GeoTransform mismatch: %s %dm. LCC diff=%.3g, DEM diff=%.3g', ...
            river, res, maxGeoDiff_L, maxGeoDiff_D);
    end

    %% -------------------- Determine threshold --------------------
    if strcmpi(thresholdMode, 'manual')

        diffThreshold_this_m = diffThreshold_manual_m;

        autoOutsideThresholdRaw_m = NaN;
        autoNOutsideValidPair = NaN;
        autoNOutsideAboveThreshold = NaN;
        autoOutsideAboveThresholdFraction = NaN;

    else

        autoStats = estimateAutoThresholdOutsideLCC( ...
            bathy_vrt, dem_vrt, lcc_vrt, ...
            bathyND, demND, tile, ...
            autoOutsideQuantile, ...
            autoHistBinWidth_m, ...
            autoHistMaxDiff_m, ...
            autoMinThreshold_m, ...
            autoMaxThreshold_m);

        diffThreshold_this_m = autoStats.threshold_m;

        autoOutsideThresholdRaw_m = autoStats.thresholdRaw_m;
        autoNOutsideValidPair = autoStats.N_outside_valid_pair;
        autoNOutsideAboveThreshold = autoStats.N_outside_above_threshold;
        autoOutsideAboveThresholdFraction = autoStats.outside_above_threshold_fraction;
    end

    fprintf('\n[%s %dm] Build MAE prediction mask\n', river, res);
    fprintf('  Bathy: %s\n', bathy_vrt);
    fprintf('  3DEP : %s\n', dem_vrt);
    fprintf('  LCC  : %s\n', lcc_vrt);
    fprintf('  rows=%d cols=%d\n', rowsB, colsB);
    fprintf('  thresholdMode = %s\n', thresholdMode);
    fprintf('  diffThreshold_this_m = %.4f\n', diffThreshold_this_m);

    if strcmpi(thresholdMode, 'auto_lcc')
        fprintf('  auto raw outside-LCC p%.1f = %.4f m\n', ...
            autoOutsideQuantile, autoOutsideThresholdRaw_m);
        fprintf('  auto outside valid pair pixels = %d\n', autoNOutsideValidPair);
        fprintf('  auto outside above threshold   = %d (%.4f)\n', ...
            autoNOutsideAboveThreshold, autoOutsideAboveThresholdFraction);
    end

    %% -------------------- Tile loop --------------------
    totalTiles = ceil(rowsB / tile) * ceil(colsB / tile);
    tileCount = 0;
    verifySaved = 0;

    % Diagnostics counters
    N_total = rowsB * colsB;
    N_LCC_candidate = 0;
    N_LCC_base_candidate = 0;
    N_bathy_valid = 0;
    N_dem_valid = 0;
    N_diff_valid_pair = 0;
    N_diff_expand_outside_LCC = 0;
    N_LCC_bathy_nodata_removed = 0;
    N_LCC_small_diff_kept = 0;
    N_LCC_dem_missing_kept = 0;
    N_final_mask = 0;

    for rLocal = 1:tile:rowsB

        rr = min(tile, rowsB - rLocal + 1);

        for cLocal = 1:tile:colsB

            cc = min(tile, colsB - cLocal + 1);

            %% Read tile
            B = double(ReadRaster(bathy_vrt, rLocal, cLocal, rr, cc));
            D = double(ReadRaster(dem_vrt,   rLocal, cLocal, rr, cc));
            L = double(ReadRaster(lcc_vrt,   rLocal, cLocal, rr, cc));

            %% Valid masks
            bathy_valid = isfinite(B) & ~isnan(B) & ...
                          (B ~= bathyND) & (B > -1e20);

            dem_valid = isfinite(D) & ~isnan(D) & ...
                        (D ~= demND) & (D > -1e20);

            % LCC is a binary 0/1 mask.
            % 0 and 1 are both valid. Only 1 is candidate.
            lcc_candidate = isfinite(L) & ~isnan(L) & (L == 1);

            diffBD = abs(B - D);
            diff_valid_pair = bathy_valid & dem_valid;

            %% Source A: LCC base mask
            % If LCC=1 and bathy is valid, always mask it.
            % This prevents bathy leakage in high-confidence river/water areas.
            lcc_base_candidate = lcc_candidate & bathy_valid;

            %% Source B: diff expansion outside LCC
            % Diff is used only to expand outside LCC.
            diff_expand_outside_LCC = (~lcc_candidate) & diff_valid_pair & ...
                                      (diffBD > diffThreshold_this_m);

            %% Diagnostic masks
            % LCC says candidate, but bathy is NoData -> removed.
            lcc_bathy_nodata_removed = lcc_candidate & ~bathy_valid;

            % LCC says candidate, bathy/DEM valid, and diff is small.
            % Under the new over-mask-safe logic, these pixels are kept,
            % not removed.
            lcc_small_diff_kept = lcc_candidate & bathy_valid & dem_valid & ...
                                  (diffBD <= diffThreshold_this_m);

            % LCC says candidate, bathy valid, but DEM is missing.
            % These are kept because we cannot compute diff.
            lcc_dem_missing_kept = lcc_candidate & bathy_valid & ~dem_valid;

            %% Final MAE prediction mask
            % final_mask is not necessarily a subset of LCC.
            % final_mask is always a subset of bathy_valid.
            final_mask = lcc_base_candidate | diff_expand_outside_LCC;
            final_mask_u8 = uint8(final_mask);

            %% Accumulate diagnostics
            N_LCC_candidate = N_LCC_candidate + nnz(lcc_candidate);
            N_LCC_base_candidate = N_LCC_base_candidate + nnz(lcc_base_candidate);
            N_bathy_valid = N_bathy_valid + nnz(bathy_valid);
            N_dem_valid = N_dem_valid + nnz(dem_valid);
            N_diff_valid_pair = N_diff_valid_pair + nnz(diff_valid_pair);
            N_diff_expand_outside_LCC = N_diff_expand_outside_LCC + nnz(diff_expand_outside_LCC);
            N_LCC_bathy_nodata_removed = N_LCC_bathy_nodata_removed + nnz(lcc_bathy_nodata_removed);
            N_LCC_small_diff_kept = N_LCC_small_diff_kept + nnz(lcc_small_diff_kept);
            N_LCC_dem_missing_kept = N_LCC_dem_missing_kept + nnz(lcc_dem_missing_kept);
            N_final_mask = N_final_mask + nnz(final_mask);

            %% Write final mask tile
            subgeoTrans = subTranscoef(geoB, rLocal, cLocal);
            tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));

            WriteRaster(tileTif, double(final_mask_u8), subgeoTrans, projB, ...
                dataTypeL, 'GTiff', maskND);

            %% Save visual check tiles
            if saveVerify && verifySaved < nVerifyTilesPerRiverRes

                nFinal = nnz(final_mask);
                nDiffExpand = nnz(diff_expand_outside_LCC);
                nRemovedND = nnz(lcc_bathy_nodata_removed);
                nSmallDiffKept = nnz(lcc_small_diff_kept);
                nLccDemMissing = nnz(lcc_dem_missing_kept);

                shouldSave = ...
                    (nFinal >= minFinalPixForVerify) || ...
                    (nDiffExpand >= minDiffExpandPixForVerify) || ...
                    (nRemovedND >= minRemovedPixForVerify) || ...
                    (nSmallDiffKept >= minRemovedPixForVerify) || ...
                    (nLccDemMissing >= minRemovedPixForVerify);

                if shouldSave

                    tag = sprintf('Tile_r%06d_c%06d', rLocal, cLocal);

                    BathyValid_u8 = uint8(bathy_valid);
                    DEMValid_u8 = uint8(dem_valid);
                    LCC_u8 = uint8(lcc_candidate);
                    LCCBase_u8 = uint8(lcc_base_candidate);
                    DiffExpandOutsideLCC_u8 = uint8(diff_expand_outside_LCC);
                    LCC_BathyNoData_Removed_u8 = uint8(lcc_bathy_nodata_removed);
                    LCC_SmallDiff_Kept_u8 = uint8(lcc_small_diff_kept);
                    LCC_DEMInvalid_Kept_u8 = uint8(lcc_dem_missing_kept);

                    % For diff visualization, invalid diff is set to bathyND.
                    DiffVis = diffBD;
                    DiffVis(~diff_valid_pair) = bathyND;

                    WriteRaster(fullfile(verifyDir, [tag '_Bathy.tif']), ...
                        B, subgeoTrans, projB, dataTypeB, 'GTiff', bathyND);

                    WriteRaster(fullfile(verifyDir, [tag '_DEM_3DEP.tif']), ...
                        D, subgeoTrans, projB, dataTypeB, 'GTiff', demND);

                    WriteRaster(fullfile(verifyDir, [tag '_Diff_m.tif']), ...
                        DiffVis, subgeoTrans, projB, dataTypeB, 'GTiff', bathyND);

                    WriteRaster(fullfile(verifyDir, [tag '_LCC.tif']), ...
                        double(LCC_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_BathyValid.tif']), ...
                        double(BathyValid_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_DEMValid.tif']), ...
                        double(DEMValid_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_LCC_BaseCandidate.tif']), ...
                        double(LCCBase_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_DiffExpand_OutsideLCC.tif']), ...
                        double(DiffExpandOutsideLCC_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_LCC_BathyNoData_Removed.tif']), ...
                        double(LCC_BathyNoData_Removed_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_LCC_SmallDiff_Kept.tif']), ...
                        double(LCC_SmallDiff_Kept_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_LCC_DEMInvalid_Kept.tif']), ...
                        double(LCC_DEMInvalid_Kept_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    WriteRaster(fullfile(verifyDir, [tag '_Final_MAE_PredictionMask.tif']), ...
                        double(final_mask_u8), subgeoTrans, projB, dataTypeL, 'GTiff', maskND);

                    verifySaved = verifySaved + 1;

                    fprintf('\n  [VERIFY %d/%d] %s\n', ...
                        verifySaved, nVerifyTilesPerRiverRes, tag);
                    fprintf('    final=%d, LCCbase=%d, diffExpandOutsideLCC=%d, keptSmallDiff=%d, keptDEMInvalid=%d, removedBathyND=%d\n', ...
                        nFinal, nnz(lcc_base_candidate), nDiffExpand, ...
                        nSmallDiffKept, nLccDemMissing, nRemovedND);
                end
            end

            tileCount = tileCount + 1;
            fprintf('\r  Progress: %6.2f%% (%d/%d)', ...
                100 * tileCount / totalTiles, tileCount, totalTiles);

            clear B D L diffBD final_mask final_mask_u8
            clear bathy_valid dem_valid lcc_candidate
            clear lcc_base_candidate diff_expand_outside_LCC
        end
    end

    fprintf('\nTile processing done.\n');

    %% Build VRT from mask tiles
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
        error('gdalbuildvrt failed for %s', outVrt);
    end

    %% Print summary
    FinalMask_fraction = N_final_mask / N_total;

    if strcmpi(thresholdMode, 'auto_lcc') && ~isnan(autoNOutsideValidPair)
        autoNOutsideAboveThreshold = N_diff_expand_outside_LCC;
        autoOutsideAboveThresholdFraction = ...
            autoNOutsideAboveThreshold / max(autoNOutsideValidPair, 1);
    end

    fprintf('\n[%s %dm] Summary\n', river, res);
    fprintf('  Threshold mode                  : %s\n', thresholdMode);
    fprintf('  Diff threshold used             : %.4f m\n', diffThreshold_this_m);
    fprintf('  LCC candidate pixels            : %d\n', N_LCC_candidate);
    fprintf('  LCC base candidate pixels       : %d\n', N_LCC_base_candidate);
    fprintf('  Bathy valid pixels              : %d\n', N_bathy_valid);
    fprintf('  DEM valid pixels                : %d\n', N_dem_valid);
    fprintf('  Diff valid pair pixels          : %d\n', N_diff_valid_pair);
    fprintf('  Diff expand outside LCC         : %d\n', N_diff_expand_outside_LCC);
    fprintf('  LCC bathy NoData removed        : %d\n', N_LCC_bathy_nodata_removed);
    fprintf('  LCC small-diff kept             : %d\n', N_LCC_small_diff_kept);
    fprintf('  LCC DEM missing kept            : %d\n', N_LCC_dem_missing_kept);
    fprintf('  Final mask pixels               : %d (%.6f)\n', ...
        N_final_mask, FinalMask_fraction);
    fprintf('  Output mask VRT                 : %s\n', outVrt);

    %% Append diagnostics
    diagRows(end+1,:) = { ...
        river, ...
        res, ...
        rowsB, ...
        colsB, ...
        N_total, ...
        N_LCC_candidate, ...
        N_LCC_base_candidate, ...
        N_bathy_valid, ...
        N_dem_valid, ...
        N_diff_valid_pair, ...
        N_diff_expand_outside_LCC, ...
        N_LCC_bathy_nodata_removed, ...
        N_LCC_small_diff_kept, ...
        N_LCC_dem_missing_kept, ...
        N_final_mask, ...
        FinalMask_fraction, ...
        thresholdMode, ...
        diffThreshold_this_m, ...
        autoOutsideQuantile, ...
        autoOutsideThresholdRaw_m, ...
        autoNOutsideValidPair, ...
        autoNOutsideAboveThreshold, ...
        autoOutsideAboveThresholdFraction};
end

%% -------------------- Write diagnostics --------------------
safeRiver = regexprep(river, '[^\w\d_]+', '_');

if strcmpi(thresholdMode, 'manual')
    thresholdTag = sprintf('manual_diff%s', ...
        strrep(sprintf('%.3f', diffThreshold_manual_m), '.', 'p'));
else
    thresholdTag = sprintf('autoOutsideP%s', ...
        strrep(sprintf('%.1f', autoOutsideQuantile), '.', 'p'));
end

diagCSV = fullfile(outDiagDir, ...
    sprintf('%s_B003_PredictionMask_Diagnostics_%s.csv', safeRiver, thresholdTag));

writecell(diagRows, diagCSV);

fprintf('\n============================================================\n');
fprintf('B003 prediction mask done for river:\n%s\n', river);
fprintf('Diagnostics written to:\n%s\n', diagCSV);
fprintf('Visual check tiles written under:\n%s\n', fullfile(verifyRoot, river));
fprintf('============================================================\n');

end

%% ============================================================
%  Local helper: estimate auto threshold from outside-LCC diff
% ============================================================
function S = estimateAutoThresholdOutsideLCC( ...
    bathy_vrt, dem_vrt, lcc_vrt, ...
    bathyND, demND, tile, ...
    autoOutsideQuantile, ...
    autoHistBinWidth_m, ...
    autoHistMaxDiff_m, ...
    autoMinThreshold_m, ...
    autoMaxThreshold_m)

    [~, rowsB, colsB, ~, ~, ~, ~] = RasterInfo(bathy_vrt);

    edges = 0:autoHistBinWidth_m:autoHistMaxDiff_m;
    if numel(edges) < 2
        error('autoHistBinWidth_m / autoHistMaxDiff_m gives invalid histogram edges.');
    end

    countsOutside = zeros(1, numel(edges)-1);
    overflowOutside = 0;

    N_outside_valid_pair = 0;

    fprintf('\nEstimating auto threshold from outside-LCC diff distribution...\n');
    fprintf('  outside quantile = p%.1f\n', autoOutsideQuantile);
    fprintf('  hist bin width   = %.4f m\n', autoHistBinWidth_m);
    fprintf('  hist max diff    = %.2f m\n', autoHistMaxDiff_m);

    totalTiles = ceil(rowsB / tile) * ceil(colsB / tile);
    tileCount = 0;

    for rLocal = 1:tile:rowsB

        rr = min(tile, rowsB - rLocal + 1);

        for cLocal = 1:tile:colsB

            cc = min(tile, colsB - cLocal + 1);

            B = double(ReadRaster(bathy_vrt, rLocal, cLocal, rr, cc));
            D = double(ReadRaster(dem_vrt,   rLocal, cLocal, rr, cc));
            L = double(ReadRaster(lcc_vrt,   rLocal, cLocal, rr, cc));

            bathy_valid = isfinite(B) & ~isnan(B) & ...
                          (B ~= bathyND) & (B > -1e20);

            dem_valid = isfinite(D) & ~isnan(D) & ...
                        (D ~= demND) & (D > -1e20);

            lcc_candidate = isfinite(L) & ~isnan(L) & (L == 1);

            outside_valid_pair = bathy_valid & dem_valid & ~lcc_candidate;

            if any(outside_valid_pair(:))
                diffVals = abs(B(outside_valid_pair) - D(outside_valid_pair));
                diffVals = diffVals(isfinite(diffVals) & ~isnan(diffVals));

                N_outside_valid_pair = N_outside_valid_pair + numel(diffVals);

                if ~isempty(diffVals)
                    inRange = diffVals <= autoHistMaxDiff_m;
                    countsOutside = countsOutside + histcounts(diffVals(inRange), edges);
                    overflowOutside = overflowOutside + nnz(~inRange);
                end
            end

            tileCount = tileCount + 1;
            fprintf('\r  Auto threshold progress: %6.2f%% (%d/%d)', ...
                100 * tileCount / totalTiles, tileCount, totalTiles);

            clear B D L bathy_valid dem_valid lcc_candidate outside_valid_pair diffVals
        end
    end

    fprintf('\n');

    thresholdRaw = approximateQuantileFromHist( ...
        countsOutside, overflowOutside, edges, autoOutsideQuantile);

    thresholdClamped = max(thresholdRaw, autoMinThreshold_m);

    if isfinite(autoMaxThreshold_m)
        thresholdClamped = min(thresholdClamped, autoMaxThreshold_m);
    end

    % Estimate outside pixels above final threshold.
    % For histogram approximation, count bins with lower edge > threshold
    % plus approximate threshold bin as fully above if its upper edge > threshold.
    N_above = approximateCountAboveThreshold( ...
        countsOutside, overflowOutside, edges, thresholdClamped);

    fracAbove = N_above / max(N_outside_valid_pair, 1);

    fprintf('  Raw outside p%.1f threshold = %.4f m\n', ...
        autoOutsideQuantile, thresholdRaw);
    fprintf('  Final threshold after clamp = %.4f m\n', thresholdClamped);
    fprintf('  Outside valid pair pixels   = %d\n', N_outside_valid_pair);
    fprintf('  Approx outside above thr    = %d (%.4f)\n', N_above, fracAbove);

    S = struct();
    S.threshold_m = thresholdClamped;
    S.thresholdRaw_m = thresholdRaw;
    S.N_outside_valid_pair = N_outside_valid_pair;
    S.N_outside_above_threshold = N_above;
    S.outside_above_threshold_fraction = fracAbove;
end

%% ============================================================
%  Local helper: approximate quantile from histogram
% ============================================================
function q = approximateQuantileFromHist(counts, overflowCount, edges, quantilePct)

    total = sum(counts) + overflowCount;

    if total <= 0
        warning('No valid outside-LCC diff pixels for auto threshold. Fallback to 0.');
        q = 0;
        return;
    end

    target = ceil((quantilePct / 100) * total);
    target = max(target, 1);

    cumCounts = cumsum(counts);

    idx = find(cumCounts >= target, 1, 'first');

    if isempty(idx)
        % Quantile falls in overflow range.
        q = edges(end);
    else
        % Use lower edge to slightly over-mask instead of under-mask.
        q = edges(idx);
    end
end

%% ============================================================
%  Local helper: approximate count above threshold from histogram
% ============================================================
function N_above = approximateCountAboveThreshold(counts, overflowCount, edges, threshold)

    N_above = overflowCount;

    for i = 1:numel(counts)
        binLow = edges(i);
        binHigh = edges(i+1);

        if binLow > threshold
            N_above = N_above + counts(i);
        elseif binLow <= threshold && binHigh > threshold
            % Conservative approximation:
            % count the whole crossing bin as above threshold.
            % This slightly overestimates expansion, consistent with over-mask preference.
            N_above = N_above + counts(i);
        end
    end
end