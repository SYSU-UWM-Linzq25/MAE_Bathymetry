function E001_SelectTiles_Overlap_FullRiver_ForRiver_V4(rivers, varargin)
% E001_SelectTiles_Overlap_FullRiver_ForRiver_V4
%
% Full-river / inference-style MAE tile extraction.
%
% Purpose:
%   This step is different from D001 training-tile extraction.
%   D001 uses HiddenMask / LossMask / bathy-valid based quality filters to
%   select high-quality training samples.  E001 only controls overlap /
%   spacing along the centerline, then extracts every in-range tile.  This is
%   for full-river recovery / moving-window tests.
%
% Main logic:
%   1) Read centerline points:
%      Center_line_points_1st/AutoClip_By_BathyMask/<river>/ESA_WorldCover_Width_proj_Clip.shp
%   2) Thin points deterministically by line_ID and target spacing.
%   3) For every spacing-thinned point, extract a fixed-size DEM tile if the
%      full tile is inside the raster extent.
%   4) No filtering by known_patch_ratio, HiddenMask, LossMask, core loss, or
%      bathy valid is applied.
%   5) Extract HiddenMask products for model input masking when available.
%   6) Extract pixel-level final LossMask = LossMask AND bathy-valid for
%      full-river recovery / output mosaicking.
%   7) Also output Core_Mask_Pixel and Core_Loss_Mask_Pixel. Downstream
%      stitching should average predictions where core-loss pixels from
%      different tile centers overlap.
%
% Required raster:
%   Processed_Results/Bathy3DEP_Merged_<res>m_FixND/<river>/Combined_BathyPriority_<res>m.vrt
%
% Optional/required mask rasters:
%   BahtyMask_Corage_Manual_Finial_Draw/HiddenMask_ByRiver_<res>m/<river>/HiddenMask_<res>m.vrt
%   LossMask_Draw/LossMask_ByRiver_<res>m/<river>/LossMask_<res>m.vrt
%   Processed_Results/Bathy_<res>m_FixND/<river>/Bathy_<res>m.vrt  (used to gate final loss mask by bathy-valid pixels)
%
% Output:
%   Processed_Results/Tiles_for_MAE_FullRiver_E001/Tiles_<res>m/...
%
% Example:
%   cd('/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z999_scripts/')
%   E001_SelectTiles_Overlap_FullRiver_ForRiver_V4('Kletzch_Combined_UpMax3Null');
%   E001_SelectTiles_Overlap_FullRiver_ForRiver_V4('ALL', 'resolution', 1);
%   E001_SelectTiles_Overlap_FullRiver_ForRiver_V4('ALL', 'resolution', 3);
%
% Notes:
%   - Tile size / patch size follow MAE: 336 and 16 by default.
%   - Default overlap logic is the same as D001: corePatchRadius=3,
%     targetCoreOverlap=0.50, so target spacing is 56 pixels.
%   - Out-of-range points are skipped to keep exact fixed-size tiles fully supported by raster data.
%   - HiddenMask output is patch-view 336x336, using ANY hidden pixel in each
%     16x16 patch. This is for model/inference input masking, not a sampling filter.
%   - LossMask output is 336x336 pixel-level final loss/recovery mask.
%   - Core_Loss_Mask_Pixel marks the part to use during mosaicking; overlapping
%     core predictions should be averaged by sum/count.

p = inputParser;
p.addRequired('rivers');
p.addParameter('processedRoot', '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', @ischar);
p.addParameter('hiddenRoot', '/tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw', @ischar);
p.addParameter('lossRoot', '/tank/data/SFS/xinyis/data/bathymetry/LossMask_Draw', @ischar);
p.addParameter('centerRoot', '/tank/data/SFS/xinyis/data/bathymetry/Center_line_points_1st/AutoClip_By_BathyMask', @ischar);
p.addParameter('outputRoot', '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_FullRiver_E001', @ischar);
p.addParameter('resolution', 1, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('tileSize', 336, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('patchSize', 16, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('corePatchRadius', 3, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('targetCoreOverlap', 0.50, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('targetSpacingPixels', [], @(x)isnumeric(x));
p.addParameter('writeHiddenMask', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writeHiddenPixelQA', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writeHiddenPatch21QA', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('requireHiddenRaster', false, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writeLossMask', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writeCoreMask', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writeCoreLossMask', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('requireLossRaster', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('useBathyValidForLossMask', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('nodataValue', -999999, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('invalidBelow', -9999, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('overwrite', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('maxTilesPerRiver', inf, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('skipWriteTiles', false, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('showProgress', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('progressEvery', 200, @(x)isnumeric(x)&&isscalar(x));
p.parse(rivers, varargin{:});
cfg = p.Results;

cfg.tileSize = round(cfg.tileSize);
cfg.patchSize = round(cfg.patchSize);
cfg.corePatchRadius = round(cfg.corePatchRadius);
cfg.progressEvery = max(1, round(cfg.progressEvery));

if mod(cfg.tileSize, cfg.patchSize) ~= 0
    error('tileSize must be divisible by patchSize. Got %d and %d.', cfg.tileSize, cfg.patchSize);
end
if cfg.targetCoreOverlap < 0 || cfg.targetCoreOverlap >= 1
    error('targetCoreOverlap must be in [0,1). Got %.3f.', cfg.targetCoreOverlap);
end

riverList = resolveRivers(cfg.rivers);
resStr = resolutionString(cfg.resolution);
patchGrid = cfg.tileSize / cfg.patchSize;
coreWidthPix = (2 * cfg.corePatchRadius + 1) * cfg.patchSize;
if isempty(cfg.targetSpacingPixels)
    targetSpacingPixels0 = max(1, round(coreWidthPix * (1 - cfg.targetCoreOverlap)));
else
    targetSpacingPixels0 = double(cfg.targetSpacingPixels);
end

fprintf('\n============================================================\n');
fprintf('E001 V4 Full-river MAE tiles with overlap only + pixel LossMask + core averaging masks [no Mapping Toolbox]\n');
fprintf('Processed root : %s\n', cfg.processedRoot);
fprintf('Center root    : %s\n', cfg.centerRoot);
fprintf('Output root    : %s\n', cfg.outputRoot);
fprintf('Rivers         : %s\n', strjoin(riverList, ', '));
fprintf('Resolution     : %s\n', resStr);
fprintf('Tile size      : %d x %d pixels\n', cfg.tileSize, cfg.tileSize);
fprintf('Patch size     : %d x %d pixels\n', cfg.patchSize, cfg.patchSize);
fprintf('Patch grid     : %d x %d\n', patchGrid, patchGrid);
fprintf('Core radius    : %d patches => %d x %d core patches\n', cfg.corePatchRadius, 2*cfg.corePatchRadius+1, 2*cfg.corePatchRadius+1);
fprintf('Target overlap : %.2f\n', cfg.targetCoreOverlap);
fprintf('Target spacing : %.2f pixels before map-unit conversion\n', targetSpacingPixels0);
fprintf('Filters        : spacing + full-tile in range ONLY\n');
fprintf('No known/loss/hidden filter is applied for tile selection.\n');
fprintf('Write Hidden   : %s, require hidden raster: %s\n', onOff(cfg.writeHiddenMask), onOff(cfg.requireHiddenRaster));
fprintf('Write Loss     : %s, require loss raster: %s, bathy-valid gating: %s\n', onOff(cfg.writeLossMask), onOff(cfg.requireLossRaster), onOff(cfg.useBathyValidForLossMask));
fprintf('Core masks     : core mask=%s, core-loss mask=%s; later mosaicking should average overlapping core-loss pixels.\n', onOff(cfg.writeCoreMask), onOff(cfg.writeCoreLossMask));
fprintf('============================================================\n');

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

for ir = 1:numel(riverList)
    river = riverList{ir};
    try
        processOneRiver(river, cfg);
    catch ME
        fprintf('\n[ERROR] %s\n%s\n', river, getReport(ME, 'extended', 'hyperlinks', 'off'));
        rethrow(ME);
    end
end

fprintf('\n============================================================\n');
fprintf('E001 finished.\n');
fprintf('============================================================\n');
end

function processOneRiver(river, cfg)
fprintf('\n============================================================\n');
fprintf('[E001] River: %s\n', river);
fprintf('============================================================\n');

PR = cfg.processedRoot;
resStr = resolutionString(cfg.resolution);
mergedVrt = fullfile(PR, sprintf('Bathy3DEP_Merged_%s_FixND', resStr), river, sprintf('Combined_BathyPriority_%s.vrt', resStr));
hiddenVrt = fullfile(cfg.hiddenRoot, sprintf('HiddenMask_ByRiver_%s', resStr), river, sprintf('HiddenMask_%s.vrt', resStr));
lossVrt   = fullfile(cfg.lossRoot,   sprintf('LossMask_ByRiver_%s', resStr),   river, sprintf('LossMask_%s.vrt', resStr));
bathyVrt  = fullfile(PR, sprintf('Bathy_%s_FixND', resStr), river, sprintf('Bathy_%s.vrt', resStr));
centerShp = fullfile(cfg.centerRoot, river, 'ESA_WorldCover_Width_proj_Clip.shp');

fprintf('[1/4] Check input files for %s resolution...\n', resStr);
mustExist(mergedVrt, sprintf('merged %s DEM', resStr));
mustExist(centerShp, 'centerline point shp');
hasHidden = exist(hiddenVrt, 'file') == 2;
hasLoss = exist(lossVrt, 'file') == 2;
hasBathy = exist(bathyVrt, 'file') == 2;
if cfg.writeHiddenMask && cfg.requireHiddenRaster && ~hasHidden
    error('Missing hidden mask raster: %s', hiddenVrt);
end
if cfg.writeHiddenMask && ~hasHidden
    warning('Hidden mask raster not found. DEM tiles will be extracted, hidden outputs skipped: %s', hiddenVrt);
end
if cfg.writeLossMask && cfg.requireLossRaster && ~hasLoss
    error('Missing loss mask raster: %s', lossVrt);
end
if cfg.writeLossMask && cfg.useBathyValidForLossMask && ~hasBathy
    error('Missing bathy raster needed for final loss mask: %s', bathyVrt);
end

fprintf('Merged DEM : %s\n', mergedVrt);
fprintf('HiddenMask : %s [%s]\n', hiddenVrt, ternary(hasHidden, 'found', 'missing/skipped'));
fprintf('LossMask   : %s [%s]\n', lossVrt, ternary(hasLoss, 'found', 'missing/skipped'));
fprintf('Bathy DEM  : %s [%s]\n', bathyVrt, ternary(hasBathy, 'found', 'missing/skipped'));
fprintf('Center pts : %s\n', centerShp);

fprintf('[2/4] Read raster metadata...\n');
[~, rowsM, colsM, geoTrans, proj, dataTypeDEM, ndMerged] = RasterInfo(mergedVrt);
if hasHidden
    [~, rowsH, colsH, ~, ~, ~, ~] = RasterInfo(hiddenVrt);
    if rowsM ~= rowsH || colsM ~= colsH
        error('Grid size mismatch for %s: merged=%d/%d hidden=%d/%d', river, rowsM, colsM, rowsH, colsH);
    end
end
if hasLoss
    [~, rowsL, colsL, ~, ~, ~, ~] = RasterInfo(lossVrt);
    if rowsM ~= rowsL || colsM ~= colsL
        error('Grid size mismatch for %s: merged=%d/%d loss=%d/%d', river, rowsM, colsM, rowsL, colsL);
    end
end
if hasBathy
    [~, rowsB, colsB, ~, ~, ~, ndBathy] = RasterInfo(bathyVrt);
    if rowsM ~= rowsB || colsM ~= colsB
        error('Grid size mismatch for %s: merged=%d/%d bathy=%d/%d', river, rowsM, colsM, rowsB, colsB);
    end
else
    ndBathy = cfg.nodataValue;
end

px = abs(geoTrans(2));
py = abs(geoTrans(6));
meanPix = mean([px py]);
coreWidthPix = (2 * cfg.corePatchRadius + 1) * cfg.patchSize;
if isempty(cfg.targetSpacingPixels)
    targetSpacingPixels = max(1, round(coreWidthPix * (1 - cfg.targetCoreOverlap)));
else
    targetSpacingPixels = double(cfg.targetSpacingPixels);
end
targetSpacingMap = targetSpacingPixels * meanPix;

fprintf('Raster rows/cols       : %d / %d\n', rowsM, colsM);
fprintf('Pixel size             : %.6f / %.6f\n', px, py);
fprintf('Core width             : %d pixels\n', coreWidthPix);
fprintf('Target spacing         : %.2f pixels = %.3f map units\n', targetSpacingPixels, targetSpacingMap);

outRoot = cfg.outputRoot;
tileRoot = fullfile(outRoot, sprintf('Tiles_%s', resStr));
folders.Train           = fullfile(tileRoot, 'FullRiver_tile');
folders.HiddenPatchView = fullfile(tileRoot, 'Hidden_Mask');
folders.HiddenPixelQA   = fullfile(tileRoot, 'Hidden_Mask_Pixel_QA');
folders.HiddenPatch21   = fullfile(tileRoot, 'Hidden_Mask_Patch21_QA');
folders.LossPixel       = fullfile(tileRoot, 'Loss_Mask_Pixel');
folders.CoreMask        = fullfile(tileRoot, 'Core_Mask_Pixel');
folders.CoreLossMask    = fullfile(tileRoot, 'Core_Loss_Mask_Pixel');
folders.QA              = fullfile(outRoot, 'QA', river);
folders.Lists           = fullfile(outRoot, 'Lists');
makeFolders(folders);

fprintf('[3/4] Read and thin centerline points...\n');
CT = readCenterPointsWithOGR(centerShp, fullfile(folders.QA, '_tmp_centerline_csv'));
if isempty(CT) || height(CT) == 0
    error('Centerline shapefile has no points: %s', centerShp);
end
[X, Y] = extractXYFromTable(CT);
lineID = extractNumericTableField(CT, {'line_ID','LineID','LINEID','Line_ID','lineid'}, (1:height(CT)).');
width  = extractNumericTableField(CT, {'Width','WIDTH','width','wid','WID'}, nan(height(CT),1));

keepSpacing = thinByLineSpacing(X, Y, lineID, targetSpacingMap);
idxSpacing = find(keepSpacing);
fprintf('Raw center points      : %d\n', height(CT));
fprintf('After spacing thinning : %d\n', numel(idxSpacing));

fprintf('[4/4] Extract every in-range tile; no mask/quality filters...\n');
tLoop = tic;
qaRows = [];
manifestRows = [];
selectedStruct = [];
candidateStruct = [];
selectedCount = 0;
checkedCount = 0;

hr = floor(cfg.tileSize / 2);
hc = floor(cfg.tileSize / 2);
patchGrid = cfg.tileSize / cfg.patchSize;
coreMask = makeCoreMask(cfg.tileSize, cfg.patchSize, cfg.corePatchRadius);
corePixelCount = nnz(coreMask);

for ii = 1:numel(idxSpacing)
    srcIdx = idxSpacing(ii);
    checkedCount = checkedCount + 1;
    x = X(srcIdx);
    y = Y(srcIdx);
    [row0, col0] = Proj2RowCol(geoTrans, y, x);
    r1 = row0 - hr;
    c1 = col0 - hc;
    h = cfg.tileSize;
    w = cfg.tileSize;

    reject = "";
    if r1 < 1 || c1 < 1 || (r1 + h - 1) > rowsM || (c1 + w - 1) > colsM
        reject = "out_of_range";
        qaRows = appendQARow(qaRows, srcIdx, NaN, x, y, lineID(srcIdx), width(srcIdx), row0, col0, false, reject, NaN, NaN, NaN, NaN, NaN);
        candidateStruct = appendCandidatePointStruct(candidateStruct, srcIdx, NaN, x, y, lineID(srcIdx), width(srcIdx), row0, col0, false, reject, NaN, NaN, NaN, NaN, NaN);
        if cfg.showProgress && (mod(checkedCount, cfg.progressEvery) == 0 || checkedCount == numel(idxSpacing))
            printProgress(river, checkedCount, numel(idxSpacing), selectedCount, tLoop, reject);
        end
        continue;
    end

    tileDEM = ReadRaster(mergedVrt, r1, c1, h, w);
    hasHiddenTile = false;
    hiddenPixelRatio = NaN;
    hiddenPatchCount = NaN;
    hiddenPatchRatio = NaN;
    hiddenPatch = [];
    tileHidden = [];
    if cfg.writeHiddenMask && hasHidden
        tileHidden = ReadRaster(hiddenVrt, r1, c1, h, w);
        hiddenPixel = (tileHidden > 0) & (tileHidden < 255) & isfinite(tileHidden);
        hiddenPatch = blockAny(hiddenPixel, cfg.patchSize);
        hiddenPixelRatio = mean(hiddenPixel(:));
        hiddenPatchCount = nnz(hiddenPatch);
        hiddenPatchRatio = hiddenPatchCount / max(1, patchGrid * patchGrid);
        hasHiddenTile = true;
    end

    hasLossTile = false;
    lossPixelRatio = NaN;
    coreLossPixelRatio = NaN;
    finalLossPixel = [];
    coreLossPixel = [];
    if cfg.writeLossMask && hasLoss
        tileLoss = ReadRaster(lossVrt, r1, c1, h, w);
        lossCandidate = (tileLoss > 0) & (tileLoss < 255) & isfinite(tileLoss);
        if cfg.useBathyValidForLossMask
            tileBathy = ReadRaster(bathyVrt, r1, c1, h, w);
            bathyValid = isfinite(tileBathy) & (tileBathy ~= cfg.nodataValue) & (tileBathy > cfg.invalidBelow);
            if isfinite(double(ndBathy))
                bathyValid = bathyValid & (tileBathy ~= double(ndBathy));
            end
            finalLossPixel = lossCandidate & bathyValid;
        else
            finalLossPixel = lossCandidate;
        end
        coreLossPixel = finalLossPixel & coreMask;
        lossPixelRatio = mean(finalLossPixel(:));
        coreLossPixelRatio = nnz(coreLossPixel) / max(1, corePixelCount);
        hasLossTile = true;
    end

    reject = "kept";
    selectedCount = selectedCount + 1;
    pointID = selectedCount;

    if selectedCount <= cfg.maxTilesPerRiver
        if ~cfg.skipWriteTiles
            subGT = subTranscoef(geoTrans, r1, c1);
            patchGT = subGT;
            patchGT(2) = patchGT(2) * cfg.patchSize;
            patchGT(6) = patchGT(6) * cfg.patchSize;

            demOut = fullfile(folders.Train, sprintf('E001_FullRiver_tile_%s_%s_ID%d.tif', resStr, river, pointID));
            hidViewOut = '';
            hidPixOut = '';
            hid21Out = '';
            lossOut = '';
            coreOut = '';
            coreLossOut = '';

            writeMaybe(demOut, tileDEM, subGT, proj, dataTypeDEM, 'GTiff', resolveNoData(ndMerged, cfg.nodataValue), cfg.overwrite);

            if hasHiddenTile
                hidViewOut = fullfile(folders.HiddenPatchView, sprintf('E001_tile_%s_%s_ID%d_HiddenMask.tif', resStr, river, pointID));
                hidPixOut  = fullfile(folders.HiddenPixelQA, sprintf('E001_tile_%s_%s_ID%d_HiddenMaskPixel_QA.tif', resStr, river, pointID));
                hid21Out   = fullfile(folders.HiddenPatch21, sprintf('E001_tile_%s_%s_ID%d_HiddenMaskPatch21_QA.tif', resStr, river, pointID));
                hidView = expandPatchMask(hiddenPatch, cfg.patchSize);
                writeMaybe(hidViewOut, hidView, subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                if cfg.writeHiddenPixelQA
                    hiddenPixel = (tileHidden > 0) & (tileHidden < 255) & isfinite(tileHidden);
                    writeMaybe(hidPixOut, double(hiddenPixel), subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                else
                    hidPixOut = '';
                end
                if cfg.writeHiddenPatch21QA
                    writeMaybe(hid21Out, double(hiddenPatch), patchGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                else
                    hid21Out = '';
                end
            end

            if hasLossTile
                lossOut = fullfile(folders.LossPixel, sprintf('E001_tile_%s_%s_ID%d_LossMaskPixel.tif', resStr, river, pointID));
                writeMaybe(lossOut, double(finalLossPixel), subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                if cfg.writeCoreMask
                    coreOut = fullfile(folders.CoreMask, sprintf('E001_tile_%s_%s_ID%d_CoreMaskPixel.tif', resStr, river, pointID));
                    writeMaybe(coreOut, double(coreMask), subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                end
                if cfg.writeCoreLossMask
                    coreLossOut = fullfile(folders.CoreLossMask, sprintf('E001_tile_%s_%s_ID%d_CoreLossMaskPixel.tif', resStr, river, pointID));
                    writeMaybe(coreLossOut, double(coreLossPixel), subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                end
            end

            manifestRows = appendManifestRow(manifestRows, pointID, river, demOut, hidViewOut, hidPixOut, hid21Out, lossOut, coreOut, coreLossOut);
        end

        selectedStruct = appendPointStruct(selectedStruct, pointID, srcIdx, x, y, lineID(srcIdx), width(srcIdx), row0, col0, hiddenPixelRatio, hiddenPatchCount, hiddenPatchRatio, lossPixelRatio, coreLossPixelRatio);
    end

    qaRows = appendQARow(qaRows, srcIdx, pointID, x, y, lineID(srcIdx), width(srcIdx), row0, col0, true, reject, hiddenPixelRatio, hiddenPatchCount, hiddenPatchRatio, lossPixelRatio, coreLossPixelRatio);
    candidateStruct = appendCandidatePointStruct(candidateStruct, srcIdx, pointID, x, y, lineID(srcIdx), width(srcIdx), row0, col0, true, reject, hiddenPixelRatio, hiddenPatchCount, hiddenPatchRatio, lossPixelRatio, coreLossPixelRatio);

    if cfg.showProgress && (mod(checkedCount, cfg.progressEvery) == 0 || checkedCount == numel(idxSpacing))
        printProgress(river, checkedCount, numel(idxSpacing), selectedCount, tLoop, reject);
    end

    if selectedCount >= cfg.maxTilesPerRiver
        fprintf('Reached maxTilesPerRiver=%d, stop early for %s.\n', cfg.maxTilesPerRiver, river);
        break;
    end
end

fprintf('Checked spacing-thinned points: %d\n', checkedCount);
fprintf('Selected full-river tiles      : %d\n', selectedCount);

qaTable = qaRowsToTable(qaRows);
qaCsv = fullfile(folders.QA, sprintf('E001_candidate_QA_%s_%s.csv', resStr, river));
writetable(qaTable, qaCsv);
fprintf('QA CSV: %s\n', qaCsv);

if ~isempty(candidateStruct)
    outCandPts = fullfile(folders.QA, sprintf('E001_candidate_points_QA_%s_%s.shp', resStr, river));
    candTable = pointStructToTable(candidateStruct);
    writePointShpWithOGR(candTable, outCandPts, centerShp, fullfile(folders.QA, '_tmp_candidate_points_csvshp'));
    fprintf('Candidate QA point shp: %s\n', outCandPts);
else
    warning('No candidate QA points for %s.', river);
end

if ~isempty(selectedStruct)
    outPts = fullfile(folders.QA, sprintf('E001_selected_points_%s_%s.shp', resStr, river));
    selTable = pointStructToTable(selectedStruct);
    writePointShpWithOGR(selTable, outPts, centerShp, fullfile(folders.QA, '_tmp_selected_points_csvshp'));
    fprintf('Selected point shp: %s\n', outPts);
else
    warning('No selected points for %s.', river);
end

if ~isempty(manifestRows)
    manifestTable = manifestRowsToTable(manifestRows);
    manifestCsv = fullfile(folders.QA, sprintf('E001_tile_manifest_%s_%s.csv', resStr, river));
writetable(manifestTable, manifestCsv);
    fprintf('Manifest CSV: %s\n', manifestCsv);

    writeList(fullfile(folders.Lists, sprintf('E001_fullriver_tiles_%s_%s.txt', resStr, river)), manifestTable.dem_path);
    if ismember('hidden_mask_path', manifestTable.Properties.VariableNames)
        writeList(fullfile(folders.Lists, sprintf('E001_hidden_masks_336patchview_%s_%s.txt', resStr, river)), manifestTable.hidden_mask_path);
    end
    if ismember('loss_mask_pixel_path', manifestTable.Properties.VariableNames)
        writeList(fullfile(folders.Lists, sprintf('E001_loss_masks_pixel_%s_%s.txt', resStr, river)), manifestTable.loss_mask_pixel_path);
    end
    if ismember('core_loss_mask_pixel_path', manifestTable.Properties.VariableNames)
        writeList(fullfile(folders.Lists, sprintf('E001_core_loss_masks_pixel_%s_%s.txt', resStr, river)), manifestTable.core_loss_mask_pixel_path);
    end
end
end

function s = onOff(tf)
if logical(tf), s = 'ON'; else, s = 'OFF'; end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function resStr = resolutionString(res)
if abs(res - round(res)) < 1e-9
    resStr = sprintf('%dm', round(res));
else
    resStr = sprintf('%gm', res);
    resStr = strrep(resStr, '.', 'p');
end
end

function printProgress(river, checked, total, kept, tLoop, lastReject)
elapsed = toc(tLoop);
rate = checked / max(elapsed, eps);
remain = max(0, total - checked);
etaMin = remain / max(rate, eps) / 60;
fprintf('  [%s] checked %d/%d (%.1f%%), kept=%d, rate=%.2f pt/s, ETA=%.1f min, last=%s\n', ...
    river, checked, total, 100*checked/max(1,total), kept, rate, etaMin, char(lastReject));
end

function rivers = resolveRivers(riversIn)
valid = { ...
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
if ischar(riversIn) || isstring(riversIn)
    s = char(riversIn);
    if strcmpi(s, 'ALL')
        rivers = valid;
    elseif strcmpi(s, 'LIST')
        fprintf('Valid rivers:\n');
        for i = 1:numel(valid), fprintf('  %s\n', valid{i}); end
        rivers = {};
    else
        rivers = {s};
    end
elseif iscell(riversIn)
    rivers = riversIn;
else
    error('rivers must be char/string/cell array.');
end
if isempty(rivers), return; end
for i = 1:numel(rivers)
    if ~ismember(rivers{i}, valid)
        error('Unknown river: %s', rivers{i});
    end
end
end

function mustExist(pth, label)
if exist(pth, 'file') ~= 2
    error('Missing %s: %s', label, pth);
end
end

function makeFolders(folders)
fn = fieldnames(folders);
for i = 1:numel(fn)
    if exist(folders.(fn{i}), 'dir') ~= 7
        mkdir(folders.(fn{i}));
    end
end
end

function GT = readCenterShpSafe(centerShp, tmpDir)
try
    GT = shaperead(centerShp);
catch ME
    msg = ME.message;
    if contains(msg, 'PointZ') || contains(msg, 'type code = 11') || contains(msg, 'Unsupported shape type')
        fprintf('  Center shp not readable by shaperead (%s). Convert to 2D Point with ogr2ogr...\n', msg);
        if exist(tmpDir, 'dir') == 7
            rmdir(tmpDir, 's');
        end
        mkdir(tmpDir);
        [~, base, ~] = fileparts(centerShp);
        tmpShp = fullfile(tmpDir, [base '_XY.shp']);
        cmd = sprintf('ogr2ogr -overwrite -dim XY -nlt POINT "%s" "%s"', tmpShp, centerShp);
        [status, out] = system(cmd);
        if status ~= 0
            error('ogr2ogr PointZ->Point conversion failed:\n%s\nCommand:\n%s', out, cmd);
        end
        GT = shaperead(tmpShp);
    else
        rethrow(ME);
    end
end
end

function [X, Y] = extractPointXY(GT)
n = numel(GT);
X = nan(n,1);
Y = nan(n,1);
for i = 1:n
    xi = GT(i).X; yi = GT(i).Y;
    xi = xi(~isnan(xi)); yi = yi(~isnan(yi));
    if isempty(xi) || isempty(yi)
        error('Point %d has empty X/Y.', i);
    end
    X(i) = xi(1);
    Y(i) = yi(1);
end
end

function vals = extractNumericField(GT, candidates, defaultVals)
n = numel(GT);
vals = defaultVals(:);
fields = fieldnames(GT);
chosen = '';
for c = 1:numel(candidates)
    k = find(strcmpi(fields, candidates{c}), 1);
    if ~isempty(k)
        chosen = fields{k};
        break;
    end
end
if isempty(chosen), return; end
for i = 1:n
    v = GT(i).(chosen);
    if isnumeric(v)
        vals(i) = double(v(1));
    elseif ischar(v) || isstring(v)
        vv = str2double(v);
        if isfinite(vv), vals(i) = vv; end
    else
        vals(i) = defaultVals(i);
    end
end
end

function keep = thinByLineSpacing(X, Y, lineID, spacing)
n = numel(X);
keep = false(n,1);
lineStr = string(lineID);
u = unique(lineStr, 'stable');
for k = 1:numel(u)
    idx = find(lineStr == u(k));
    if isempty(idx), continue; end
    keep(idx(1)) = true;
    prev = idx(1);
    accum = 0;
    for j = 2:numel(idx)
        cur = idx(j);
        d = hypot(X(cur) - X(prev), Y(cur) - Y(prev));
        accum = accum + d;
        if accum >= spacing
            keep(cur) = true;
            accum = 0;
        end
        prev = cur;
    end
end
end


function T = readCenterPointsWithOGR(centerShp, tmpDir)
% Read point/PointZ shapefile without MATLAB Mapping Toolbox.
% Uses ogr2ogr CSV driver with GEOMETRY=AS_XY, so PointZ is flattened to X/Y.
%
% Important GDAL behavior:
%   For -f CSV, the output directory must NOT be pre-created. The CSV
%   driver creates it. If the directory already exists, some GDAL builds
%   fail with "file system object ... already exists". Therefore we remove
%   any old temp folder and pass a non-existing csvOutDir to ogr2ogr.
if exist(tmpDir, 'dir') == 7
    rmdir(tmpDir, 's');
end
mkdir(tmpDir);

csvOutDir = fullfile(tmpDir, 'csv_out');
if exist(csvOutDir, 'dir') == 7
    rmdir(csvOutDir, 's');
elseif exist(csvOutDir, 'file') == 2
    delete(csvOutDir);
end

layerName = 'center_points';
cmd = sprintf('ogr2ogr -overwrite -f CSV -lco GEOMETRY=AS_XY "%s" "%s" -nln %s', csvOutDir, centerShp, layerName);
[status, out] = system(cmd);
if status ~= 0
    error('ogr2ogr shapefile->CSV failed:\n%s\nCommand:\n%s', out, cmd);
end
csvPath = fullfile(csvOutDir, [layerName '.csv']);
if exist(csvPath, 'file') ~= 2
    % Some GDAL versions keep the original layer name. Fall back to the first CSV.
    d = dir(fullfile(csvOutDir, '*.csv'));
    if isempty(d)
        error('ogr2ogr produced no CSV in: %s', csvOutDir);
    end
    csvPath = fullfile(csvOutDir, d(1).name);
end
T = readtable(csvPath, 'VariableNamingRule', 'preserve');
fprintf('Center shp read mode  : ogr2ogr CSV, no Mapping Toolbox license required\n');
end
function [X, Y] = extractXYFromTable(T)
vars = T.Properties.VariableNames;
ix = find(strcmpi(vars, 'X'), 1);
iy = find(strcmpi(vars, 'Y'), 1);
if isempty(ix) || isempty(iy)
    error('Cannot find X/Y columns in ogr2ogr CSV. Available columns: %s', strjoin(vars, ', '));
end
X = tableColumnToDouble(T.(vars{ix}));
Y = tableColumnToDouble(T.(vars{iy}));
if any(~isfinite(X)) || any(~isfinite(Y))
    error('Some centerline point X/Y values are not finite.');
end
end

function vals = extractNumericTableField(T, candidates, defaultVals)
vars = T.Properties.VariableNames;
vals = defaultVals(:);
chosen = '';
for c = 1:numel(candidates)
    k = find(strcmpi(vars, candidates{c}), 1);
    if ~isempty(k)
        chosen = vars{k};
        break;
    end
end
if isempty(chosen), return; end
v = tableColumnToDouble(T.(chosen));
if numel(v) == numel(vals)
    good = isfinite(v);
    vals(good) = v(good);
end
end

function v = tableColumnToDouble(col)
if isnumeric(col)
    v = double(col);
elseif iscell(col)
    v = nan(numel(col),1);
    for i = 1:numel(col)
        if isnumeric(col{i})
            v(i) = double(col{i}(1));
        else
            v(i) = str2double(string(col{i}));
        end
    end
elseif isstring(col) || ischar(col)
    v = str2double(string(col));
else
    try
        v = double(col);
    catch
        v = str2double(string(col));
    end
end
v = v(:);
end

function T = pointStructToTable(S)
T = struct2table(S);
% Remove Mapping Toolbox geometry field. OGR will build geometry from X/Y.
if ismember('Geometry', T.Properties.VariableNames)
    T.Geometry = [];
end
% Keep field names short for Shapefile compatibility.
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);
end

function writePointShpWithOGR(T, outShp, srcShp, tmpDir)
% Write point shapefile without MATLAB Mapping Toolbox, via CSV + ogr2ogr.
if isempty(T) || height(T) == 0
    return;
end
if exist(tmpDir, 'dir') == 7
    rmdir(tmpDir, 's');
end
mkdir(tmpDir);
[dstDir, dstBase, ~] = fileparts(outShp);
if exist(dstDir, 'dir') ~= 7
    mkdir(dstDir);
end
csvPath = fullfile(tmpDir, [dstBase '.csv']);
writetable(T, csvPath);

% Remove existing shapefile component files first.
parts = {'.shp','.shx','.dbf','.prj','.cpg','.sbn','.sbx','.qix','.fix'};
for i = 1:numel(parts)
    f = fullfile(dstDir, [dstBase parts{i}]);
    if exist(f, 'file') == 2, delete(f); end
end

shpTmpDir = fullfile(tmpDir, 'shp');
mkdir(shpTmpDir);
cmd = sprintf(['ogr2ogr -overwrite -f "ESRI Shapefile" "%s" "%s" ', ...
               '-nln "%s" -oo X_POSSIBLE_NAMES=X -oo Y_POSSIBLE_NAMES=Y -oo KEEP_GEOM_COLUMNS=YES'], ...
               shpTmpDir, csvPath, dstBase);
[status, out] = system(cmd);
if status ~= 0
    error('ogr2ogr CSV->Shapefile failed:\n%s\nCommand:\n%s', out, cmd);
end

created = dir(fullfile(shpTmpDir, [dstBase '.*']));
if isempty(created)
    % GDAL may have sanitized layer name; copy first shapefile layer if needed.
    created = dir(fullfile(shpTmpDir, '*.*'));
end
for i = 1:numel(created)
    if ~created(i).isdir
        copyfile(fullfile(shpTmpDir, created(i).name), fullfile(dstDir, created(i).name), 'f');
    end
end
copyPrj(srcShp, outShp);
end

function C = makeCoreMask(tileSize, patchSize, corePatchRadius)
patchGrid = tileSize / patchSize;
midPatch = ceil(patchGrid / 2);
patchIdx = (midPatch - corePatchRadius):(midPatch + corePatchRadius);
r1 = (patchIdx(1)-1) * patchSize + 1;
r2 = patchIdx(end) * patchSize;
C = false(tileSize, tileSize);
C(r1:r2, r1:r2) = true;
end

function P = blockAny(mask, p)
[H,W] = size(mask);
gh = H / p; gw = W / p;
P = false(gh, gw);
for rr = 1:gh
    r0 = (rr-1)*p + 1;
    for cc = 1:gw
        c0 = (cc-1)*p + 1;
        b = mask(r0:r0+p-1, c0:c0+p-1);
        P(rr,cc) = any(b(:));
    end
end
end

function out = expandPatchMask(P, p)
out = double(kron(double(P), ones(p,p)));
end

function nd = resolveNoData(nd0, fallback)
if isempty(nd0) || ~isfinite(double(nd0))
    nd = fallback;
else
    nd = double(nd0);
end
end

function writeMaybe(outPath, A, geoTrans, proj, dataType, outFormat, nodata, overwrite)
if exist(outPath, 'file') == 2
    if ~overwrite
        return;
    else
        delete(outPath);
        auxPath = [outPath '.aux.xml'];
        if exist(auxPath, 'file') == 2, delete(auxPath); end
    end
end
if exist(fileparts(outPath), 'dir') ~= 7
    mkdir(fileparts(outPath));
end
if dataType == 1
    A = double(A);
    A(~isfinite(A)) = nodata;
end
WriteRaster(outPath, A, geoTrans, proj, dataType, outFormat, nodata);
end

function rows = appendQARow(rows, srcIdx, pointID, x, y, lineID, width, row0, col0, kept, reject, hiddenPixRatio, hiddenPatchCount, hiddenPatchRatio, lossPixelRatio, coreLossPixelRatio)
r.src_idx = srcIdx;
r.point_id = pointID;
r.x = x;
r.y = y;
r.line_id = lineID;
r.width = width;
r.row0 = row0;
r.col0 = col0;
r.kept = double(kept);
r.reject = char(reject);
r.hidden_pixel_ratio = hiddenPixRatio;
r.hidden_patch_count = hiddenPatchCount;
r.hidden_patch_ratio = hiddenPatchRatio;
r.loss_pixel_ratio = lossPixelRatio;
r.core_loss_pixel_ratio = coreLossPixelRatio;
if isempty(rows), rows = r; else, rows(end+1,1) = r; end
end

function T = qaRowsToTable(rows)
if isempty(rows), T = table(); else, T = struct2table(rows); end
end

function rows = appendManifestRow(rows, pointID, river, demPath, hiddenViewPath, hiddenPixelPath, hidden21Path, lossPixelPath, coreMaskPath, coreLossMaskPath)
r.point_id = pointID;
r.river = river;
r.dem_path = demPath;
r.hidden_mask_path = hiddenViewPath;
r.hidden_pixel_QA_path = hiddenPixelPath;
r.hidden_patch21_QA_path = hidden21Path;
r.loss_mask_pixel_path = lossPixelPath;
r.core_mask_pixel_path = coreMaskPath;
r.core_loss_mask_pixel_path = coreLossMaskPath;
if isempty(rows), rows = r; else, rows(end+1,1) = r; end
end

function T = manifestRowsToTable(rows)
if isempty(rows), T = table(); else, T = struct2table(rows); end
end

function S = appendCandidatePointStruct(S, srcIdx, pointID, x, y, lineID, width, row0, col0, kept, reject, hiddenPixRatio, hiddenPatchCount, hiddenPatchRatio, lossPixelRatio, coreLossPixelRatio)
r.Geometry = 'Point';
r.X = x; r.Y = y;
r.SrcID = safeInt(srcIdx, 0);
r.PointID = safeInt(pointID, 0);
r.Kept = double(kept);
r.Reject = char(reject);
r.RjCode = rejectCode(reject);
r.LineID = safeNum(lineID, -9999);
r.Width = safeNum(width, -9999);
r.Row0 = safeNum(row0, -9999);
r.Col0 = safeNum(col0, -9999);
r.HPixR = safeNum(hiddenPixRatio, -9999);
r.HidPN = safeNum(hiddenPatchCount, -9999);
r.HidPR = safeNum(hiddenPatchRatio, -9999);
r.LPixR = safeNum(lossPixelRatio, -9999);
r.CLossPixR = safeNum(coreLossPixelRatio, -9999);
if isempty(S), S = r; else, S(end+1,1) = r; end
end

function S = appendPointStruct(S, pointID, srcIdx, x, y, lineID, width, row0, col0, hiddenPixRatio, hiddenPatchCount, hiddenPatchRatio, lossPixelRatio, coreLossPixelRatio)
r.Geometry = 'Point';
r.X = x; r.Y = y;
r.PointID = pointID;
r.SrcID = srcIdx;
r.LineID = lineID;
r.Width = width;
r.Row0 = row0;
r.Col0 = col0;
r.HPixR = safeNum(hiddenPixRatio, -9999);
r.HidPN = safeNum(hiddenPatchCount, -9999);
r.HidPR = safeNum(hiddenPatchRatio, -9999);
r.LPixR = safeNum(lossPixelRatio, -9999);
r.CLossPixR = safeNum(coreLossPixelRatio, -9999);
if isempty(S), S = r; else, S(end+1,1) = r; end
end

function v = safeNum(x, fillv)
if isempty(x) || ~isnumeric(x) || ~isfinite(double(x(1)))
    v = fillv;
else
    v = double(x(1));
end
end

function v = safeInt(x, fillv)
v = safeNum(x, fillv);
if isfinite(v), v = round(v); end
end

function c = rejectCode(reject)
s = char(reject);
switch s
    case 'kept'
        c = 0;
    case 'out_of_range'
        c = 1;
    otherwise
        c = 99;
end
end

function copyPrj(srcShp, dstShp)
[srcDir, srcBase, ~] = fileparts(srcShp);
[dstDir, dstBase, ~] = fileparts(dstShp);
srcPrj = fullfile(srcDir, [srcBase '.prj']);
dstPrj = fullfile(dstDir, [dstBase '.prj']);
if exist(srcPrj, 'file') == 2
    copyfile(srcPrj, dstPrj, 'f');
end
end

function writeList(path, vals)
if istable(vals), vals = table2array(vals); end
fid = fopen(path, 'w');
if fid < 0, error('Cannot open list file: %s', path); end
if iscell(vals)
    for i = 1:numel(vals), fprintf(fid, '%s\n', vals{i}); end
elseif isstring(vals)
    for i = 1:numel(vals), fprintf(fid, '%s\n', vals(i)); end
else
    for i = 1:numel(vals), fprintf(fid, '%s\n', vals(i)); end
end
fclose(fid);
end
