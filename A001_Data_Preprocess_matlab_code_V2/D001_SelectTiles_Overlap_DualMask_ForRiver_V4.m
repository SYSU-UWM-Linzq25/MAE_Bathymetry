function D001_SelectTiles_Overlap_DualMask_ForRiver_V4(rivers, varargin)
% D001_SelectTiles_Overlap_DualMask_ForRiver_V4
%
% New MAE tile sampling / extraction step.
%
% Main ideas:
%   1) Do NOT randomly keep 40% centerline points.
%   2) Thin centerline points deterministically by a target spacing derived
%      from desired core-patch overlap.
%   3) Extract 336 x 336 merged DEM tiles plus two pixel-level masks:
%        - HiddenMask: model must not see these pixels. For the model this
%          is exported as patch-level mask: hidden_patch = ANY hidden pixel
%          in a 16 x 16 patch.
%        - LossMask  : final training-loss pixels = manual LossMask AND bathy valid.
%          This is kept as pixel-level mask for future pixel-level loss.
%   4) Also explicitly convert loss pixels to patch-level QA masks using
%      MAE patch_size=16:
%        - loss_patch = ALL pixels are final loss pixels in a 16 x 16 patch.
%          This is only a QA / inspection product, not necessarily the final
%          training loss unit.
%   5) Save patch-view rasters so GIS inspection can compare pixel-level
%      masks with the MAE patch/token-level hidden mask.
%
% Input centerline points:
%   /tank/data/SFS/xinyis/data/bathymetry/Center_line_points_1st/AutoClip_By_BathyMask/<river>/ESA_WorldCover_Width_proj_Clip.shp
%
% Required rasters:
%   Processed_Results/Bathy3DEP_Merged_1m_FixND/<river>/Combined_BathyPriority_1m.vrt
%   Processed_Results/Bathy_1m_FixND/<river>/Bathy_1m.vrt
%   BahtyMask_Corage_Manual_Finial_Draw/HiddenMask_ByRiver_1m/<river>/HiddenMask_1m.vrt
%   LossMask_Draw/LossMask_ByRiver_1m/<river>/LossMask_1m.vrt
%
% Output:
%   Processed_Results/Tiles_for_MAE_v2/Tiles_1m/...
%
% Example:
%   cd('/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z999_scripts/')
%   D001_SelectTiles_Overlap_DualMask_ForRiver_V4('Kletzch_Combined_UpMax3Null');
%   D001_SelectTiles_Overlap_DualMask_ForRiver_V4('ALL');
%
% Useful options:
%   'targetCoreOverlap', 0.50
%   'targetSpacingPixels', []       % default computed from core width
%   'minCoreLossPixelCount', 64       % primary filter for pixel-level loss
%   'minCoreLossPatchCount', 0        % optional QA/filter for all-pixel loss patches
%   'corePatchRadius', 3            % MAE code default: 7 x 7 core patches
%   'tileSize', 336
%   'patchSize', 16
%   'maxTilesPerRiver', inf
%
% Notes:
%   - 0 is valid elevation, NOT NoData.
%   - NoData is detected by nodata value and very negative threshold.
%   - Loss_Mask output is the final loss pixel mask, already combined with
%     bathy valid pixels.

p = inputParser;
p.addRequired('rivers');
p.addParameter('processedRoot', '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', @ischar);
p.addParameter('hiddenRoot', '/tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw', @ischar);
p.addParameter('lossRoot', '/tank/data/SFS/xinyis/data/bathymetry/LossMask_Draw', @ischar);
p.addParameter('centerRoot', '/tank/data/SFS/xinyis/data/bathymetry/Center_line_points_1st/AutoClip_By_BathyMask', @ischar);
p.addParameter('outputRoot', '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2', @ischar);
p.addParameter('tileSize', 336, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('patchSize', 16, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('corePatchRadius', 3, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('targetCoreOverlap', 0.50, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('targetSpacingPixels', [], @(x)isnumeric(x));
p.addParameter('minCoreLossPatchCount', 0, @(x)isnumeric(x)&&isscalar(x));  % QA/filter only; pixel-level loss is primary
p.addParameter('minCoreLossPatchRatio', 0.0, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('minCoreLossPixelCount', 64, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('minCoreLossPixelRatio', 0.0, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('minVisiblePatchCount', 1, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('requireCenterLoss', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('nodataValue', -999999, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('invalidBelow', -9999, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('overwrite', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writeHiddenPixelQA', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writePatchView', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('writePatch21', true, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('maxTilesPerRiver', inf, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('skipWriteTiles', false, @(x)islogical(x)||ismember(x,[0 1]));
p.parse(rivers, varargin{:});
cfg = p.Results;

cfg.tileSize = round(cfg.tileSize);
cfg.patchSize = round(cfg.patchSize);
cfg.corePatchRadius = round(cfg.corePatchRadius);
cfg.minCoreLossPatchCount = round(cfg.minCoreLossPatchCount);
cfg.minVisiblePatchCount = round(cfg.minVisiblePatchCount);

if mod(cfg.tileSize, cfg.patchSize) ~= 0
    error('tileSize must be divisible by patchSize. Got %d and %d.', cfg.tileSize, cfg.patchSize);
end
if cfg.targetCoreOverlap < 0 || cfg.targetCoreOverlap >= 1
    error('targetCoreOverlap must be in [0,1). Got %.3f.', cfg.targetCoreOverlap);
end

riverList = resolveRivers(cfg.rivers);

fprintf('\n============================================================\n');
fprintf('D001 V4 Select MAE tiles with overlap + dual masks\n');
fprintf('Processed root : %s\n', cfg.processedRoot);
fprintf('Center root    : %s\n', cfg.centerRoot);
fprintf('Output root    : %s\n', cfg.outputRoot);
fprintf('Rivers         : %s\n', strjoin(riverList, ', '));
fprintf('Tile size      : %d x %d pixels\n', cfg.tileSize, cfg.tileSize);
fprintf('Patch size     : %d x %d pixels\n', cfg.patchSize, cfg.patchSize);
fprintf('Patch grid     : %d x %d\n', cfg.tileSize/cfg.patchSize, cfg.tileSize/cfg.patchSize);
fprintf('Core radius    : %d patches => %d x %d core patches\n', cfg.corePatchRadius, 2*cfg.corePatchRadius+1, 2*cfg.corePatchRadius+1);
fprintf('Target overlap : %.2f\n', cfg.targetCoreOverlap);
fprintf('Hidden output  : 336x336 patch-view mask for model, ANY hidden pixel in patch\n');
fprintf('Loss output    : 336x336 pixel-level final-loss mask for model loss\n');
fprintf('Loss patch QA  : ALL final-loss pixels in patch, saved for inspection\n');
fprintf('Core loss pixel min: count >= %d, ratio >= %.4f\n', cfg.minCoreLossPixelCount, cfg.minCoreLossPixelRatio);
fprintf('Core loss patch min: count >= %d, ratio >= %.4f (QA/filter only)\n', cfg.minCoreLossPatchCount, cfg.minCoreLossPatchRatio);
fprintf('NoData         : %g, invalidBelow=%g\n', cfg.nodataValue, cfg.invalidBelow);
fprintf('Zero as NoData : NO\n');
fprintf('============================================================\n');

% Load GDAL MEX utilities. This code follows the existing preprocessing pipeline.
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
fprintf('D001 finished.\n');
fprintf('============================================================\n');
end

function processOneRiver(river, cfg)
fprintf('\n============================================================\n');
fprintf('[D001] River: %s\n', river);
fprintf('============================================================\n');

PR = cfg.processedRoot;
mergedVrt = fullfile(PR, 'Bathy3DEP_Merged_1m_FixND', river, 'Combined_BathyPriority_1m.vrt');
bathyVrt  = fullfile(PR, 'Bathy_1m_FixND', river, 'Bathy_1m.vrt');
hiddenVrt = fullfile(cfg.hiddenRoot, 'HiddenMask_ByRiver_1m', river, 'HiddenMask_1m.vrt');
lossVrt   = fullfile(cfg.lossRoot, 'LossMask_ByRiver_1m', river, 'LossMask_1m.vrt');
centerShp = fullfile(cfg.centerRoot, river, 'ESA_WorldCover_Width_proj_Clip.shp');

mustExist(mergedVrt, 'merged 1m DEM');
mustExist(bathyVrt,  'bathy 1m');
mustExist(hiddenVrt, 'hidden mask 1m');
mustExist(lossVrt,   'loss mask 1m');
mustExist(centerShp, 'centerline point shp');

fprintf('Merged DEM : %s\n', mergedVrt);
fprintf('Bathy DEM  : %s\n', bathyVrt);
fprintf('HiddenMask : %s\n', hiddenVrt);
fprintf('LossMask   : %s\n', lossVrt);
fprintf('Center pts : %s\n', centerShp);

[~, rowsM, colsM, geoTrans, proj, dataTypeDEM, ndMerged] = RasterInfo(mergedVrt);
[~, rowsB, colsB, ~,        ~,    ~,           ~       ] = RasterInfo(bathyVrt);
[~, rowsH, colsH, ~,        ~,    ~,           ~       ] = RasterInfo(hiddenVrt);
[~, rowsL, colsL, ~,        ~,    ~,           ~       ] = RasterInfo(lossVrt);

if ~(rowsM == rowsB && rowsM == rowsH && rowsM == rowsL && colsM == colsB && colsM == colsH && colsM == colsL)
    error('Grid size mismatch for %s: merged=%d/%d bathy=%d/%d hidden=%d/%d loss=%d/%d', ...
        river, rowsM, colsM, rowsB, colsB, rowsH, colsH, rowsL, colsL);
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
tileRoot = fullfile(outRoot, 'Tiles_1m');
folders.Train            = fullfile(tileRoot, 'Train_tile');
folders.HiddenPixelQA    = fullfile(tileRoot, 'Hidden_Mask_Pixel_QA');
folders.HiddenPatchView  = fullfile(tileRoot, 'Hidden_Mask');
folders.HiddenPatch21    = fullfile(tileRoot, 'Hidden_Mask_Patch21_QA');
folders.LossPixel        = fullfile(tileRoot, 'Loss_Mask_Pixel');
folders.LossPatchView    = fullfile(tileRoot, 'Loss_Mask_PatchView_QA');
folders.LossPatch21      = fullfile(tileRoot, 'Loss_Mask_Patch21_QA');
folders.QA         = fullfile(outRoot, 'QA', river);
folders.Lists      = fullfile(outRoot, 'Lists');
makeFolders(folders);

GT = shaperead(centerShp);
if isempty(GT)
    error('Centerline shapefile has no points: %s', centerShp);
end

[X, Y] = extractPointXY(GT);
lineID = extractNumericField(GT, {'line_ID','LineID','LINEID','Line_ID','lineid'}, (1:numel(GT)).');
width  = extractNumericField(GT, {'Width','WIDTH','width','wid','WID'}, nan(numel(GT),1));

keepSpacing = thinByLineSpacing(X, Y, lineID, targetSpacingMap);
idxSpacing = find(keepSpacing);
fprintf('Raw center points      : %d\n', numel(GT));
fprintf('After spacing thinning : %d\n', numel(idxSpacing));

% Prepare output lists and QA tables.
qaRows = [];
manifestRows = [];
selectedStruct = [];
selectedCount = 0;
checkedCount = 0;

hr = floor(cfg.tileSize / 2);
hc = floor(cfg.tileSize / 2);
patchGrid = cfg.tileSize / cfg.patchSize;
coreMask = makeCoreMask(patchGrid, cfg.corePatchRadius);
coreTotal = nnz(coreMask);
corePixelMask = logical(expandPatchMask(coreMask, cfg.patchSize));
corePixelTotal = nnz(corePixelMask);

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
        qaRows = appendQARow(qaRows, srcIdx, NaN, x, y, lineID(srcIdx), width(srcIdx), ...
            row0, col0, false, reject, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN);
        continue;
    end

    tileDEM    = ReadRaster(mergedVrt, r1, c1, h, w);
    tileBathy  = ReadRaster(bathyVrt,  r1, c1, h, w);
    tileHidden = ReadRaster(hiddenVrt, r1, c1, h, w);
    tileLoss0  = ReadRaster(lossVrt,   r1, c1, h, w);

    inputValid = isValidValue(tileDEM, cfg.nodataValue, cfg.invalidBelow);
    bathyValid = isValidValue(tileBathy, cfg.nodataValue, cfg.invalidBelow);

    hiddenPixel = (tileHidden > 0) & (tileHidden < 255) & isfinite(tileHidden);
    manualLossPixel = (tileLoss0 > 0) & (tileLoss0 < 255) & isfinite(tileLoss0);
    finalLossPixel = manualLossPixel & bathyValid;

    hiddenPatch = blockAny(hiddenPixel, cfg.patchSize);
    lossPatch   = blockAll(finalLossPixel, cfg.patchSize);
    inputValidPatch = blockAll(inputValid, cfg.patchSize);
    visiblePatch = (~hiddenPatch) & inputValidPatch;

    coreLossCount = nnz(lossPatch & coreMask);
    coreLossRatio = coreLossCount / max(1, coreTotal);
    coreLossPixelCount = nnz(finalLossPixel & corePixelMask);
    coreLossPixelRatio = coreLossPixelCount / max(1, corePixelTotal);
    coreHiddenCount = nnz(hiddenPatch & coreMask);
    hiddenPatchCount = nnz(hiddenPatch);
    lossPatchCount = nnz(lossPatch);
    visiblePatchCount = nnz(visiblePatch);
    centerR = row0 - r1 + 1;
    centerC = col0 - c1 + 1;
    centerLoss = false;
    if centerR >= 1 && centerR <= h && centerC >= 1 && centerC <= w
        centerLoss = logical(finalLossPixel(centerR, centerC));
    end

    if cfg.requireCenterLoss && ~centerLoss
        reject = "center_not_final_loss";
    elseif coreLossPixelCount < cfg.minCoreLossPixelCount
        reject = "core_loss_pixel_count";
    elseif coreLossPixelRatio < cfg.minCoreLossPixelRatio
        reject = "core_loss_pixel_ratio";
    elseif coreLossCount < cfg.minCoreLossPatchCount
        reject = "core_loss_patch_count";
    elseif coreLossRatio < cfg.minCoreLossPatchRatio
        reject = "core_loss_patch_ratio";
    elseif visiblePatchCount < cfg.minVisiblePatchCount
        reject = "visible_patch_count";
    else
        reject = "kept";
    end

    isKeep = strcmp(reject, "kept");
    pointID = NaN;
    if isKeep
        selectedCount = selectedCount + 1;
        pointID = selectedCount;

        if selectedCount <= cfg.maxTilesPerRiver
            if ~cfg.skipWriteTiles
                subGT = subTranscoef(geoTrans, r1, c1);
                patchGT = subGT;
                patchGT(2) = patchGT(2) * cfg.patchSize;
                patchGT(6) = patchGT(6) * cfg.patchSize;

                demOut = fullfile(folders.Train, sprintf('Select_tile_Basin_1m_%s_ID%d.tif', river, pointID));
                hidPixOut = fullfile(folders.HiddenPixelQA, sprintf('Select_tile_1m_%s_ID%d_HiddenMaskPixel_QA.tif', river, pointID));
                hidViewOut = fullfile(folders.HiddenPatchView, sprintf('Select_tile_1m_%s_ID%d_HiddenMask.tif', river, pointID));
                hid21Out = fullfile(folders.HiddenPatch21, sprintf('Select_tile_1m_%s_ID%d_HiddenMaskPatch21_QA.tif', river, pointID));
                losPixOut = fullfile(folders.LossPixel, sprintf('Select_tile_1m_%s_ID%d_LossMaskPixel.tif', river, pointID));
                losViewOut = fullfile(folders.LossPatchView, sprintf('Select_tile_1m_%s_ID%d_LossMaskPatchView_QA.tif', river, pointID));
                los21Out = fullfile(folders.LossPatch21, sprintf('Select_tile_1m_%s_ID%d_LossMaskPatch21_QA.tif', river, pointID));

                writeMaybe(demOut, tileDEM, subGT, proj, dataTypeDEM, 'GTiff', resolveNoData(ndMerged, cfg.nodataValue), cfg.overwrite);

                % Hidden mask for model: patch-level. Hidden pixel mask is
                % optional QA only. The exact model token mask should use
                % Hidden_Mask_Patch21_QA or Hidden_Mask.
                if cfg.writeHiddenPixelQA
                    writeMaybe(hidPixOut, double(hiddenPixel), subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                else
                    hidPixOut = '';
                end

                % Loss mask for model: pixel-level final loss mask, already
                % combined with bathy-valid pixels.
                writeMaybe(losPixOut, double(finalLossPixel), subGT, proj, 1, 'GTiff', 255, cfg.overwrite);

                if cfg.writePatchView
                    hidView = expandPatchMask(hiddenPatch, cfg.patchSize);
                    losView = expandPatchMask(lossPatch, cfg.patchSize);
                    writeMaybe(hidViewOut, hidView, subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                    writeMaybe(losViewOut, losView, subGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                else
                    hidViewOut = '';
                    losViewOut = '';
                end

                if cfg.writePatch21
                    writeMaybe(hid21Out, double(hiddenPatch), patchGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                    writeMaybe(los21Out, double(lossPatch), patchGT, proj, 1, 'GTiff', 255, cfg.overwrite);
                else
                    hid21Out = '';
                    los21Out = '';
                end

                manifestRows = appendManifestRow(manifestRows, pointID, river, demOut, ...
                    hidPixOut, hidViewOut, hid21Out, losPixOut, losViewOut, los21Out);
            end

            selectedStruct = appendPointStruct(selectedStruct, pointID, srcIdx, x, y, lineID(srcIdx), width(srcIdx), ...
                coreLossCount, coreLossRatio, coreLossPixelCount, coreLossPixelRatio, coreHiddenCount, hiddenPatchCount, lossPatchCount, visiblePatchCount);
        end
    end

    qaRows = appendQARow(qaRows, srcIdx, pointID, x, y, lineID(srcIdx), width(srcIdx), ...
        row0, col0, isKeep, reject, mean(hiddenPixel(:)), mean(finalLossPixel(:)), mean(bathyValid(:)), ...
        hiddenPatchCount, lossPatchCount, visiblePatchCount, coreHiddenCount, coreLossCount, coreLossRatio, coreLossPixelCount, coreLossPixelRatio);

    if mod(checkedCount, 100) == 0
        fprintf('  checked %d/%d spacing-thinned points, kept=%d\n', checkedCount, numel(idxSpacing), selectedCount);
    end

    if selectedCount >= cfg.maxTilesPerRiver
        fprintf('Reached maxTilesPerRiver=%d, stop early for %s.\n', cfg.maxTilesPerRiver, river);
        break;
    end
end

fprintf('Checked spacing-thinned points: %d\n', checkedCount);
fprintf('Selected tiles               : %d\n', selectedCount);

% Write QA CSV.
qaTable = qaRowsToTable(qaRows);
qaCsv = fullfile(folders.QA, sprintf('D001_candidate_patch_QA_%s.csv', river));
writetable(qaTable, qaCsv);
fprintf('QA CSV: %s\n', qaCsv);

% Write selected point shapefile.
if ~isempty(selectedStruct)
    outPts = fullfile(folders.QA, sprintf('D001_selected_points_%s.shp', river));
    shapewrite(selectedStruct, outPts);
    copyPrj(centerShp, outPts);
    fprintf('Selected point shp: %s\n', outPts);
else
    warning('No selected points for %s.', river);
end

% Write manifest and paired lists.
if ~isempty(manifestRows)
    manifestTable = manifestRowsToTable(manifestRows);
    manifestCsv = fullfile(folders.QA, sprintf('D001_tile_manifest_%s.csv', river));
    writetable(manifestTable, manifestCsv);
    fprintf('Manifest CSV: %s\n', manifestCsv);

    writeList(fullfile(folders.Lists, sprintf('D001_train_tiles_%s.txt', river)), manifestTable.dem_path);
    writeList(fullfile(folders.Lists, sprintf('D001_hidden_masks_336patchview_%s.txt', river)), manifestTable.hidden_mask_path);
    writeList(fullfile(folders.Lists, sprintf('D001_loss_pixel_masks_%s.txt', river)), manifestTable.loss_pixel_mask_path);
    writeList(fullfile(folders.Lists, sprintf('D001_hidden_patch21_QA_%s.txt', river)), manifestTable.hidden_patch21_QA_path);
    writeList(fullfile(folders.Lists, sprintf('D001_loss_patchview_QA_%s.txt', river)), manifestTable.loss_patchview_QA_path);
    writeList(fullfile(folders.Lists, sprintf('D001_loss_patch21_QA_%s.txt', river)), manifestTable.loss_patch21_QA_path);
end
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

if isempty(rivers)
    return;
end
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

function [X, Y] = extractPointXY(GT)
n = numel(GT);
X = nan(n,1);
Y = nan(n,1);
for i = 1:n
    xi = GT(i).X;
    yi = GT(i).Y;
    xi = xi(~isnan(xi));
    yi = yi(~isnan(yi));
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
if isempty(chosen)
    return;
end
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
        % Accumulate distance along the original ordered points within this
        % line_ID. This gives stable, approximately uniform spacing even if
        % the original point spacing is irregular.
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

function valid = isValidValue(A, nodata, invalidBelow)
A = double(A);
valid = isfinite(A) & (A > invalidBelow);
if ~isempty(nodata) && isfinite(nodata)
    tol = max(1e-6, abs(double(nodata)) * 1e-7);
    valid = valid & (abs(A - double(nodata)) > tol);
end
end

function P = blockAny(mask, p)
[H,W] = size(mask);
gh = H / p;
gw = W / p;
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

function P = blockAll(mask, p)
[H,W] = size(mask);
gh = H / p;
gw = W / p;
P = false(gh, gw);
for rr = 1:gh
    r0 = (rr-1)*p + 1;
    for cc = 1:gw
        c0 = (cc-1)*p + 1;
        b = mask(r0:r0+p-1, c0:c0+p-1);
        P(rr,cc) = all(b(:));
    end
end
end

function core = makeCoreMask(gridN, radius)
core = false(gridN, gridN);
c = floor(gridN/2) + 1;
y0 = max(1, c - radius);
y1 = min(gridN, c + radius);
x0 = max(1, c - radius);
x1 = min(gridN, c + radius);
core(y0:y1, x0:x1) = true;
end

function out = expandPatchMask(P, p)
% Return double 0/1 matrix. The custom WriteRaster.mexa64 used in this
% pipeline is safest with MATLAB double/single arrays; passing uint8/logical
% masks can generate corrupted sparse-looking rasters in some environments.
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
        if exist(auxPath, 'file') == 2
            delete(auxPath);
        end
    end
end
if exist(fileparts(outPath), 'dir') ~= 7
    mkdir(fileparts(outPath));
end

% Important for mask rasters:
% WriteRaster.mexa64 in the existing pipeline is reliable when the MATLAB
% array is double/single.  Passing uint8/logical arrays can produce sparse or
% striped-looking 0/1 rasters even though the intended output data type is
% Byte.  Therefore all Byte masks are passed as double 0/1 arrays while
% dataType=1 still writes Byte GeoTIFF.
if dataType == 1
    A = double(A);
    A(~isfinite(A)) = nodata;
end
WriteRaster(outPath, A, geoTrans, proj, dataType, outFormat, nodata);
end

function rows = appendQARow(rows, srcIdx, pointID, x, y, lineID, width, row0, col0, kept, reject, ...
    hiddenPixRatio, lossPixRatio, bathyValidRatio, hiddenPatchCount, lossPatchCount, visiblePatchCount, coreHiddenCount, coreLossCount, coreLossRatio, coreLossPixelCount, coreLossPixelRatio)
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
r.final_loss_pixel_ratio = lossPixRatio;
r.bathy_valid_pixel_ratio = bathyValidRatio;
r.hidden_patch_count = hiddenPatchCount;
r.loss_patch_count = lossPatchCount;
r.visible_patch_count = visiblePatchCount;
r.core_hidden_patch_count = coreHiddenCount;
r.core_loss_patch_count = coreLossCount;
r.core_loss_patch_ratio = coreLossRatio;
r.core_loss_pixel_count = coreLossPixelCount;
r.core_loss_pixel_ratio = coreLossPixelRatio;
if isempty(rows)
    rows = r;
else
    rows(end+1,1) = r;
end
end

function T = qaRowsToTable(rows)
if isempty(rows)
    T = table();
else
    T = struct2table(rows);
end
end

function rows = appendManifestRow(rows, pointID, river, demPath, hiddenPixelPath, hiddenViewPath, hidden21Path, lossPixelPath, lossViewPath, loss21Path)
r.point_id = pointID;
r.river = river;
r.dem_path = demPath;
r.hidden_pixel_QA_path = hiddenPixelPath;
r.hidden_mask_path = hiddenViewPath;              % 336 x 336 patch-level mask for model
r.hidden_patch21_QA_path = hidden21Path;          % 21 x 21 QA/token-view mask
r.loss_pixel_mask_path = lossPixelPath;           % 336 x 336 pixel-level loss mask for model
r.loss_patchview_QA_path = lossViewPath;
r.loss_patch21_QA_path = loss21Path;
if isempty(rows)
    rows = r;
else
    rows(end+1,1) = r;
end
end

function T = manifestRowsToTable(rows)
if isempty(rows)
    T = table();
else
    T = struct2table(rows);
end
end

function S = appendPointStruct(S, pointID, srcIdx, x, y, lineID, width, coreLoss, coreLossRatio, coreLossPix, coreLossPixRatio, coreHidden, hiddenPatch, lossPatch, visiblePatch)
% Short DBF field names are used for shapefile compatibility.
r.Geometry = 'Point';
r.X = x;
r.Y = y;
r.PointID = pointID;
r.SrcID = srcIdx;
r.LineID = lineID;
r.Width = width;
r.CLossN = coreLoss;
r.CLossR = coreLossRatio;
r.CLossPixN = coreLossPix;
r.CLossPixR = coreLossPixRatio;
r.CHidN = coreHidden;
r.HidPN = hiddenPatch;
r.LossPN = lossPatch;
r.VisPN = visiblePatch;
if isempty(S)
    S = r;
else
    S(end+1,1) = r;
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
if istable(vals)
    vals = table2array(vals);
end
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
