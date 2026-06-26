function C003_Align_BetterMask_Aux_To_BathyGrid_All(varargin)
% C003_Align_BetterMask_Aux_To_BathyGrid_All
%
% Align the two new BetterMask auxiliary datasets to the existing bathy grids.
%
% Inputs prepared by previous steps:
%   C001: Data_for_BetterMask/US_Detailed_ByRiver_SHP/<river>/USRiverClip.shp
%   C002: Data_for_BetterMask/Water_Prob_VRT/Water_Prob_CONUS.vrt
%
% Reference grids:
%   Processed_Results/Bathy_<res>m_FixND/<river>/Bathy_<res>m.vrt
%
% Outputs, all under a new BetterMask folder:
%   Data_for_BetterMask/Auxiliary_ByRiver_<res>m/<river>/WaterProb_<res>m.vrt
%   Data_for_BetterMask/Auxiliary_ByRiver_<res>m/<river>/USRiver_<res>m.tif
%   Data_for_BetterMask/Auxiliary_ByRiver_<res>m/<river>/target_bathy_srs.wkt
%
% Notes:
%   - Water Probability is warped with nearest-neighbor resampling.
%   - Water Probability NoData is treated as 255.
%   - US detailed RiverOnly shapefile is rasterized as Byte: river=1, background=0.
%   - The output grid is exactly the reference bathy grid: same CRS, extent, rows, cols.
%   - This script does not overwrite any existing LCC or bathy/3DEP products.
%
% Example:
%   C003_Align_BetterMask_Aux_To_BathyGrid_All( ...
%       'selectedRivers', {'MD_PotomacRiver_Bathy_2019'}, ...
%       'resolutions', [10], ...
%       'overwrite', true)
%
%   C003_Align_BetterMask_Aux_To_BathyGrid_All( ...
%       'selectedRivers', {'MD_PotomacRiver_Bathy_2019'}, ...
%       'resolutions', [1 3 5 10], ...
%       'overwrite', true)

p = inputParser;
addParameter(p, 'processedRoot', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results', @ischar);
addParameter(p, 'betterMaskRoot', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Data_for_BetterMask', @ischar);
addParameter(p, 'selectedRivers', {}, @(x) iscell(x) || isstring(x) || ischar(x));
addParameter(p, 'resolutions', [1 3 5 10], @isnumeric);
addParameter(p, 'overwrite', true, @islogical);
addParameter(p, 'runWaterProb', true, @islogical);
addParameter(p, 'runUSDetailed', true, @islogical);
addParameter(p, 'waterProbVRT', '', @ischar);
addParameter(p, 'usByRiverRoot', '', @ischar);
addParameter(p, 'outputPrefix', 'Auxiliary_ByRiver', @ischar);
addParameter(p, 'waterProbOutFormat', 'VRT', @ischar);  % 'VRT' or 'GTiff'
addParameter(p, 'waterProbSrcNoData', 255, @isnumeric);
addParameter(p, 'waterProbDstNoData', 255, @isnumeric);
addParameter(p, 'usAllTouched', true, @islogical);
addParameter(p, 'dryRun', false, @islogical);
parse(p, varargin{:});
opt = p.Results;

if ischar(opt.selectedRivers) || isstring(opt.selectedRivers)
    opt.selectedRivers = cellstr(opt.selectedRivers);
end
opt.resolutions = unique(opt.resolutions(:)');

if isempty(opt.waterProbVRT)
    opt.waterProbVRT = fullfile(opt.betterMaskRoot, 'Water_Prob_VRT', 'Water_Prob_CONUS.vrt');
end
if isempty(opt.usByRiverRoot)
    opt.usByRiverRoot = fullfile(opt.betterMaskRoot, 'US_Detailed_ByRiver_SHP');
end

logDir = fullfile(opt.betterMaskRoot, 'Logs');
if ~exist(logDir, 'dir'); mkdir(logDir); end
logCsv = fullfile(logDir, 'C003_Align_BetterMask_Aux_To_BathyGrid_Log.csv');

if opt.runWaterProb && ~exist(opt.waterProbVRT, 'file')
    error('Water Probability VRT not found: %s', opt.waterProbVRT);
end

rivers = getRiverList(opt.processedRoot, opt.selectedRivers);

fprintf('\n============================================================\n');
fprintf('C003 align BetterMask auxiliary data to bathy grids\n');
fprintf('Processed root    : %s\n', opt.processedRoot);
fprintf('BetterMask root   : %s\n', opt.betterMaskRoot);
fprintf('WaterProb VRT     : %s\n', opt.waterProbVRT);
fprintf('US by-river root  : %s\n', opt.usByRiverRoot);
fprintf('Resolutions       : %s m\n', mat2str(opt.resolutions));
fprintf('Number of rivers  : %d\n', numel(rivers));
fprintf('WaterProb format  : %s\n', opt.waterProbOutFormat);
fprintf('Overwrite         : %d\n', opt.overwrite);
fprintf('Dry run           : %d\n', opt.dryRun);
fprintf('============================================================\n\n');

records = {};
recN = 0;

tStartAll = tic;
for ir = 1:numel(rivers)
    river = rivers{ir};
    fprintf('\n============================================================\n');
    fprintf('[%d/%d] River: %s\n', ir, numel(rivers), river);
    fprintf('============================================================\n');

    for res = opt.resolutions
        fprintf('\n--- Resolution: %dm ---\n', res);
        refGrid = getBathyRefGrid(opt.processedRoot, river, res);
        outDir = fullfile(opt.betterMaskRoot, sprintf('%s_%dm', opt.outputPrefix, res), river);
        if ~exist(outDir, 'dir'); mkdir(outDir); end

        recBase = struct();
        recBase.river = river;
        recBase.resolution_m = res;
        recBase.ref_grid = refGrid;
        recBase.out_dir = outDir;
        recBase.waterprob_status = "SKIP";
        recBase.usriver_status = "SKIP";
        recBase.message = "";
        recBase.rows = NaN;
        recBase.cols = NaN;
        recBase.xmin = NaN;
        recBase.ymin = NaN;
        recBase.xmax = NaN;
        recBase.ymax = NaN;
        recBase.waterprob_out = "";
        recBase.usriver_out = "";

        if ~exist(refGrid, 'file')
            msg = sprintf('Missing reference grid: %s', refGrid);
            warning('[%s %dm] %s', river, res, msg);
            recBase.message = string(msg);
            records = addRecord(records, recBase); %#ok<AGROW>
            continue;
        end

        try
            info = getRasterInfoGDAL(refGrid, outDir);
            recBase.rows = info.rows;
            recBase.cols = info.cols;
            recBase.xmin = info.xmin;
            recBase.ymin = info.ymin;
            recBase.xmax = info.xmax;
            recBase.ymax = info.ymax;
        catch ME
            msg = sprintf('Failed to read reference grid info: %s', ME.message);
            warning('[%s %dm] %s', river, res, msg);
            recBase.message = string(msg);
            records = addRecord(records, recBase); %#ok<AGROW>
            continue;
        end

        srsWkt = fullfile(outDir, 'target_bathy_srs.wkt');
        try
            writeRasterSRS(refGrid, srsWkt, info);
        catch ME
            msg = sprintf('Failed to write target SRS WKT: %s', ME.message);
            warning('[%s %dm] %s', river, res, msg);
            recBase.message = string(msg);
            records = addRecord(records, recBase); %#ok<AGROW>
            continue;
        end

        fprintf('Reference grid: %s\n', refGrid);
        fprintf('Rows/Cols     : %d / %d\n', info.rows, info.cols);
        fprintf('BBox          : %.10f %.10f %.10f %.10f\n', info.xmin, info.ymin, info.xmax, info.ymax);
        fprintf('Target SRS    : %s\n', srsWkt);

        rec = recBase;

        % ------------------------------------------------------------
        % 1) Water Probability aligned raster
        % ------------------------------------------------------------
        if opt.runWaterProb
            if strcmpi(opt.waterProbOutFormat, 'GTiff') || strcmpi(opt.waterProbOutFormat, 'TIF') || strcmpi(opt.waterProbOutFormat, 'TIFF')
                wpOut = fullfile(outDir, sprintf('WaterProb_%dm.tif', res));
                wpFormat = 'GTiff';
                wpCreateOpts = '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ';
            else
                wpOut = fullfile(outDir, sprintf('WaterProb_%dm.vrt', res));
                wpFormat = 'VRT';
                wpCreateOpts = '';
            end
            rec.waterprob_out = string(wpOut);

            if exist(wpOut, 'file') && ~opt.overwrite
                fprintf('[WaterProb] Exists, skip: %s\n', wpOut);
                rec.waterprob_status = "EXISTS";
            else
                if exist(wpOut, 'file') && opt.overwrite
                    deleteIfExists(wpOut);
                end

                cmdWP = sprintf([ ...
                    'gdalwarp -overwrite -of %s ', ...
                    '-r near ', ...
                    '-t_srs %s -te_srs %s ', ...
                    '-te %.10f %.10f %.10f %.10f ', ...
                    '-ts %d %d ', ...
                    '-srcnodata %g -dstnodata %g ', ...
                    '%s', ...
                    '%s %s'], ...
                    wpFormat, ...
                    qpath(srsWkt), qpath(srsWkt), ...
                    info.xmin, info.ymin, info.xmax, info.ymax, ...
                    info.cols, info.rows, ...
                    opt.waterProbSrcNoData, opt.waterProbDstNoData, ...
                    wpCreateOpts, ...
                    qpath(opt.waterProbVRT), qpath(wpOut));

                fprintf('[WaterProb] Running:\n%s\n', cmdWP);
                if opt.dryRun
                    rec.waterprob_status = "DRYRUN";
                else
                    [statusWP, msgWP] = system(cmdWP, '-echo');
                    if statusWP ~= 0
                        warning('[%s %dm] WaterProb gdalwarp failed:\n%s', river, res, msgWP);
                        rec.waterprob_status = "FAIL";
                        rec.message = appendMessage(rec.message, sprintf('WaterProb failed: %s', strtrim(msgWP)));
                    else
                        rec.waterprob_status = "OK";
                    end
                end
            end
        end

        % ------------------------------------------------------------
        % 2) US Detailed RiverOnly rasterized to the same grid
        % ------------------------------------------------------------
        if opt.runUSDetailed
            usShp = fullfile(opt.usByRiverRoot, river, 'USRiverClip.shp');
            usOut = fullfile(outDir, sprintf('USRiver_%dm.tif', res));
            rec.usriver_out = string(usOut);

            if ~exist(usShp, 'file')
                msg = sprintf('Missing US by-river SHP: %s', usShp);
                warning('[%s %dm] %s', river, res, msg);
                rec.usriver_status = "MISSING_SHP";
                rec.message = appendMessage(rec.message, msg);
            elseif exist(usOut, 'file') && ~opt.overwrite
                fprintf('[USRiver] Exists, skip: %s\n', usOut);
                rec.usriver_status = "EXISTS";
            else
                if exist(usOut, 'file') && opt.overwrite
                    deleteIfExists(usOut);
                end

                allTouchedFlag = '';
                if opt.usAllTouched
                    allTouchedFlag = '-at ';
                end

                % cmdUS = sprintf([ ...
                %     'gdal_rasterize ', ...
                %     '-burn 1 ', ...
                %     '-ot Byte ', ...
                %     '-init 0 ', ...
                %     '-a_nodata 0 ', ...
                %     '-a_srs %s ', ...
                %     '-te %.10f %.10f %.10f %.10f ', ...
                %     '-ts %d %d ', ...
                %     '%s', ...
                %     '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ', ...
                %     '%s %s'], ...
                %     qpath(srsWkt), ...
                %     info.xmin, info.ymin, info.xmax, info.ymax, ...
                %     info.cols, info.rows, ...
                %     allTouchedFlag, ...
                %     qpath(usShp), qpath(usOut));

                allTouchedFlag = '';
                if opt.usAllTouched
                    allTouchedFlag = '-at ';
                end

                optimFlag = '';
                if res == 1
                    optimFlag = '-optim VECTOR ';
                end

                cmdUS = sprintf([ ...
                    'gdal_rasterize ', ...
                    '-burn 1 ', ...
                    '-ot Byte ', ...
                    '-a_nodata 0 ', ...
                    '-a_srs %s ', ...
                    '-te %.10f %.10f %.10f %.10f ', ...
                    '-ts %d %d ', ...
                    '%s', ...
                    '%s', ...
                    '-co TILED=YES ', ...
                    '-co BLOCKXSIZE=512 -co BLOCKYSIZE=512 ', ...
                    '-co COMPRESS=LZW ', ...
                    '-co BIGTIFF=YES ', ...
                    '-co SPARSE_OK=YES ', ...
                    '%s %s'], ...
                    qpath(srsWkt), ...
                    info.xmin, info.ymin, info.xmax, info.ymax, ...
                    info.cols, info.rows, ...
                    allTouchedFlag, ...
                    optimFlag, ...
                    qpath(usShp), qpath(usOut));

                fprintf('[USRiver] Running:\n%s\n', cmdUS);
                if opt.dryRun
                    rec.usriver_status = "DRYRUN";
                else
                    [statusUS, msgUS] = system(cmdUS, '-echo');

                    if statusUS ~= 0
                        warning('[%s %dm] Full USRiver rasterize failed. Try tiled rasterize...', river, res);

                        try
                            usVrtOut = rasterizeUSRiverTiled( ...
                                usShp, srsWkt, info, outDir, res, opt.usAllTouched);

                            rec.usriver_out = string(usVrtOut);
                            rec.usriver_status = "OK_TILED_VRT";
                            rec.message = appendMessage(rec.message, 'Full rasterize failed; tiled VRT created.');
                        catch ME
                            warning('[%s %dm] Tiled USRiver rasterize also failed:\n%s', river, res, ME.message);
                            rec.usriver_status = "FAIL";
                            rec.message = appendMessage(rec.message, sprintf('USRiver failed: %s', strtrim(msgUS)));
                        end
                    else
                        rec.usriver_status = "OK";
                    end
                end
            end
        end

        records = addRecord(records, rec); %#ok<AGROW>
    end
end

T = recordsToTable(records);
writetable(T, logCsv);

fprintf('\n============================================================\n');
fprintf('C003 done. Elapsed: %.1f seconds\n', toc(tStartAll));
fprintf('Log written to:\n%s\n', logCsv);
fprintf('============================================================\n');

end

% ======================================================================
% Helpers
% ======================================================================

function rivers = getRiverList(processedRoot, selectedRivers)
if ~isempty(selectedRivers)
    rivers = selectedRivers;
    return;
end
bathy1Root = fullfile(processedRoot, 'Bathy_1m_FixND');
if ~exist(bathy1Root, 'dir')
    error('Cannot find Bathy_1m_FixND root: %s', bathy1Root);
end
d = dir(bathy1Root);
keep = [d.isdir] & ~startsWith({d.name}, '.');
rivers = sort({d(keep).name});
end

function refGrid = getBathyRefGrid(processedRoot, river, res)
refGrid = fullfile(processedRoot, sprintf('Bathy_%dm_FixND', res), river, sprintf('Bathy_%dm.vrt', res));
end

function info = getRasterInfoGDAL(refGrid, tmpDir)
jsonFile = fullfile(tmpDir, 'reference_grid_gdalinfo.json');
cmd = sprintf('gdalinfo -json %s > %s', qpath(refGrid), qpath(jsonFile));
[status, msg] = system(cmd);
if status ~= 0
    error('gdalinfo -json failed: %s', msg);
end
jtxt = fileread(jsonFile);
j = jsondecode(jtxt);

info.cols = double(j.size(1));
info.rows = double(j.size(2));

if ~isfield(j, 'geoTransform') || numel(j.geoTransform) < 6
    error('No geoTransform found in gdalinfo json for: %s', refGrid);
end
gt = double(j.geoTransform(:)');

x0 = gt(1); px = gt(2); rx = gt(3);
y0 = gt(4); ry = gt(5); py = gt(6);

xs = [x0, x0 + info.cols*px, x0 + info.rows*rx, x0 + info.cols*px + info.rows*rx];
ys = [y0, y0 + info.cols*ry, y0 + info.rows*py, y0 + info.cols*ry + info.rows*py];

info.xmin = min(xs);
info.xmax = max(xs);
info.ymin = min(ys);
info.ymax = max(ys);
info.geoTransform = gt;

if isfield(j, 'coordinateSystem') && isfield(j.coordinateSystem, 'wkt')
    info.wkt = j.coordinateSystem.wkt;
else
    info.wkt = '';
end
end

function writeRasterSRS(refGrid, srsWkt, info)
cmd = sprintf('gdalsrsinfo -o wkt %s > %s', qpath(refGrid), qpath(srsWkt));
[status, msg] = system(cmd);
if status ~= 0 || ~exist(srsWkt, 'file') || dir(srsWkt).bytes == 0
    if isfield(info, 'wkt') && ~isempty(info.wkt)
        fid = fopen(srsWkt, 'w');
        if fid < 0
            error('Cannot open WKT for writing: %s', srsWkt);
        end
        fprintf(fid, '%s\n', info.wkt);
        fclose(fid);
    else
        error('Cannot get SRS WKT. gdalsrsinfo message: %s', msg);
    end
end
end

function deleteIfExists(path0)
% Delete a file and common sidecar files for shapefile-like or raster-like outputs.
[folder, base, ext] = fileparts(path0);
if strcmpi(ext, '.shp')
    exts = {'.shp','.shx','.dbf','.prj','.cpg','.qpj'};
elseif strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
    exts = {ext, [ext '.aux.xml'], '.aux.xml', '.ovr'};
elseif strcmpi(ext, '.vrt')
    exts = {'.vrt'};
else
    exts = {ext};
end
for i = 1:numel(exts)
    f = fullfile(folder, [base exts{i}]);
    if exist(f, 'file')
        delete(f);
    end
end
end

function s = qpath(x)
s = ['"', char(x), '"'];
end

function msg = appendMessage(msg, newMsg)
if strlength(string(msg)) == 0
    msg = string(newMsg);
else
    msg = string(msg) + " | " + string(newMsg);
end
end

function records = addRecord(records, rec)
records{end+1,1} = rec;
end

function T = recordsToTable(records)
if isempty(records)
    T = table();
    return;
end
names = fieldnames(records{1});
S = struct();
for i = 1:numel(names)
    name = names{i};
    vals = cell(numel(records),1);
    for k = 1:numel(records)
        if isfield(records{k}, name)
            vals{k} = records{k}.(name);
        else
            vals{k} = missing;
        end
    end
    % Convert to reasonable table columns.
    if all(cellfun(@(x) isnumeric(x) && isscalar(x), vals))
        S.(name) = cellfun(@double, vals);
    else
        S.(name) = string(vals);
    end
end
T = struct2table(S);
end

function usVrtOut = rasterizeUSRiverTiled(usShp, srsWkt, info, outDir, res, usAllTouched)
% Rasterize USRiver in spatial tiles and build a VRT mosaic.
% Output:
%   USRiver_<res>m.vrt
%   USRiver_<res>m_tiles/*.tif

tileSize = 10000;  % safer for huge 1m/3m grids
tileDir = fullfile(outDir, sprintf('USRiver_%dm_tiles', res));
if ~exist(tileDir, 'dir'); mkdir(tileDir); end

usVrtOut = fullfile(outDir, sprintf('USRiver_%dm.vrt', res));
tileList = fullfile(tileDir, sprintf('USRiver_%dm_tile_list.txt', res));

% Clean old tiled outputs
oldTiles = dir(fullfile(tileDir, sprintf('USRiver_%dm_tile_*.tif', res)));
for i = 1:numel(oldTiles)
    delete(fullfile(oldTiles(i).folder, oldTiles(i).name));
end
if exist(tileList, 'file'); delete(tileList); end
if exist(usVrtOut, 'file'); delete(usVrtOut); end

dx = (info.xmax - info.xmin) / info.cols;
dy = (info.ymax - info.ymin) / info.rows;

allTouchedFlag = '';
if usAllTouched
    allTouchedFlag = '-at ';
end

fid = fopen(tileList, 'w');
if fid < 0
    error('Cannot open tile list for writing: %s', tileList);
end

nTiles = 0;
for r0 = 0:tileSize:(info.rows-1)
    r1 = min(r0 + tileSize, info.rows);
    tileRows = r1 - r0;

    % North-up grid:
    % row 0 starts at ymax.
    yMaxTile = info.ymax - r0 * dy;
    yMinTile = info.ymax - r1 * dy;

    for c0 = 0:tileSize:(info.cols-1)
        c1 = min(c0 + tileSize, info.cols);
        tileCols = c1 - c0;

        xMinTile = info.xmin + c0 * dx;
        xMaxTile = info.xmin + c1 * dx;

        tileName = sprintf('USRiver_%dm_tile_r%06d_c%06d.tif', res, r0, c0);
        tilePath = fullfile(tileDir, tileName);

        cmdTile = sprintf([ ...
            'gdal_rasterize ', ...
            '-burn 1 ', ...
            '-ot Byte ', ...
            '-a_nodata 0 ', ...
            '-a_srs %s ', ...
            '-te %.10f %.10f %.10f %.10f ', ...
            '-ts %d %d ', ...
            '%s', ...
            '-co TILED=YES ', ...
            '-co BLOCKXSIZE=512 -co BLOCKYSIZE=512 ', ...
            '-co COMPRESS=LZW ', ...
            '-co BIGTIFF=YES ', ...
            '-co SPARSE_OK=YES ', ...
            '%s %s'], ...
            qpath(srsWkt), ...
            xMinTile, yMinTile, xMaxTile, yMaxTile, ...
            tileCols, tileRows, ...
            allTouchedFlag, ...
            qpath(usShp), qpath(tilePath));

        fprintf('[USRiver tiled] r=%d:%d c=%d:%d\n', r0, r1, c0, c1);
        [statusTile, msgTile] = system(cmdTile, '-echo');
        if statusTile ~= 0
            fclose(fid);
            error('Tile rasterize failed: %s\n%s', tilePath, msgTile);
        end

        fprintf(fid, '%s\n', tilePath);
        nTiles = nTiles + 1;
    end
end

fclose(fid);

cmdVRT = sprintf([ ...
    'gdalbuildvrt -overwrite ', ...
    '-srcnodata 0 -vrtnodata 0 ', ...
    '-input_file_list %s %s'], ...
    qpath(tileList), qpath(usVrtOut));

fprintf('[USRiver tiled] Build VRT:\n%s\n', cmdVRT);
[statusVRT, msgVRT] = system(cmdVRT, '-echo');
if statusVRT ~= 0
    error('gdalbuildvrt failed: %s', msgVRT);
end

fprintf('[USRiver tiled] Done. nTiles=%d\n', nTiles);
fprintf('[USRiver tiled] Output VRT: %s\n', usVrtOut);

end