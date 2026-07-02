function Summary = C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver(riverName, targetResM, varargin)
% C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver
% 2026-07-01
%
% Purpose
%   Rasterize manually edited Hidden_Mask.shp to the bathymetry grids at
%   1 m / 3 m / 5 m / 10 m resolutions.
%
% Input shapefile
%   /tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw/Mask_ShapeFiles/<river>/Hidden_Mask.shp
%
% Reference grids
%   /tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_<res>m_FixND/<river>/Bathy_<res>m.vrt
%
% Output
%   /tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw/HiddenMask_ByRiver_<res>m/<river>/HiddenMask_<res>m.tif
%   /tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw/HiddenMask_ByRiver_<res>m/<river>/HiddenMask_<res>m.vrt
%
% Mask values
%   1   = hidden area, burn from Hidden_Mask.shp
%   0   = non-hidden background inside the reference grid extent
%   255 = NoData metadata value, normally not written except if explicitly needed
%
% Notes
%   - Output grid exactly matches the target bathy grid: CRS, extent, rows, cols.
%   - 0 is a valid mask value. It is NOT NoData.
%   - NoData metadata is set to 255 to avoid confusing 0-background with NoData.
%   - Default uses -at, so any pixel touched by the polygon becomes hidden=1.
%   - Large outputs can automatically use tiled GeoTIFFs plus a VRT mosaic.
%   - Safe to run different rivers in different MATLAB sessions.
%
% Usage examples
%   C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver('MD_PotomacRiver_Bathy_2019');
%   C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver('CA_KlamathRiver_TopoBathy_2018_D18', [1 3 5 10]);
%   C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver('OR_MKRC_Topobathy_2021', [3 5 10]);
%   C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver({'MD_PotomacRiver_Bathy_2019','CA_KlamathRiver_TopoBathy_2018_D18'}, [10]);
%   C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver('ALL');
%   C001_Rasterize_HiddenMask_To_BathyGrid_ForRiver('LIST');

if nargin < 1 || isempty(riverName)
    riverName = 'LIST';
end
if nargin < 2 || isempty(targetResM)
    targetResM = [1 3 5 10];
end

cfg = defaultConfig();
cfg = parseOptions(cfg, varargin{:});

setupGdalPaths();
ensureDir(cfg.logRoot);

validRivers = selectedRivers();

if ischar(riverName) || isstring(riverName)
    riverName = char(riverName);
    if strcmpi(riverName, 'LIST')
        printRiverList(validRivers);
        Summary = table();
        return;
    elseif strcmpi(riverName, 'ALL')
        riversToRun = validRivers;
    else
        riversToRun = {riverName};
    end
elseif iscell(riverName)
    riversToRun = riverName;
else
    error('riverName must be char/string, cell array, ''ALL'', or ''LIST''.');
end

if ~isnumeric(targetResM) || isempty(targetResM)
    error('targetResM must be numeric vector, e.g. [1 3 5 10].');
end
targetResM = unique(targetResM(:)');

for i = 1:numel(riversToRun)
    if ~ismember(riversToRun{i}, validRivers)
        fprintf('\nUnknown river: %s\n', riversToRun{i});
        printRiverList(validRivers);
        error('River is not in selected valid river list.');
    end
end

fprintf('\n============================================================\n');
fprintf('C001 Hidden_Mask shapefile -> bathy-grid raster\n');
fprintf('Mask SHP root : %s\n', cfg.maskShpRoot);
fprintf('Bathy root    : %s\n', cfg.prRoot);
fprintf('Output root   : %s/HiddenMask_ByRiver_<res>m\n', cfg.maskRoot);
fprintf('Rivers        : %s\n', strjoin(riversToRun, ', '));
fprintf('Resolutions   : %s m\n', mat2str(targetResM));
fprintf('Burn value    : %d\n', cfg.burnValue);
fprintf('Background    : %d\n', cfg.backgroundValue);
fprintf('NoData meta   : %d\n', cfg.maskNoData);
fprintf('All touched   : %d\n', cfg.allTouched);
fprintf('Output mode   : %s\n', cfg.outputMode);
fprintf('Overwrite     : %d\n', cfg.overwrite);
fprintf('============================================================\n');

summaryRows = {};
rowId = 0;

for r = 1:numel(riversToRun)
    river = riversToRun{r};
    shp = fullfile(cfg.maskShpRoot, river, 'Hidden_Mask.shp');

    fprintf('\n############################################################\n');
    fprintf('River: %s\n', river);
    fprintf('Hidden SHP: %s\n', shp);
    fprintf('############################################################\n');

    if exist(shp, 'file') ~= 2
        msg = sprintf('Missing Hidden_Mask.shp: %s', shp);
        if cfg.stopOnMissing
            error(msg);
        else
            warning(msg);
            for j = 1:numel(targetResM)
                rowId = rowId + 1;
                summaryRows(rowId,:) = {river, targetResM(j), '', shp, '', '', 'MISSING_SHP', msg}; %#ok<AGROW>
            end
            continue;
        end
    end

    for j = 1:numel(targetResM)
        resM = targetResM(j);
        refGrid = fullfile(cfg.prRoot, sprintf('Bathy_%dm_FixND', resM), river, sprintf('Bathy_%dm.vrt', resM));
        outRoot = fullfile(cfg.maskRoot, sprintf('HiddenMask_ByRiver_%dm', resM), river);
        ensureDir(outRoot);

        outTif = fullfile(outRoot, sprintf('HiddenMask_%dm.tif', resM));
        outVrt = fullfile(outRoot, sprintf('HiddenMask_%dm.vrt', resM));
        infoJson = fullfile(outRoot, 'reference_grid_gdalinfo.json');
        srsWkt = fullfile(outRoot, 'target_bathy_srs.wkt');

        fprintf('\n============================================================\n');
        fprintf('[%s] Rasterize Hidden_Mask to %dm bathy grid\n', river, resM);
        fprintf('Reference: %s\n', refGrid);
        fprintf('Output   : %s\n', outTif);
        fprintf('============================================================\n');

        if exist(refGrid, 'file') ~= 2
            msg = sprintf('Missing reference bathy grid: %s', refGrid);
            if cfg.stopOnMissing
                error(msg);
            else
                warning(msg);
                rowId = rowId + 1;
                summaryRows(rowId,:) = {river, resM, refGrid, shp, outTif, outVrt, 'MISSING_REF', msg}; %#ok<AGROW>
                continue;
            end
        end

        try
            info = getRasterInfoGDAL(refGrid, infoJson);
            writeRasterSRS(refGrid, srsWkt, info);
        catch ME
            if cfg.stopOnError
                rethrow(ME);
            end
            warning('[%s %dm] Failed to read reference grid/SRS: %s', river, resM, ME.message);
            rowId = rowId + 1;
            summaryRows(rowId,:) = {river, resM, refGrid, shp, outTif, outVrt, 'REF_INFO_FAIL', ME.message}; %#ok<AGROW>
            continue;
        end

        fprintf('Rows/Cols: %d / %d\n', info.rows, info.cols);
        fprintf('BBox     : %.10f %.10f %.10f %.10f\n', info.xmin, info.ymin, info.xmax, info.ymax);

        try
            shpUse = prepareShapefileForTargetSRS(shp, srsWkt, outRoot, cfg);
            modeUsed = chooseOutputMode(cfg, info, resM);

            if strcmpi(modeUsed, 'tiled')
                outMain = rasterizeHiddenMaskTiled(shpUse, srsWkt, info, outRoot, resM, cfg);
                outVrt = outMain;
                outTif = '';
                statusText = 'OK_TILED_VRT';
            else
                rasterizeHiddenMaskSingle(shpUse, srsWkt, info, outTif, cfg);
                buildVRTFromSingleTif(outTif, outVrt, cfg);
                statusText = 'OK';
            end

            printRasterInfo(outVrt);
            quickMaskStats(outVrt, cfg);
            msg = sprintf('mode=%s', modeUsed);

        catch ME
            % If direct single GeoTIFF failed, try tiled fallback once.
            if strcmpi(cfg.outputMode, 'single') || strcmpi(cfg.outputMode, 'auto')
                warning('[%s %dm] Single rasterize failed, try tiled fallback: %s', river, resM, ME.message);
                try
                    shpUse = prepareShapefileForTargetSRS(shp, srsWkt, outRoot, cfg);
                    outMain = rasterizeHiddenMaskTiled(shpUse, srsWkt, info, outRoot, resM, cfg);
                    outVrt = outMain;
                    outTif = '';
                    statusText = 'OK_TILED_VRT_FALLBACK';
                    printRasterInfo(outVrt);
                    quickMaskStats(outVrt, cfg);
                    msg = 'single rasterize failed; tiled fallback succeeded';
                catch ME2
                    if cfg.stopOnError
                        rethrow(ME2);
                    end
                    warning('[%s %dm] Tiled fallback failed: %s', river, resM, ME2.message);
                    statusText = 'FAIL';
                    msg = ME2.message;
                end
            else
                if cfg.stopOnError
                    rethrow(ME);
                end
                warning('[%s %dm] Rasterize failed: %s', river, resM, ME.message);
                statusText = 'FAIL';
                msg = ME.message;
            end
        end

        rowId = rowId + 1;
        summaryRows(rowId,:) = {river, resM, refGrid, shp, outTif, outVrt, statusText, msg}; %#ok<AGROW>
    end
end

Summary = rowsToTable(summaryRows);
summaryFile = fullfile(cfg.logRoot, sprintf('C001_HiddenMask_Rasterize_%s_%s.csv', safeName(strjoin(riversToRun, '__')), datestr(now, 'yyyymmdd_HHMMSS')));
writeSummaryCSV(summaryFile, summaryRows);

fprintf('\n============================================================\n');
fprintf('C001 Hidden_Mask rasterization finished. Summary written:\n%s\n', summaryFile);
fprintf('Outputs should be under:\n%s/HiddenMask_ByRiver_<res>m/<river>/\n', cfg.maskRoot);
fprintf('============================================================\n');

end

%% ============================================================
%  Local functions
% ============================================================

function cfg = defaultConfig()
    cfg.prRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';
    cfg.maskRoot = '/tank/data/SFS/xinyis/data/bathymetry/BahtyMask_Corage_Manual_Finial_Draw';
    cfg.maskShpRoot = fullfile(cfg.maskRoot, 'Mask_ShapeFiles');
    cfg.logRoot = fullfile(cfg.maskRoot, 'Logs');

    cfg.burnValue = 1;
    cfg.backgroundValue = 0;
    cfg.maskNoData = 255;      % Important: 0 is valid background, not NoData.
    cfg.outputType = 'Byte';
    cfg.allTouched = true;
    cfg.overwrite = true;
    cfg.stopOnMissing = true;
    cfg.stopOnError = true;

    % outputMode:
    %   'single' = write one HiddenMask_<res>m.tif
    %   'tiled'  = write HiddenMask_<res>m_tiles/*.tif + HiddenMask_<res>m.vrt
    %   'auto'   = single for smaller grids, tiled for very large grids
    cfg.outputMode = 'auto';
    cfg.autoTilePixelThreshold = 2.0e9;
    cfg.tileSize = 10000;

    % Reproject the shapefile to the target bathy SRS before rasterizing.
    % This is safer than relying on gdal_rasterize to infer transformations.
    cfg.reprojectShp = true;
end

function cfg = parseOptions(cfg, varargin)
    if mod(numel(varargin), 2) ~= 0
        error('Options must be name-value pairs.');
    end
    for i = 1:2:numel(varargin)
        key = lower(char(varargin{i}));
        val = varargin{i+1};
        switch key
            case 'processedroot'
                cfg.prRoot = char(val);
            case 'maskroot'
                cfg.maskRoot = char(val);
                cfg.maskShpRoot = fullfile(cfg.maskRoot, 'Mask_ShapeFiles');
                cfg.logRoot = fullfile(cfg.maskRoot, 'Logs');
            case 'maskshproot'
                cfg.maskShpRoot = char(val);
            case 'outputmode'
                cfg.outputMode = lower(char(val));
            case 'autotilepixelthreshold'
                cfg.autoTilePixelThreshold = double(val);
            case 'tilesize'
                cfg.tileSize = double(val);
            case 'alltouched'
                cfg.allTouched = logical(val);
            case 'overwrite'
                cfg.overwrite = logical(val);
            case 'stoponmissing'
                cfg.stopOnMissing = logical(val);
            case 'stoponerror'
                cfg.stopOnError = logical(val);
            case 'reprojectshp'
                cfg.reprojectShp = logical(val);
            otherwise
                error('Unknown option: %s', key);
        end
    end

    if ~ismember(lower(cfg.outputMode), {'auto','single','tiled'})
        error('outputMode must be auto, single, or tiled.');
    end
end

function rivers = selectedRivers()
    rivers = { ...
        'BadgerFinNull'
        'Estabrook_Combined'
        'KewaFix2Null'
        'Kletzch_Combined_UpMax3Null'
        'CA_KlamathRiver_TopoBathy_2018_D18'
        'CO_UpperColorado_Topobathy_1_2020'
        'MD_PotomacRiver_Bathy_2019'
        'NE_Niobrara_Topobathy_2018'
        'OR_MKRC_Topobathy_2021'
        'OR_SantiamRiverTB_Topobathy_1_D23'
        'WA_ChehalisRiverTB_Topobathy_1_D23'
        'WA_Nisqually_Bathymetric_2020'
        };
end

function printRiverList(rivers)
    fprintf('\nSelected valid rivers for C001 HiddenMask rasterization:\n');
    for i = 1:numel(rivers)
        fprintf('  %2d. %s\n', i, rivers{i});
    end
    fprintf('\n');
end

function setupGdalPaths()
    try
        cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
        GDALLoad();
    catch ME
        fprintf('[WARN] GDALLoad setup skipped or failed: %s\n', ME.message);
    end
    try
        addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
    catch
    end
end

function ensureDir(d)
    if exist(d, 'dir') ~= 7
        mkdir(d);
    end
end

function modeUsed = chooseOutputMode(cfg, info, resM) %#ok<INUSD>
    switch lower(cfg.outputMode)
        case 'single'
            modeUsed = 'single';
        case 'tiled'
            modeUsed = 'tiled';
        otherwise
            nPix = double(info.rows) * double(info.cols);
            if nPix > cfg.autoTilePixelThreshold
                modeUsed = 'tiled';
            else
                modeUsed = 'single';
            end
    end
end

function info = getRasterInfoGDAL(refGrid, jsonFile)
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
    if status ~= 0 || exist(srsWkt, 'file') ~= 2 || dir(srsWkt).bytes == 0
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

function shpUse = prepareShapefileForTargetSRS(shp, srsWkt, outRoot, cfg)
    if ~cfg.reprojectShp
        shpUse = shp;
        return;
    end

    reprojDir = fullfile(outRoot, '_HiddenMask_reproj_to_bathy_srs');
    ensureDir(reprojDir);
    shpUse = fullfile(reprojDir, 'Hidden_Mask_reproj.shp');

    if exist(shpUse, 'file') == 2 && ~cfg.overwrite
        return;
    end

    deleteShapefile(shpUse);

    cmd = sprintf('ogr2ogr -overwrite -t_srs %s %s %s', qpath(srsWkt), qpath(shpUse), qpath(shp));
    fprintf('[Reproject SHP]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0 || exist(shpUse, 'file') ~= 2
        error('ogr2ogr failed to reproject Hidden_Mask.shp: %s', msg);
    end
end

function rasterizeHiddenMaskSingle(shpUse, srsWkt, info, outTif, cfg)
    if exist(outTif, 'file') == 2
        if cfg.overwrite
            deleteIfExists(outTif);
        else
            fprintf('[SKIP] Existing output: %s\n', outTif);
            return;
        end
    end

    allTouchedFlag = '';
    if cfg.allTouched
        allTouchedFlag = '-at ';
    end

    cmd = sprintf([ ...
        'gdal_rasterize ', ...
        '-burn %d ', ...
        '-ot %s ', ...
        '-init %d ', ...
        '-a_nodata %d ', ...
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
        cfg.burnValue, cfg.outputType, cfg.backgroundValue, cfg.maskNoData, ...
        qpath(srsWkt), ...
        info.xmin, info.ymin, info.xmax, info.ymax, ...
        info.cols, info.rows, ...
        allTouchedFlag, ...
        qpath(shpUse), qpath(outTif));

    fprintf('[Rasterize single]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0
        error('gdal_rasterize single failed: %s', msg);
    end
end

function buildVRTFromSingleTif(outTif, outVrt, cfg)
    if exist(outVrt, 'file') == 2
        if cfg.overwrite
            deleteIfExists(outVrt);
        else
            return;
        end
    end
    cmd = sprintf('gdalbuildvrt -overwrite -srcnodata %d -vrtnodata %d %s %s', ...
        cfg.maskNoData, cfg.maskNoData, qpath(outVrt), qpath(outTif));
    fprintf('[Build VRT]\n%s\n', cmd);
    [status, msg] = system(cmd, '-echo');
    if status ~= 0
        error('gdalbuildvrt failed: %s', msg);
    end
end

function outVrt = rasterizeHiddenMaskTiled(shpUse, srsWkt, info, outRoot, resM, cfg)
    tileDir = fullfile(outRoot, sprintf('HiddenMask_%dm_tiles', resM));
    ensureDir(tileDir);

    outVrt = fullfile(outRoot, sprintf('HiddenMask_%dm.vrt', resM));
    tileList = fullfile(tileDir, sprintf('HiddenMask_%dm_tile_list.txt', resM));

    if cfg.overwrite
        oldTiles = dir(fullfile(tileDir, sprintf('HiddenMask_%dm_tile_*.tif', resM)));
        for i = 1:numel(oldTiles)
            delete(fullfile(oldTiles(i).folder, oldTiles(i).name));
        end
        if exist(tileList, 'file') == 2; delete(tileList); end
        if exist(outVrt, 'file') == 2; delete(outVrt); end
    elseif exist(outVrt, 'file') == 2
        fprintf('[SKIP] Existing tiled VRT: %s\n', outVrt);
        return;
    end

    dx = (info.xmax - info.xmin) / info.cols;
    dy = (info.ymax - info.ymin) / info.rows;

    allTouchedFlag = '';
    if cfg.allTouched
        allTouchedFlag = '-at ';
    end

    fid = fopen(tileList, 'w');
    if fid < 0
        error('Cannot open tile list for writing: %s', tileList);
    end

    nTiles = 0;
    fprintf('[Rasterize tiled] tileSize=%d\n', cfg.tileSize);

    for r0 = 0:cfg.tileSize:(info.rows-1)
        r1 = min(r0 + cfg.tileSize, info.rows);
        tileRows = r1 - r0;

        yMaxTile = info.ymax - r0 * dy;
        yMinTile = info.ymax - r1 * dy;

        for c0 = 0:cfg.tileSize:(info.cols-1)
            c1 = min(c0 + cfg.tileSize, info.cols);
            tileCols = c1 - c0;

            xMinTile = info.xmin + c0 * dx;
            xMaxTile = info.xmin + c1 * dx;

            tileName = sprintf('HiddenMask_%dm_tile_r%06d_c%06d.tif', resM, r0, c0);
            tilePath = fullfile(tileDir, tileName);

            cmd = sprintf([ ...
                'gdal_rasterize ', ...
                '-burn %d ', ...
                '-ot %s ', ...
                '-init %d ', ...
                '-a_nodata %d ', ...
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
                cfg.burnValue, cfg.outputType, cfg.backgroundValue, cfg.maskNoData, ...
                qpath(srsWkt), ...
                xMinTile, yMinTile, xMaxTile, yMaxTile, ...
                tileCols, tileRows, ...
                allTouchedFlag, ...
                qpath(shpUse), qpath(tilePath));

            fprintf('[Tile] r=%d:%d c=%d:%d\n', r0, r1, c0, c1);
            [status, msg] = system(cmd, '-echo');
            if status ~= 0
                fclose(fid);
                error('Tile rasterize failed: %s\n%s', tilePath, msg);
            end

            fprintf(fid, '%s\n', tilePath);
            nTiles = nTiles + 1;
        end
    end

    fclose(fid);

    cmdVRT = sprintf('gdalbuildvrt -overwrite -srcnodata %d -vrtnodata %d -input_file_list %s %s', ...
        cfg.maskNoData, cfg.maskNoData, qpath(tileList), qpath(outVrt));
    fprintf('[Build tiled VRT]\n%s\n', cmdVRT);
    [statusVRT, msgVRT] = system(cmdVRT, '-echo');
    if statusVRT ~= 0
        error('gdalbuildvrt tiled failed: %s', msgVRT);
    end

    fprintf('[Rasterize tiled] Done. nTiles=%d\n', nTiles);
end

function printRasterInfo(rasterPath)
    if exist(rasterPath, 'file') ~= 2
        fprintf('[WARN] Cannot print raster info; missing: %s\n', rasterPath);
        return;
    end
    cmd = sprintf('gdalinfo %s | grep -E "Size is|Pixel Size|Upper Left|Lower Right|NoData"', qpath(rasterPath));
    system(cmd);
end

function quickMaskStats(rasterPath, cfg)
    cmd = sprintf('gdalinfo -approx_stats %s | grep -E "Minimum|Maximum|Mean|StdDev|NoData|STATISTICS"', qpath(rasterPath));
    fprintf('[Quick mask stats]\n');
    system(cmd);
    fprintf('Expected value convention: hidden=1, background=0, nodata(meta)=%d\n', cfg.maskNoData);
end

function deleteIfExists(path0)
    [folder, base, ext] = fileparts(path0);
    if strcmpi(ext, '.shp')
        exts = {'.shp','.shx','.dbf','.prj','.cpg','.qpj','.sbn','.sbx','.shp.xml'};
    elseif strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
        exts = {ext, [ext '.aux.xml'], '.aux.xml', '.ovr'};
    elseif strcmpi(ext, '.vrt')
        exts = {'.vrt'};
    else
        exts = {ext};
    end
    for i = 1:numel(exts)
        f = fullfile(folder, [base exts{i}]);
        if exist(f, 'file') == 2
            delete(f);
        end
    end
end

function deleteShapefile(shp)
    deleteIfExists(shp);
end

function s = qpath(x)
    s = ['"', char(x), '"'];
end

function T = rowsToTable(rows)
    if isempty(rows)
        T = table();
        return;
    end
    T = cell2table(rows, 'VariableNames', {'river','res_m','refGrid','hiddenShp','outTif','outVrt','status','message'});
end

function writeSummaryCSV(summaryFile, rows)
    ensureDir(fileparts(summaryFile));
    fid = fopen(summaryFile, 'w');
    if fid < 0
        warning('Cannot write summary CSV: %s', summaryFile);
        return;
    end
    fprintf(fid, 'river,res_m,refGrid,hiddenShp,outTif,outVrt,status,message\n');
    for i = 1:size(rows,1)
        fprintf(fid, '"%s",%g,"%s","%s","%s","%s","%s","%s"\n', ...
            rows{i,1}, rows{i,2}, rows{i,3}, rows{i,4}, rows{i,5}, rows{i,6}, rows{i,7}, strrep(rows{i,8}, '"', ''''));
    end
    fclose(fid);
end

function s = safeName(s)
    s = regexprep(s, '[^A-Za-z0-9_\-]+', '_');
    if numel(s) > 120
        s = s(1:120);
    end
end
