%% ============================================================
%  OR_MKRC_Topobathy_2021 unit fix: elevation feet -> meters
%  IMPORTANT:
%    Original horizontal grid is 2 ft, not 1 m.
%    This section only converts elevation values from feet to meters.
%    It does NOT resample horizontal resolution to 1 m.
% ============================================================

% 垂直的feet转m

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

SrcRoot = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
DstRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/';

targetRiver = 'OR_MKRC_Topobathy_2021';

globalND    = -999999;
feetToMeter = 0.3048;
tile        = 2048;

srcFolder = fullfile(SrcRoot, targetRiver);
listFile  = fullfile(srcFolder, 'Filelist.txt');

if exist(listFile, 'file') ~= 2
    error('Filelist.txt not found: %s', listFile);
end

dstFolder = fullfile(DstRoot, targetRiver);
if exist(dstFolder, 'dir') ~= 7
    mkdir(dstFolder);
end

% Clear naming:
% 2ft_rawElev_ft  : horizontal grid = 2 ft, elevation = feet
% 2ft_elev_m      : horizontal grid = 2 ft, elevation = meters
rawFeetVRT = fullfile(dstFolder, 'Bathy_2ft_rawElev_ft.vrt');
meter2ftTif = fullfile(dstFolder, 'Bathy_2ft_elev_m.tif');
meter2ftVRT = fullfile(dstFolder, 'Bathy_2ft_elev_m.vrt');

% 1) Build raw 2ft feet-elevation VRT
cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
               'export PROJ_NETWORK=ON; ' ...
               'gdalbuildvrt -overwrite ' ...
               '-input_file_list "%s" "%s"'], ...
               listFile, rawFeetVRT);

fprintf('\nBuild raw 2ft feet-elevation VRT:\n%s\n', cmd);
status = system(cmd);

if status ~= 0
    error('gdalbuildvrt failed for %s:\n%s', targetRiver, cmd);
end

[~, rows, cols, geoTrans, proj, dataType, nd_raw] = RasterInfo(rawFeetVRT);

fprintf('\nRaw VRT info:\n');
fprintf('  rows = %d, cols = %d\n', rows, cols);
fprintf('  raw nodata = %g\n', nd_raw);

if isempty(nd_raw) || ~isfinite(nd_raw)
    nd_raw = globalND;
end

% 2) Convert valid elevation pixels from feet to meters
totalTiles = ceil(rows / tile) * ceil(cols / tile);
tileCount  = 0;

minBefore = inf;
maxBefore = -inf;
minAfter  = inf;
maxAfter  = -inf;

for rLocal = 1:tile:rows
    rr = min(tile, rows - rLocal + 1);

    for cLocal = 1:tile:cols
        cc = min(tile, cols - cLocal + 1);

        A = double(ReadRaster(rawFeetVRT, rLocal, cLocal, rr, cc));

        invalid = isnan(A) | ~isfinite(A) | (A == nd_raw) | (A == globalND);
        valid   = ~invalid;

        if any(valid(:))
            minBefore = min(minBefore, min(A(valid)));
            maxBefore = max(maxBefore, max(A(valid)));
        end

        A(valid)   = A(valid) * feetToMeter;
        A(invalid) = globalND;

        if any(valid(:))
            minAfter = min(minAfter, min(A(valid)));
            maxAfter = max(maxAfter, max(A(valid)));
        end

        WriteRaster(meter2ftTif, A, geoTrans, proj, dataType, ...
                    'GTiff', globalND, rLocal, cLocal, rows, cols);

        tileCount = tileCount + 1;
        fprintf('\rUnit conversion progress: %6.2f%% (%d/%d)', ...
                100 * tileCount / totalTiles, tileCount, totalTiles);
    end
end

fprintf('\n\n[UNIT FIX DONE] %s elevation feet -> meters\n', targetRiver);
fprintf('  before ft: min = %.3f, max = %.3f\n', minBefore, maxBefore);
fprintf('  after  m : min = %.3f, max = %.3f\n', minAfter,  maxAfter);
fprintf('  output 2ft meter tif: %s\n', meter2ftTif);

% 3) Build 2ft meter VRT
cmd2 = sprintf('gdalbuildvrt -overwrite -vrtnodata %g "%s" "%s"', ...
               globalND, meter2ftVRT, meter2ftTif);

fprintf('\nBuild 2ft meter VRT:\n%s\n', cmd2);
status2 = system(cmd2);

if status2 ~= 0
    error('gdalbuildvrt failed after unit conversion:\n%s', cmd2);
end

fprintf('Final 2ft meter VRT: %s\n', meter2ftVRT);

%% 水平feet转换成m

%% ============================================================
%  OR_MKRC_Topobathy_2021: create true 1m Warped VRT
%  Source:
%    Bathy_2ft_elev_m.tif
%      - horizontal grid: 2 ft
%      - elevation unit : meter
%      - NoData         : -999999
%
%  Output:
%    Bathy_1m.vrt
%      - virtual horizontal grid: true 1 m = 3.280839895 ft
%      - elevation unit          : meter
%      - NoData                  : -999999
%
%  Important:
%    Do NOT materialize a full Bathy_1m.tif.
%    It is too large and may be killed on the submit node.
% ============================================================

clear; clc;

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';
targetRiver = 'OR_MKRC_Topobathy_2021';

globalND = -999999;

inRaster = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver, 'Bathy_2ft_elev_m.tif');
outVrt   = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver, 'Bathy_1m.vrt');

if exist(inRaster, 'file') ~= 2
    error('Missing corrected 2ft meter source raster: %s', inRaster);
end

% Remove failed materialized outputs if they exist
failedTif = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver, 'Bathy_1m.tif');
if exist(failedTif, 'file') == 2
    delete(failedTif);
end
if exist(outVrt, 'file') == 2
    delete(outVrt);
end

% Horizontal CRS unit is foot.
% Therefore, gdalwarp -tr must be in feet.
meterToFoot = 1 / 0.3048;
res_1m_in_foot = 1 * meterToFoot;

cmd = sprintf([ ...
    'gdalwarp -overwrite -of VRT ', ...
    '-tr %.9f %.9f ', ...
    '-r near ', ...
    '-srcnodata %g -dstnodata %g ', ...
    '"%s" "%s"' ], ...
    res_1m_in_foot, res_1m_in_foot, ...
    globalND, globalND, inRaster, outVrt);

fprintf('\nCreate true 1m OR_MKRC Warped VRT:\n');
fprintf('Input  : %s\n', inRaster);
fprintf('Output : %s\n', outVrt);
fprintf('Desired resolution = 1 m; gdalwarp -tr = %.9f ft\n', res_1m_in_foot);
fprintf('%s\n', cmd);

status = system(cmd);

if status ~= 0
    error('gdalwarp VRT creation failed.');
end

fprintf('\nDone true 1m VRT:\n%s\n', outVrt)

%% ============================================================
%  OR_MKRC_Topobathy_2021: rebuild Bathy_3m/5m/10m_FixND
%
%  Source:
%    Bathy_2ft_elev_m.tif
%      - horizontal grid: 2 ft
%      - elevation unit : meter
%      - NoData         : -999999
%
%  Output:
%    Bathy_3m_FixND/OR_MKRC_Topobathy_2021/Bathy_3m.tif + Bathy_3m.vrt
%    Bathy_5m_FixND/OR_MKRC_Topobathy_2021/Bathy_5m.tif + Bathy_5m.vrt
%    Bathy_10m_FixND/OR_MKRC_Topobathy_2021/Bathy_10m.tif + Bathy_10m.vrt
%
%  Important:
%    Horizontal CRS unit is foot.
%    Therefore, target resolution in meters must be converted to feet
%    before passing to gdalwarp -tr.
% ============================================================

% bathy的升尺度

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';
targetRiver = 'OR_MKRC_Topobathy_2021';

globalND = -999999;

% Use corrected 2ft source directly.
% Do not use the failed/intermediate Bathy_1m.tif.
% Bathy_1m.vrt is still kept for the 1m workflow.
inRaster = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver, 'Bathy_2ft_elev_m.tif');

if exist(inRaster, 'file') ~= 2
    error('Missing corrected 2ft meter raster: %s', inRaster);
end

targetRes_m = [3, 5, 10];
meterToFoot = 1 / 0.3048;

for j = 3%:numel(targetRes_m)

    res_m  = targetRes_m(j);
    res_ft = res_m * meterToFoot;

    outRoot = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res_m));
    outSub  = fullfile(outRoot, targetRiver);

    if exist(outSub, 'dir') ~= 7
        mkdir(outSub);
    end

    outTif = fullfile(outSub, sprintf('Bathy_%dm.tif', res_m));
    outVrt = fullfile(outSub, sprintf('Bathy_%dm.vrt', res_m));

    % Delete previous partial outputs if they exist
    if exist(outTif, 'file') == 2
        delete(outTif);
    end
    if exist(outVrt, 'file') == 2
        delete(outVrt);
    end

    cmd = sprintf([ ...
        'gdalwarp -overwrite -of GTiff ', ...
        '-tr %.9f %.9f ', ...
        '-r average ', ...
        '-ot Float32 ', ...
        '-srcnodata %g -dstnodata %g ', ...
        '-multi -wo NUM_THREADS=ALL_CPUS -wm 1024 ', ...
        '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ', ...
        '"%s" "%s"' ], ...
        res_ft, res_ft, ...
        globalND, globalND, ...
        inRaster, outTif);

    fprintf('\n[%s] Rebuild Bathy_%dm_FixND\n', targetRiver, res_m);
    fprintf('Input  : %s\n', inRaster);
    fprintf('Output : %s\n', outTif);
    fprintf('Desired resolution = %d m; gdalwarp -tr = %.9f ft\n', res_m, res_ft);
    fprintf('%s\n', cmd);

    status = system(cmd);

    if status ~= 0
        error('gdalwarp failed for %s Bathy_%dm_FixND.', targetRiver, res_m);
    end

    cmd2 = sprintf('gdalbuildvrt -overwrite -vrtnodata %g "%s" "%s"', ...
                   globalND, outVrt, outTif);

    fprintf('\nBuild VRT:\n%s\n', cmd2);
    status2 = system(cmd2);

    if status2 ~= 0
        error('gdalbuildvrt failed for %s', outVrt);
    end

    fprintf('Done: %s\n', outVrt);
end

%% 3DEP重采样

%% ============================================================
%  OR_MKRC_Topobathy_2021: resample 3DEP to corrected true 1m bathy grid
%
%  Input:
%    Bathy_1m_FixND/OR_MKRC_Topobathy_2021/Bathy_1m.vrt
%      - true 1m virtual grid
%      - horizontal CRS unit: foot
%      - elevation unit: meter
%
%    3DEP_1m_VRT/OR_MKRC_Topobathy_2021/DEM_3DEP_1m_FixND.vrt
%
%  Output:
%    3DEP_1m_ResampleClip/OR_MKRC_Topobathy_2021/
%      DEM_3DEP_1m_ResampleandClip.vrt
%
%  Purpose:
%    Force 3DEP to have exactly the same grid as corrected bathy_1m.
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

targetRiver = 'OR_MKRC_Topobathy_2021';
globalND = -999999;

bathy_vrt = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver, 'Bathy_1m.vrt');
dep3_vrt  = fullfile(rootPR, '3DEP_1m_VRT', targetRiver, 'DEM_3DEP_1m_FixND.vrt');

out_subdir = fullfile(rootPR, '3DEP_1m_ResampleClip', targetRiver);
if exist(out_subdir, 'dir') ~= 7
    mkdir(out_subdir);
end

dstFile = fullfile(out_subdir, 'DEM_3DEP_1m_ResampleandClip.vrt');

if exist(bathy_vrt, 'file') ~= 2
    error('Missing corrected bathy 1m VRT: %s', bathy_vrt);
end

if exist(dep3_vrt, 'file') ~= 2
    error('Missing 3DEP VRT: %s', dep3_vrt);
end

[~, rows, cols, geoTrans, proj, ~, ~] = RasterInfo(bathy_vrt);

xmin = geoTrans(1);
xres = geoTrans(2);
ymax = geoTrans(4);
yres = geoTrans(6);

xmax = xmin + cols * xres;
ymin = ymax + rows * yres;

proj_arg = sprintf('''%s''', proj);

fprintf('\nCorrected bathy grid info:\n');
fprintf('  rows = %d, cols = %d\n', rows, cols);
fprintf('  xmin = %.10f\n', xmin);
fprintf('  xmax = %.10f\n', xmax);
fprintf('  ymin = %.10f\n', ymin);
fprintf('  ymax = %.10f\n', ymax);
fprintf('  pixel size x = %.10f\n', xres);
fprintf('  pixel size y = %.10f\n', yres);

cmd = sprintf([ ...
    'gdalwarp -of VRT ', ...
    '-r near ', ...
    '-t_srs %s -te_srs %s ', ...
    '-te %.10f %.10f %.10f %.10f ', ...
    '-ts %d %d ', ...
    '-srcnodata %g -dstnodata %g ', ...
    '-wo INIT_DEST=NO_DATA -wo SKIP_NOSOURCE=YES ', ...
    '-overwrite ', ...
    '"%s" "%s"' ], ...
    proj_arg, proj_arg, ...
    xmin, ymin, xmax, ymax, ...
    cols, rows, ...
    globalND, globalND, ...
    dep3_vrt, dstFile);

fprintf('\nResample 3DEP to corrected OR_MKRC true 1m bathy grid:\n');
fprintf('Bathy grid : %s\n', bathy_vrt);
fprintf('3DEP input : %s\n', dep3_vrt);
fprintf('Output     : %s\n', dstFile);
fprintf('%s\n', cmd);

status = system(cmd);

if status ~= 0
    error('gdalwarp failed when resampling 3DEP to bathy grid.');
end

fprintf('\nDone: %s\n', dstFile);

%% 融合代码

%% ====== OR_MKRC only: merge corrected bathy + resampled 3DEP ======
clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); 
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder_bathy = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/';
Folder_dem   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/3DEP_1m_ResampleClip/';
OutFolder    = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_1m/';
VerifyRoot   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z001_Verify_Merge/';

targetRiver = 'OR_MKRC_Topobathy_2021';

if exist(OutFolder,'dir') ~= 7
    mkdir(OutFolder);
end

if exist(VerifyRoot,'dir') ~= 7
    mkdir(VerifyRoot);
end

globalND = -999999;
tile     = 2048;

nVerifyTiles = 2;
minFillPix   = 1000;

d = dir(Folder_bathy);
d = d([d.isdir]);
d(1:2) = [];

fprintf('Found %d rivers under Bathy folder.\n', numel(d));

for iRiver = 1:numel(d)

    name = d(iRiver).name;

    % Only rebuild OR_MKRC
    if ~strcmpi(name, targetRiver)
        continue;
    end

    verifySaved = 0;

    bathy_vrt = fullfile(Folder_bathy, name, 'Bathy_1m.vrt');
    dem_vrt   = fullfile(Folder_dem,   name, 'DEM_3DEP_1m_ResampleandClip.vrt');

    if exist(bathy_vrt,'file') ~= 2
        error('Missing bathy vrt: %s', bathy_vrt);
    end

    if exist(dem_vrt,'file') ~= 2
        error('Missing dem vrt: %s', dem_vrt);
    end

    out_subdir = fullfile(OutFolder, name);
    if exist(out_subdir,'dir') ~= 7
        mkdir(out_subdir);
    end

    tilesDir = fullfile(out_subdir, '_tiles');
    if exist(tilesDir,'dir') ~= 7
        mkdir(tilesDir);
    end

    verifyDir = fullfile(VerifyRoot, name);
    if exist(verifyDir,'dir') ~= 7
        mkdir(verifyDir);
    end

    [~, rows, cols, geoTrans, proj, dataType_bathy, ~] = RasterInfo(bathy_vrt);

    fprintf('\n[%d/%d] Start merging: %s\n', iRiver, numel(d), name);
    fprintf('  Bathy: %s\n', bathy_vrt);
    fprintf('  3DEP : %s\n', dem_vrt);
    fprintf('  rows = %d, cols = %d\n', rows, cols);

    totalTiles = ceil(rows/tile) * ceil(cols/tile);
    tileCount  = 0;

    for rLocal = 1:tile:rows

        rr = min(tile, rows - rLocal + 1);

        for cLocal = 1:tile:cols

            cc = min(tile, cols - cLocal + 1);

            absRow = rLocal;
            absCol = cLocal;

            B = double(ReadRaster(bathy_vrt, absRow, absCol, rr, cc));
            D = double(ReadRaster(dem_vrt,   absRow, absCol, rr, cc));

            C = B;

            isHoleB  = isnan(B) | ~isfinite(B) | (B == globalND);
            isValidD = isfinite(D) & ~isnan(D) & (D ~= globalND);

            mask_fill = isHoleB & isValidD;
            C(mask_fill) = D(mask_fill);

            C(~isfinite(C) | isnan(C) | (C == globalND)) = globalND;

            subgeoTrans = subTranscoef(geoTrans, absRow, absCol);
            tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));

            WriteRaster(tileTif, C, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);

            if verifySaved < nVerifyTiles

                nFill = nnz(mask_fill);

                if nFill >= minFillPix

                    tag = sprintf('Tile_r%04d_c%04d_fill%d', rLocal, cLocal, nFill);

                    outTif1 = fullfile(verifyDir, sprintf('%s_%s_Merged.tif', name, tag));
                    outTif2 = fullfile(verifyDir, sprintf('%s_%s_Bathy.tif',  name, tag));
                    outTif3 = fullfile(verifyDir, sprintf('%s_%s_3DEP.tif',   name, tag));

                    WriteRaster(outTif1, C, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);
                    WriteRaster(outTif2, B, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);
                    WriteRaster(outTif3, D, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);

                    verifySaved = verifySaved + 1;

                    fprintf('\n  [VERIFY %d/%d] nFill=%d (r=%d c=%d)\n', ...
                        verifySaved, nVerifyTiles, nFill, rLocal, cLocal);
                    fprintf('    %s\n    %s\n    %s\n', outTif1, outTif2, outTif3);
                end
            end

            tileCount = tileCount + 1;
            fprintf('\r  Merge progress: %6.2f%% (%d/%d)', ...
                100*tileCount/totalTiles, tileCount, totalTiles);

            clear B D C mask_fill
        end
    end

    fprintf('\nTile writing done.\n');

    listTxt = fullfile(out_subdir, 'tile_list.txt');

    cmdList = sprintf('find "%s" -maxdepth 1 -type f -name "tile_*.tif" | sort > "%s"', ...
        tilesDir, listTxt);

    status = system(cmdList);
    if status ~= 0
        error('Failed to build tile list. CMD=%s', cmdList);
    end

    info = dir(listTxt);
    if isempty(info) || info.bytes == 0
        error('No tiles found in %s. tile_list.txt empty.', tilesDir);
    end

    % Main output name used by B005
    outVrt = fullfile(out_subdir, 'Combined_Bathy_Priority_1m.vrt');

    cmdV = sprintf('gdalbuildvrt -overwrite -vrtnodata %g -input_file_list "%s" "%s"', ...
        globalND, listTxt, outVrt);

    fprintf('\nBuild main merged VRT:\n%s\n', cmdV);
    status = system(cmdV);
    if status ~= 0
        error('gdalbuildvrt failed: %s', cmdV);
    end

    % Optional legacy name
    outVrtLegacy = fullfile(out_subdir, sprintf('%s_Merged_1m.vrt', name));

    cmdV2 = sprintf('gdalbuildvrt -overwrite -vrtnodata %g -input_file_list "%s" "%s"', ...
        globalND, listTxt, outVrtLegacy);

    fprintf('\nBuild legacy merged VRT:\n%s\n', cmdV2);
    status2 = system(cmdV2);
    if status2 ~= 0
        warning('Legacy VRT build failed: %s', cmdV2);
    end

    fprintf('\n[%s] merged VRT done:\n%s\n', name, outVrt);
end

fprintf('\nOR_MKRC merge done.\n');


%% 融合后的升尺度

%% ============================================================
%  OR_MKRC_Topobathy_2021: upscale merged true 1m result to 3m/5m/10m
%
%  Input:
%    Bathy3DEP_Merged_Tiff_1m/OR_MKRC_Topobathy_2021/
%      Combined_Bathy_Priority_1m.vrt
%
%  Output:
%    Bathy3DEP_Merged_Tiff_3m/OR_MKRC_Topobathy_2021/
%      Combined_Bathy_Priority_3m.tif
%
%    Bathy3DEP_Merged_Tiff_5m/OR_MKRC_Topobathy_2021/
%      Combined_Bathy_Priority_5m.tif
%
%    Bathy3DEP_Merged_Tiff_10m/OR_MKRC_Topobathy_2021/
%      Combined_Bathy_Priority_10m.tif
%
%  Important:
%    The horizontal CRS unit is foot.
%    The elevation values are already in meters.
%    Therefore, gdalwarp -tr must use feet values.
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

targetRiver = 'OR_MKRC_Topobathy_2021';
globalND = -999999;

inVRT = fullfile(rootPR, 'Bathy3DEP_Merged_Tiff_1m', targetRiver, ...
                 'Combined_Bathy_Priority_1m.vrt');

if exist(inVRT, 'file') ~= 2
    error('Missing merged 1m VRT: %s', inVRT);
end

targetRes_m = [3, 5, 10];
meterToFoot = 1 / 0.3048;

for j = 3 %:numel(targetRes_m)

    res_m  = targetRes_m(j);
    res_ft = res_m * meterToFoot;

    outRoot = fullfile(rootPR, sprintf('Bathy3DEP_Merged_Tiff_%dm', res_m));
    outSub  = fullfile(outRoot, targetRiver);

    if exist(outSub, 'dir') ~= 7
        mkdir(outSub);
    end

    outTif = fullfile(outSub, sprintf('Combined_Bathy_Priority_%dm.tif', res_m));

    if exist(outTif, 'file') == 2
        delete(outTif);
    end

    cmd = sprintf([ ...
        'export GDAL_CACHEMAX=512; ' ...
        'gdalwarp -overwrite -of GTiff ', ...
        '-tr %.9f %.9f ', ...
        '-r average ', ...
        '-ot Float32 ', ...
        '-srcnodata %g -dstnodata %g ', ...
        '-multi -wo NUM_THREADS=4 -wm 512 ', ...
        '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ', ...
        '"%s" "%s"' ], ...
        res_ft, res_ft, ...
        globalND, globalND, ...
        inVRT, outTif);

    fprintf('\n[%s] Upscale merged result to %dm\n', targetRiver, res_m);
    fprintf('Input  : %s\n', inVRT);
    fprintf('Output : %s\n', outTif);
    fprintf('Desired resolution = %dm; gdalwarp -tr = %.9f ft\n', res_m, res_ft);
    fprintf('%s\n', cmd);

    status = system(cmd);

    if status ~= 0
        error('gdalwarp failed for merged %dm output.', res_m);
    end

    fprintf('Done: %s\n', outTif);
end