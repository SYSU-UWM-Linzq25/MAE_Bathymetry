%% ============================================================
%  KewaFix2Null: elevation feet -> meters
%  Horizontal CRS is already meter, pixel size is already 1m.
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

targetRiver = 'KewaFix2Null';

rawRaster = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/KewaFix2Null.tif';

outSub = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver);
if exist(outSub, 'dir') ~= 7
    mkdir(outSub);
end

outTif = fullfile(outSub, 'Bathy_1m.tif');
outVrt = fullfile(outSub, 'Bathy_1m.vrt');

globalND = -999999;
feetToMeter = 0.3048;

if exist(rawRaster, 'file') ~= 2
    error('Missing raw Kewa raster: %s', rawRaster);
end

[~, rows, cols, geoTrans, proj, dataType, nd_raw] = RasterInfo(rawRaster);

fprintf('\nRaw Kewa raster info:\n');
fprintf('  rows = %d, cols = %d\n', rows, cols);
fprintf('  raw nodata = %g\n', nd_raw);
fprintf('  pixel size = %.6f, %.6f\n', geoTrans(2), geoTrans(6));

A = double(ReadRaster(rawRaster));

invalid = isnan(A) | ~isfinite(A) | (A == nd_raw) | (A == globalND) | (A < -1e20);
valid   = ~invalid;

fprintf('\nBefore conversion:\n');
fprintf('  min ft = %.3f\n', min(A(valid)));
fprintf('  max ft = %.3f\n', max(A(valid)));
fprintf('  mean ft = %.3f\n', mean(A(valid), 'omitnan'));

A(valid)   = A(valid) * feetToMeter;
A(invalid) = globalND;

fprintf('\nAfter conversion:\n');
fprintf('  min m = %.3f\n', min(A(valid)));
fprintf('  max m = %.3f\n', max(A(valid)));
fprintf('  mean m = %.3f\n', mean(A(valid), 'omitnan'));

WriteRaster(outTif, A, geoTrans, proj, dataType, 'GTiff', globalND);

cmd = sprintf('gdalbuildvrt -overwrite -vrtnodata %g "%s" "%s"', ...
              globalND, outVrt, outTif);

fprintf('\nBuild VRT:\n%s\n', cmd);
status = system(cmd);

if status ~= 0
    error('gdalbuildvrt failed for Kewa Bathy_1m.vrt.');
end

fprintf('\nDone Kewa Bathy_1m_FixND:\n%s\n', outVrt);

%% bathy的升尺度

%% ============================================================
%  KewaFix2Null: rebuild Bathy_3m/5m/10m_FixND
%  Source:
%    Bathy_1m_FixND/KewaFix2Null/Bathy_1m.tif
%
%  Output:
%    Bathy_3m_FixND/KewaFix2Null/Bathy_3m.tif + .vrt
%    Bathy_5m_FixND/KewaFix2Null/Bathy_5m.tif + .vrt
%    Bathy_10m_FixND/KewaFix2Null/Bathy_10m.tif + .vrt
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

targetRiver = 'KewaFix2Null';
globalND = -999999;

inRaster = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver, 'Bathy_1m.tif');

if exist(inRaster, 'file') ~= 2
    error('Missing corrected Kewa Bathy_1m.tif: %s', inRaster);
end

targetRes = [3, 5, 10];

for j = 1:numel(targetRes)

    res = targetRes(j);

    outRoot = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res));
    outSub  = fullfile(outRoot, targetRiver);

    if exist(outSub, 'dir') ~= 7
        mkdir(outSub);
    end

    outTif = fullfile(outSub, sprintf('Bathy_%dm.tif', res));
    outVrt = fullfile(outSub, sprintf('Bathy_%dm.vrt', res));

    if exist(outTif, 'file') == 2
        delete(outTif);
    end
    if exist(outVrt, 'file') == 2
        delete(outVrt);
    end

    cmd = sprintf([ ...
        'gdalwarp -overwrite -of GTiff ', ...
        '-tr %d %d ', ...
        '-r average ', ...
        '-ot Float32 ', ...
        '-srcnodata %g -dstnodata %g ', ...
        '-multi -wo NUM_THREADS=4 -wm 512 ', ...
        '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ', ...
        '"%s" "%s"' ], ...
        res, res, ...
        globalND, globalND, ...
        inRaster, outTif);

    fprintf('\n[KewaFix2Null] Rebuild Bathy_%dm_FixND\n', res);
    fprintf('%s\n', cmd);

    status = system(cmd);
    if status ~= 0
        error('gdalwarp failed for Kewa Bathy_%dm.', res);
    end

    cmd2 = sprintf('gdalbuildvrt -overwrite -vrtnodata %g "%s" "%s"', ...
                   globalND, outVrt, outTif);

    fprintf('Build VRT:\n%s\n', cmd2);
    status2 = system(cmd2);

    if status2 ~= 0
        error('gdalbuildvrt failed for %s', outVrt);
    end

    fprintf('Done: %s\n', outVrt);
end


%% 3DEP的重采样

%% ============================================================
%  KewaFix2Null: resample 3DEP to corrected bathy 1m grid
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

targetRiver = 'KewaFix2Null';
globalND = -999999;

bathy_vrt = fullfile(rootPR, 'Bathy_1m_FixND', targetRiver, 'Bathy_1m.vrt');

% Milwaukee shared 3DEP source
dep3_vrt = fullfile(rootPR, '3DEP_1m_VRT', 'milwaukee_river_3DEP', 'DEM_3DEP_1m_FixND.vrt');

% fallback
if exist(dep3_vrt, 'file') ~= 2
    dep3_vrt = fullfile(rootPR, '3DEP_1m_VRT', targetRiver, 'DEM_3DEP_1m_FixND.vrt');
end

out_subdir = fullfile(rootPR, '3DEP_1m_ResampleClip', targetRiver);
if exist(out_subdir, 'dir') ~= 7
    mkdir(out_subdir);
end

dstFile = fullfile(out_subdir, 'DEM_3DEP_1m_ResampleandClip.vrt');

if exist(bathy_vrt, 'file') ~= 2
    error('Missing corrected Kewa bathy VRT: %s', bathy_vrt);
end

if exist(dep3_vrt, 'file') ~= 2
    error('Missing 3DEP VRT for Kewa: %s', dep3_vrt);
end

[~, rows, cols, geoTrans, proj, ~, ~] = RasterInfo(bathy_vrt);

xmin = geoTrans(1);
xres = geoTrans(2);
ymax = geoTrans(4);
yres = geoTrans(6);

xmax = xmin + cols * xres;
ymin = ymax + rows * yres;

proj_arg = sprintf('''%s''', proj);

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

fprintf('\nResample 3DEP to corrected Kewa bathy grid:\n');
fprintf('Bathy grid : %s\n', bathy_vrt);
fprintf('3DEP input : %s\n', dep3_vrt);
fprintf('Output     : %s\n', dstFile);
fprintf('%s\n', cmd);

status = system(cmd);

if status ~= 0
    error('gdalwarp failed when resampling 3DEP to Kewa bathy grid.');
end

fprintf('\nDone: %s\n', dstFile);

%% 融合

%% ============================================================
%  KewaFix2Null: merge corrected bathy + resampled 3DEP at 1m
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder_bathy = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/';
Folder_dem   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/3DEP_1m_ResampleClip/';
OutFolder    = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_1m/';
VerifyRoot   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z001_Verify_Merge/';

targetRiver = 'KewaFix2Null';

globalND = -999999;
tile     = 2048;

nVerifyTiles = 2;
minFillPix   = 1000;
verifySaved  = 0;

bathy_vrt = fullfile(Folder_bathy, targetRiver, 'Bathy_1m.vrt');
dem_vrt   = fullfile(Folder_dem,   targetRiver, 'DEM_3DEP_1m_ResampleandClip.vrt');

if exist(bathy_vrt,'file') ~= 2
    error('Missing bathy vrt: %s', bathy_vrt);
end

if exist(dem_vrt,'file') ~= 2
    error('Missing dem vrt: %s', dem_vrt);
end

out_subdir = fullfile(OutFolder, targetRiver);
if exist(out_subdir,'dir') ~= 7
    mkdir(out_subdir);
end

tilesDir = fullfile(out_subdir, '_tiles');
if exist(tilesDir,'dir') ~= 7
    mkdir(tilesDir);
end

verifyDir = fullfile(VerifyRoot, targetRiver);
if exist(verifyDir,'dir') ~= 7
    mkdir(verifyDir);
end

[~, rows, cols, geoTrans, proj, dataType_bathy, ~] = RasterInfo(bathy_vrt);

fprintf('\nStart merging: %s\n', targetRiver);
fprintf('  Bathy: %s\n', bathy_vrt);
fprintf('  3DEP : %s\n', dem_vrt);
fprintf('  rows = %d, cols = %d\n', rows, cols);

totalTiles = ceil(rows/tile) * ceil(cols/tile);
tileCount  = 0;

for rLocal = 1:tile:rows

    rr = min(tile, rows - rLocal + 1);

    for cLocal = 1:tile:cols

        cc = min(tile, cols - cLocal + 1);

        B = double(ReadRaster(bathy_vrt, rLocal, cLocal, rr, cc));
        D = double(ReadRaster(dem_vrt,   rLocal, cLocal, rr, cc));

        C = B;

        isHoleB  = isnan(B) | ~isfinite(B) | (B == globalND);
        isValidD = isfinite(D) & ~isnan(D) & (D ~= globalND);

        mask_fill = isHoleB & isValidD;
        C(mask_fill) = D(mask_fill);

        C(~isfinite(C) | isnan(C) | (C == globalND)) = globalND;

        subgeoTrans = subTranscoef(geoTrans, rLocal, cLocal);
        tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));

        WriteRaster(tileTif, C, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);

        if verifySaved < nVerifyTiles
            nFill = nnz(mask_fill);

            if nFill >= minFillPix
                tag = sprintf('Tile_r%04d_c%04d_fill%d', rLocal, cLocal, nFill);

                outTif1 = fullfile(verifyDir, sprintf('%s_%s_Merged.tif', targetRiver, tag));
                outTif2 = fullfile(verifyDir, sprintf('%s_%s_Bathy.tif',  targetRiver, tag));
                outTif3 = fullfile(verifyDir, sprintf('%s_%s_3DEP.tif',   targetRiver, tag));

                WriteRaster(outTif1, C, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);
                WriteRaster(outTif2, B, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);
                WriteRaster(outTif3, D, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);

                verifySaved = verifySaved + 1;
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

outVrt = fullfile(out_subdir, 'Combined_Bathy_Priority_1m.vrt');

cmdV = sprintf('gdalbuildvrt -overwrite -vrtnodata %g -input_file_list "%s" "%s"', ...
    globalND, listTxt, outVrt);

fprintf('\nBuild main merged VRT:\n%s\n', cmdV);
status = system(cmdV);
if status ~= 0
    error('gdalbuildvrt failed: %s', cmdV);
end

outVrtLegacy = fullfile(out_subdir, sprintf('%s_Merged_1m.vrt', targetRiver));
cmdV2 = sprintf('gdalbuildvrt -overwrite -vrtnodata %g -input_file_list "%s" "%s"', ...
    globalND, listTxt, outVrtLegacy);
system(cmdV2);

fprintf('\n[%s] merged VRT done:\n%s\n', targetRiver, outVrt);

%% merge以后的升尺度

%% ============================================================
%  KewaFix2Null: upscale merged 1m result to 3m/5m/10m
%
%  Input:
%    Bathy3DEP_Merged_Tiff_1m/KewaFix2Null/
%      Combined_Bathy_Priority_1m.vrt
%
%  Output:
%    Bathy3DEP_Merged_Tiff_3m/KewaFix2Null/
%      Combined_Bathy_Priority_3m.tif
%
%    Bathy3DEP_Merged_Tiff_5m/KewaFix2Null/
%      Combined_Bathy_Priority_5m.tif
%
%    Bathy3DEP_Merged_Tiff_10m/KewaFix2Null/
%      Combined_Bathy_Priority_10m.tif
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

targetRiver = 'KewaFix2Null';
globalND = -999999;

inVRT = fullfile(rootPR, 'Bathy3DEP_Merged_Tiff_1m', targetRiver, ...
                 'Combined_Bathy_Priority_1m.vrt');

if exist(inVRT, 'file') ~= 2
    error('Missing merged 1m VRT: %s', inVRT);
end

targetRes = [3, 5, 10];

for j = 1:numel(targetRes)

    res = targetRes(j);

    outRoot = fullfile(rootPR, sprintf('Bathy3DEP_Merged_Tiff_%dm', res));
    outSub  = fullfile(outRoot, targetRiver);

    if exist(outSub, 'dir') ~= 7
        mkdir(outSub);
    end

    outTif = fullfile(outSub, sprintf('Combined_Bathy_Priority_%dm.tif', res));

    if exist(outTif, 'file') == 2
        delete(outTif);
    end

    cmd = sprintf([ ...
        'gdalwarp -overwrite -of GTiff ', ...
        '-tr %d %d ', ...
        '-r average ', ...
        '-ot Float32 ', ...
        '-srcnodata %g -dstnodata %g ', ...
        '-multi -wo NUM_THREADS=4 -wm 512 ', ...
        '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ', ...
        '"%s" "%s"' ], ...
        res, res, ...
        globalND, globalND, ...
        inVRT, outTif);

    fprintf('\n[KewaFix2Null] Upscale merged result to %dm\n', res);
    fprintf('Input  : %s\n', inVRT);
    fprintf('Output : %s\n', outTif);
    fprintf('%s\n', cmd);

    status = system(cmd);

    if status ~= 0
        error('gdalwarp failed for Kewa merged %dm output.', res);
    end

    fprintf('Done: %s\n', outTif);
end
