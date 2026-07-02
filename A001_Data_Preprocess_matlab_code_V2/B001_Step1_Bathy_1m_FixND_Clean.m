%% B001_Step1_Bathy_1m_FixND_Clean.m
% 2026-07-01
%
% Purpose
%   Build a clean and reproducible bathymetry 1 m preprocessing stage.
%
% Output products
%   1) Processed_Results/Bathy_1m/<river>/Bathy_1m.vrt
%      - merge-only VRT, produced from raw Filelist.txt or Milwaukee raw tif.
%      - no unit conversion, no pixel NoData rewriting.
%
%   2) Processed_Results/Bathy_1m_FixND/<river>/Bathy_1m.vrt
%      - final bathy grid for later mask alignment and 3DEP alignment.
%      - unified NoData = -999999.
%      - no global zero-to-NoData conversion.
%      - special unit corrections are handled here:
%           KewaFix2Null: elevation feet -> meters, horizontal grid already 1 m.
%           OR_MKRC_Topobathy_2021: elevation feet -> meters, horizontal grid 2 ft -> true 1 m VRT.
%
% Important design
%   - Only process the 12 selected training rivers.
%   - Raw folders are input only. Outputs are written only under Processed_Results.
%   - MD and CA supplemental data are included by their updated Filelist.txt.
%     For CA, Filelist.txt should contain the reprojected TL2018 files:
%       TL2018_reproj_to_main_srs/*_reproj.tif
%   - CA has an extra unit/NoData QA warning because the newly supplemented
%     CA bathy may have similar NoData/unit issues as the original CA source.
%
% Run from MATLAB on the cluster after confirming:
%   USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/Filelist.txt
%   USGS_3DEP_bathymetry_DEM/CA_KlamathRiver_TopoBathy_2018_D18/Filelist.txt
%
% Later steps should use only:
%   Processed_Results/Bathy_1m_FixND/<river>/Bathy_1m.vrt

clear; clc;

%% -------------------- Configuration --------------------
cfg.rawRoot = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM';
cfg.prRoot  = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

cfg.bathy1mRoot   = fullfile(cfg.prRoot, 'Bathy_1m');
cfg.bathyFixRoot  = fullfile(cfg.prRoot, 'Bathy_1m_FixND');
cfg.logRoot       = fullfile(cfg.prRoot, 'Logs');

cfg.globalND = -999999;
cfg.f32MinND = -3.4028235e38;
cfg.feetToMeter = 0.3048;
cfg.tileSize = 2048;
cfg.overwrite = true;

% Switches. For this current stage, keep both true.
cfg.buildMergeOnly = true;
cfg.buildFixND     = true;

% If true, stop when CA valid range strongly suggests possible feet units.
% Default false: only print warning and write summary CSV, because range-based
% unit detection is a QA flag rather than proof.
cfg.errorOnCAUnitSuspect = false;

setupGdalPaths();
ensureDir(cfg.bathy1mRoot);
ensureDir(cfg.bathyFixRoot);
ensureDir(cfg.logRoot);

%% -------------------- Selected rivers only --------------------
USGSRivers = { ...
    'CA_KlamathRiver_TopoBathy_2018_D18'
    'CO_UpperColorado_Topobathy_1_2020'
    'MD_PotomacRiver_Bathy_2019'
    'NE_Niobrara_Topobathy_2018'
    'OR_MKRC_Topobathy_2021'
    'OR_SantiamRiverTB_Topobathy_1_D23'
    'WA_ChehalisRiverTB_Topobathy_1_D23'
    'WA_Nisqually_Bathymetric_2020'
    };

MilwaukeeDirect = { ...
    'BadgerFinNull'
    'Estabrook_Combined'
    'KewaFix2Null'
    };

MilwaukeeCombined = { ...
    'Kletzch_Combined_UpMax3Null'
    };

% Normal meter-unit rivers: only need unified output NoData metadata.
NormalFixNDRivers = { ...
    'CO_UpperColorado_Topobathy_1_2020'
    'MD_PotomacRiver_Bathy_2019'
    'NE_Niobrara_Topobathy_2018'
    'OR_SantiamRiverTB_Topobathy_1_D23'
    'WA_ChehalisRiverTB_Topobathy_1_D23'
    'WA_Nisqually_Bathymetric_2020'
    };

% Rivers with known float32-min NoData issue. These are canonicalized by tiles,
% so f32-min / NaN / Inf / source NoData / -999999 / very negative sentinels
% are all rewritten to -999999.
F32MinFixNDRivers = { ...
    'CA_KlamathRiver_TopoBathy_2018_D18'
    'BadgerFinNull'
    'Estabrook_Combined'
    'Kletzch_Combined_UpMax3Null'
    };

%% ============================================================
%  STEP 1. Build merge-only Bathy_1m VRTs under Processed_Results
% ============================================================
if cfg.buildMergeOnly
    fprintf('\n============================================================\n');
    fprintf('STEP 1: Build merge-only Bathy_1m VRTs\n');
    fprintf('Output root: %s\n', cfg.bathy1mRoot);
    fprintf('============================================================\n');

    % ---- 1A. USGS rivers from raw Filelist.txt ----
    for i = 1:numel(USGSRivers)
        river = USGSRivers{i};
        srcFolder = fullfile(cfg.rawRoot, river);
        listFile  = fullfile(srcFolder, 'Filelist.txt');
        outSub    = fullfile(cfg.bathy1mRoot, river);
        outVrt    = fullfile(outSub, 'Bathy_1m.vrt');

        fprintf('\n[%d/%d] USGS merge-only: %s\n', i, numel(USGSRivers), river);
        if exist(listFile, 'file') ~= 2
            error('Missing Filelist.txt for %s: %s', river, listFile);
        end
        ensureDir(outSub);
        buildVRTFromFilelist(listFile, outVrt, cfg.overwrite);
        printRasterInfo(outVrt);

        if strcmp(river, 'CA_KlamathRiver_TopoBathy_2018_D18')
            nTL = countTextInFile(listFile, 'TL2018_reproj_to_main_srs');
            fprintf('[CA QA] Filelist contains %d TL2018 reprojected files. Expected about 391.\n', nTL);
            if nTL == 0
                warning('[CA QA] CA Filelist.txt does not contain TL2018_reproj_to_main_srs. Supplemental CA data may be missing.');
            end
        end
    end

    % ---- 1B. Milwaukee direct products ----
    rawMilwaukee = fullfile(cfg.rawRoot, 'milwaukee_river_3DEP');
    for i = 1:numel(MilwaukeeDirect)
        river = MilwaukeeDirect{i};
        srcTif = fullfile(rawMilwaukee, [river, '.tif']);
        outSub = fullfile(cfg.bathy1mRoot, river);
        outVrt = fullfile(outSub, 'Bathy_1m.vrt');

        fprintf('\n[%d/%d] Milwaukee direct merge-only: %s\n', i, numel(MilwaukeeDirect), river);
        if exist(srcTif, 'file') ~= 2
            error('Missing Milwaukee source tif for %s: %s', river, srcTif);
        end
        ensureDir(outSub);
        buildVRTFromRasterList({srcTif}, outVrt, cfg.overwrite);
        printRasterInfo(outVrt);
    end

    % ---- 1C. Milwaukee Kletzch combined ----
    river = MilwaukeeCombined{1};
    Kletzch_proj = fullfile(rawMilwaukee, 'Kletzch_proj.tif');
    UpMax3Null   = fullfile(rawMilwaukee, 'UpMax3Null.tif');
    outSub = fullfile(cfg.bathy1mRoot, river);
    outVrt = fullfile(outSub, 'Bathy_1m.vrt');

    fprintf('\n[Milwaukee combined merge-only] %s\n', river);
    if exist(Kletzch_proj, 'file') ~= 2; error('Missing source: %s', Kletzch_proj); end
    if exist(UpMax3Null,   'file') ~= 2; error('Missing source: %s', UpMax3Null); end
    ensureDir(outSub);

    % Keep original priority: Kletzch first, UpMax3Null second.
    buildVRTFromRasterList({Kletzch_proj, UpMax3Null}, outVrt, cfg.overwrite);
    printRasterInfo(outVrt);
end

%% ============================================================
%  STEP 2. Build Bathy_1m_FixND under Processed_Results
% ============================================================
if cfg.buildFixND
    fprintf('\n============================================================\n');
    fprintf('STEP 2: Build Bathy_1m_FixND\n');
    fprintf('Output root: %s\n', cfg.bathyFixRoot);
    fprintf('============================================================\n');

    % ---- 2A. Normal meter-unit rivers: VRT-level unified NoData ----
    for i = 1:numel(NormalFixNDRivers)
        river = NormalFixNDRivers{i};
        srcVrt = fullfile(cfg.bathy1mRoot, river, 'Bathy_1m.vrt');
        outSub = fullfile(cfg.bathyFixRoot, river);
        outVrt = fullfile(outSub, 'Bathy_1m.vrt');

        fprintf('\n[%d/%d] Normal FixND VRT: %s\n', i, numel(NormalFixNDRivers), river);
        if exist(srcVrt, 'file') ~= 2
            error('Missing merge-only Bathy_1m.vrt for %s: %s', river, srcVrt);
        end
        ensureDir(outSub);
        buildFixNDWarpVRT(srcVrt, outVrt, cfg.globalND, cfg.overwrite);
        printRasterInfo(outVrt);
    end

    % ---- 2B. f32-min NoData rivers: tiled canonicalization ----
    for i = 1:numel(F32MinFixNDRivers)
        river = F32MinFixNDRivers{i};
        srcVrt = fullfile(cfg.bathy1mRoot, river, 'Bathy_1m.vrt');
        outSub = fullfile(cfg.bathyFixRoot, river);

        fprintf('\n[%d/%d] f32-min NoData canonicalization: %s\n', i, numel(F32MinFixNDRivers), river);
        if exist(srcVrt, 'file') ~= 2
            error('Missing merge-only Bathy_1m.vrt for %s: %s', river, srcVrt);
        end

        summary = canonicalizeScaleTiled(srcVrt, outSub, 'Bathy_1m', cfg, 1.0, ...
            'none_no_unit_conversion__fix_f32min_nodata', ...
            true, false, river);

        if strcmp(river, 'CA_KlamathRiver_TopoBathy_2018_D18')
            checkCAUnitWarning(summary, cfg);
        end
        printRasterInfo(fullfile(outSub, 'Bathy_1m.vrt'));
    end

    % ---- 2C. KewaFix2Null: elevation feet -> meters, grid already 1 m ----
    fprintf('\n[Special unit fix] KewaFix2Null: elevation feet -> meters\n');
    buildKewaFixND_1m(cfg);
    printRasterInfo(fullfile(cfg.bathyFixRoot, 'KewaFix2Null', 'Bathy_1m.vrt'));

    % ---- 2D. OR_MKRC: elevation feet -> meters, 2 ft horizontal grid -> true 1 m VRT ----
    fprintf('\n[Special unit fix] OR_MKRC_Topobathy_2021: elevation ft -> m, horizontal 2 ft -> true 1 m\n');
    buildORMKRCFixND_1m(cfg);
    printRasterInfo(fullfile(cfg.bathyFixRoot, 'OR_MKRC_Topobathy_2021', 'Bathy_1m.vrt'));
end

%% -------------------- Final check --------------------
allFinalRivers = { ...
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

fprintf('\n============================================================\n');
fprintf('FINAL CHECK: expected Bathy_1m_FixND outputs\n');
fprintf('============================================================\n');

for i = 1:numel(allFinalRivers)
    river = allFinalRivers{i};
    f = fullfile(cfg.bathyFixRoot, river, 'Bathy_1m.vrt');
    if exist(f, 'file') == 2
        fprintf('[OK]   %s\n', river);
    else
        fprintf('[MISS] %s -> %s\n', river, f);
    end
end

fprintf('\nDone. Next steps should use:\n  %s/<river>/Bathy_1m.vrt\n', cfg.bathyFixRoot);

%% ============================================================
%  Local functions
% ============================================================

function setupGdalPaths()
    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
    GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
end

function ensureDir(p)
    if exist(p, 'dir') ~= 7
        mkdir(p);
    end
end

function s = q(p)
    s = ['"', char(p), '"'];
end

function runCmd(cmd, errMsg)
    fprintf('CMD:\n%s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('%s\nCMD failed:\n%s', errMsg, cmd);
    end
end

function buildVRTFromFilelist(listFile, outVrt, overwrite)
    if exist(outVrt, 'file') == 2 && ~overwrite
        fprintf('[SKIP] Existing VRT: %s\n', outVrt);
        return;
    end
    cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
                   'export PROJ_NETWORK=ON; ' ...
                   'gdalbuildvrt -overwrite -input_file_list %s %s'], ...
                   q(listFile), q(outVrt));
    runCmd(cmd, 'gdalbuildvrt failed');
end

function buildVRTFromRasterList(rasters, outVrt, overwrite)
    if exist(outVrt, 'file') == 2 && ~overwrite
        fprintf('[SKIP] Existing VRT: %s\n', outVrt);
        return;
    end
    listFile = [tempname, '.txt'];
    fid = fopen(listFile, 'w');
    if fid < 0; error('Cannot write temp list: %s', listFile); end
    for i = 1:numel(rasters)
        fprintf(fid, '%s\n', rasters{i});
    end
    fclose(fid);
    buildVRTFromFilelist(listFile, outVrt, overwrite);
    if exist(listFile, 'file') == 2
        delete(listFile);
    end
end

function buildFixNDWarpVRT(srcVrt, outVrt, globalND, overwrite)
    if exist(outVrt, 'file') == 2 && ~overwrite
        fprintf('[SKIP] Existing FixND VRT: %s\n', outVrt);
        return;
    end
    cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
                   'export PROJ_NETWORK=ON; ' ...
                   'gdalwarp -overwrite -of VRT -r near ' ...
                   '-dstnodata %.17g %s %s'], ...
                   globalND, q(srcVrt), q(outVrt));
    runCmd(cmd, 'gdalwarp FixND VRT failed');
end

function printRasterInfo(rasterFile)
    if exist(rasterFile, 'file') ~= 2
        fprintf('[MISSING] %s\n', rasterFile);
        return;
    end
    cmd = sprintf('gdalinfo %s | grep -E "Size is|Pixel Size|Upper Left|Lower Right|NoData"', q(rasterFile));
    system(cmd);
end

function n = countTextInFile(fileName, pattern)
    if exist(fileName, 'file') ~= 2
        n = 0;
        return;
    end
    cmd = sprintf('grep -c %s %s', q(pattern), q(fileName));
    [status, out] = system(cmd);
    if status ~= 0
        n = 0;
    else
        n = str2double(strtrim(out));
        if isnan(n); n = 0; end
    end
end

function summary = canonicalizeScaleTiled(srcRaster, outSub, outBaseName, cfg, unitScale, unitAction, fixF32Min, zeroIsNoData, river)
    ensureDir(outSub);

    tilesDir = fullfile(outSub, ['_', outBaseName, '_tiles']);
    listTxt  = fullfile(outSub, [outBaseName, '_tile_list.txt']);
    outVrt   = fullfile(outSub, [outBaseName, '.vrt']);
    summaryCsv = fullfile(outSub, [outBaseName, '_summary.csv']);

    if cfg.overwrite
        if exist(tilesDir, 'dir') == 7
            cmdClean = sprintf('find %s -maxdepth 1 -type f -name "tile_*.tif" -delete', q(tilesDir));
            runCmd(cmdClean, 'Failed to clean old tile outputs');
        else
            mkdir(tilesDir);
        end
        if exist(listTxt, 'file') == 2; delete(listTxt); end
        if exist(outVrt, 'file') == 2; delete(outVrt); end
        if exist(summaryCsv, 'file') == 2; delete(summaryCsv); end
    else
        ensureDir(tilesDir);
        if exist(outVrt, 'file') == 2
            fprintf('[SKIP] Existing canonical VRT: %s\n', outVrt);
            summary = table();
            return;
        end
    end

    [~, rows, cols, geoTrans, proj, ~, srcND] = RasterInfo(srcRaster);
    outDataType = 6; % Float32

    srcNDPrint = NaN;
    if ~isempty(srcND) && isfinite(srcND)
        srcNDPrint = double(srcND);
    end

    fprintf('\nCanonicalize/scale raster by tiles\n');
    fprintf('  River/action : %s / %s\n', river, unitAction);
    fprintf('  Source       : %s\n', srcRaster);
    fprintf('  Output VRT   : %s\n', outVrt);
    fprintf('  Rows/Cols    : %d / %d\n', rows, cols);
    fprintf('  Source ND    : %.17g\n', srcNDPrint);
    fprintf('  Global ND    : %.17g\n', cfg.globalND);
    fprintf('  Unit scale   : %.12g\n', unitScale);
    fprintf('  Fix f32-min  : %d\n', fixF32Min);
    fprintf('  Zero is ND   : %d\n', zeroIsNoData);

    totalTiles = ceil(rows / cfg.tileSize) * ceil(cols / cfg.tileSize);
    tileCount = 0;

    N_total = 0;
    N_nonfinite = 0;
    N_srcDeclaredND = 0;
    N_globalND = 0;
    N_f32MinND = 0;
    N_veryNegative = 0;
    N_zeroInput = 0;
    N_zeroToND = 0;
    N_valid = 0;

    rawMin = inf; rawMax = -inf; rawSum = 0;
    outMin = inf; outMax = -inf; outSum = 0;

    zeroTol = 1e-8;

    for rLocal = 1:cfg.tileSize:rows
        rr = min(cfg.tileSize, rows - rLocal + 1);

        for cLocal = 1:cfg.tileSize:cols
            cc = min(cfg.tileSize, cols - cLocal + 1);

            A = double(ReadRaster(srcRaster, rLocal, cLocal, rr, cc));

            finite = isfinite(A) & ~isnan(A);
            nonfinite = ~finite;

            srcDeclaredND = false(size(A));
            if ~isempty(srcND) && isfinite(srcND)
                srcDeclaredND = finite & (A == double(srcND));
            end

            isGlobalND = finite & (A == cfg.globalND);
            isF32MinND = false(size(A));
            if fixF32Min
                % Use both exact f32 sentinel and very small cutoff.
                isF32MinND = finite & ((A == cfg.f32MinND) | (A < -1e30));
            end
            isVeryNegative = finite & (A < -1e20);

            zeroInput = finite & abs(A) <= zeroTol;
            zeroToND = false(size(A));
            if zeroIsNoData
                zeroToND = zeroInput;
            end

            invalid = nonfinite | srcDeclaredND | isGlobalND | isF32MinND | isVeryNegative | zeroToND;
            valid = ~invalid;

            if any(valid(:))
                vals = A(valid);
                rawMin = min(rawMin, min(vals));
                rawMax = max(rawMax, max(vals));
                rawSum = rawSum + sum(vals(:));
            end

            Aout = A;
            Aout(valid) = A(valid) * unitScale;
            Aout(invalid) = cfg.globalND;

            if any(valid(:))
                valsOut = Aout(valid);
                outMin = min(outMin, min(valsOut));
                outMax = max(outMax, max(valsOut));
                outSum = outSum + sum(valsOut(:));
            end

            N_total = N_total + numel(A);
            N_nonfinite = N_nonfinite + nnz(nonfinite);
            N_srcDeclaredND = N_srcDeclaredND + nnz(srcDeclaredND);
            N_globalND = N_globalND + nnz(isGlobalND);
            N_f32MinND = N_f32MinND + nnz(isF32MinND);
            N_veryNegative = N_veryNegative + nnz(isVeryNegative);
            N_zeroInput = N_zeroInput + nnz(zeroInput);
            N_zeroToND = N_zeroToND + nnz(zeroToND);
            N_valid = N_valid + nnz(valid);

            subgeoTrans = subTranscoef(geoTrans, rLocal, cLocal);
            tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));
            WriteRaster(tileTif, Aout, subgeoTrans, proj, outDataType, 'GTiff', cfg.globalND);

            tileCount = tileCount + 1;
            fprintf('\r  Progress: %6.2f%% (%d/%d)', 100 * tileCount / totalTiles, tileCount, totalTiles);
        end
    end
    fprintf('\n');

    cmdList = sprintf('find %s -maxdepth 1 -type f -name "tile_*.tif" | sort > %s', q(tilesDir), q(listTxt));
    runCmd(cmdList, 'Failed to build tile list');

    cmdVrt = sprintf(['gdalbuildvrt -overwrite -srcnodata %.17g -vrtnodata %.17g ' ...
                      '-input_file_list %s %s'], ...
                      cfg.globalND, cfg.globalND, q(listTxt), q(outVrt));
    runCmd(cmdVrt, 'gdalbuildvrt failed for canonicalized tiles');

    rawMean = NaN; outMean = NaN;
    if N_valid > 0
        rawMean = rawSum / N_valid;
        outMean = outSum / N_valid;
    end
    if isinf(rawMin); rawMin = NaN; rawMax = NaN; end
    if isinf(outMin); outMin = NaN; outMax = NaN; end

    unitWarning = string("");
    if strcmp(river, 'CA_KlamathRiver_TopoBathy_2018_D18')
        if outMax > 2500 || outMean > 1600
            unitWarning = unitWarning + "CA valid elevations look high for meters; check possible feet-unit source. ";
        end
        if outMin < -1000
            unitWarning = unitWarning + "CA valid minimum is very low; check remaining NoData/sentinel values. ";
        end
    end

    summary = table(string(river), string(outBaseName), string(srcRaster), string(outVrt), ...
        rows, cols, string(unitAction), unitScale, srcNDPrint, cfg.globalND, ...
        N_total, N_nonfinite, N_srcDeclaredND, N_globalND, N_f32MinND, N_veryNegative, ...
        N_zeroInput, N_zeroToND, N_valid, rawMin, rawMax, rawMean, outMin, outMax, outMean, unitWarning, ...
        'VariableNames', {'River','OutputBase','Source','OutputVRT', ...
        'Rows','Cols','UnitAction','UnitScale','SourceNoData','UnifiedNoData', ...
        'N_total','N_nonfinite','N_source_declared_nd','N_global_nd','N_f32min_nd','N_very_negative', ...
        'N_zero_input','N_zero_to_nd','N_valid','Raw_min','Raw_max','Raw_mean','Output_min','Output_max','Output_mean','UnitWarning'});

    writetable(summary, summaryCsv);

    fprintf('Summary written: %s\n', summaryCsv);
    fprintf('  Valid output range: %.6f to %.6f, mean %.6f\n', outMin, outMax, outMean);
    fprintf('  f32-min ND count  : %d\n', N_f32MinND);
    fprintf('  very negative cnt : %d\n', N_veryNegative);
    if strlength(unitWarning) > 0
        warning('%s', char(unitWarning));
    end
end

function checkCAUnitWarning(summary, cfg)
    if isempty(summary) || height(summary) == 0
        return;
    end
    warningText = summary.UnitWarning(1);
    if strlength(warningText) > 0
        fprintf('\n[CA UNIT QA WARNING]\n%s\n', warningText);
        if cfg.errorOnCAUnitSuspect
            error('Stop because cfg.errorOnCAUnitSuspect=true and CA unit QA is suspicious.');
        end
    else
        fprintf('[CA UNIT QA] Range-based check did not flag obvious feet-unit issue. Still review summary CSV.\n');
    end
end

function buildKewaFixND_1m(cfg)
    river = 'KewaFix2Null';
    rawRaster = fullfile(cfg.rawRoot, 'milwaukee_river_3DEP', 'KewaFix2Null.tif');
    outSub = fullfile(cfg.bathyFixRoot, river);

    if exist(rawRaster, 'file') ~= 2
        error('Missing raw Kewa raster: %s', rawRaster);
    end

    canonicalizeScaleTiled(rawRaster, outSub, 'Bathy_1m', cfg, cfg.feetToMeter, ...
        'elevation_feet_to_meter__horizontal_grid_already_1m', ...
        true, false, river);
end

function buildORMKRCFixND_1m(cfg)
    river = 'OR_MKRC_Topobathy_2021';
    srcFolder = fullfile(cfg.rawRoot, river);
    listFile = fullfile(srcFolder, 'Filelist.txt');
    outSub = fullfile(cfg.bathyFixRoot, river);
    ensureDir(outSub);

    if exist(listFile, 'file') ~= 2
        error('Missing OR_MKRC Filelist.txt: %s', listFile);
    end

    rawFeetVRT = fullfile(outSub, 'Bathy_2ft_rawElev_ft.vrt');
    elevMeter2ftVRT = fullfile(outSub, 'Bathy_2ft_elev_m.vrt');
    out1mVRT = fullfile(outSub, 'Bathy_1m.vrt');

    fprintf('\n[OR_MKRC] Build raw 2ft/elevation-feet VRT\n');
    buildVRTFromFilelist(listFile, rawFeetVRT, cfg.overwrite);
    printRasterInfo(rawFeetVRT);

    fprintf('\n[OR_MKRC] Convert elevation feet -> meters on original 2ft horizontal grid\n');
    canonicalizeScaleTiled(rawFeetVRT, outSub, 'Bathy_2ft_elev_m', cfg, cfg.feetToMeter, ...
        'elevation_feet_to_meter__horizontal_grid_still_2ft', ...
        true, false, river);

    if exist(elevMeter2ftVRT, 'file') ~= 2
        error('Expected OR 2ft meter VRT not found: %s', elevMeter2ftVRT);
    end

    if exist(out1mVRT, 'file') == 2 && cfg.overwrite
        delete(out1mVRT);
    end

    % OR horizontal CRS unit is foot. 1 meter = 3.280839895 ft.
    meterToFoot = 1 / cfg.feetToMeter;
    res1mInFoot = meterToFoot;

    cmd = sprintf(['gdalwarp -overwrite -of VRT ' ...
                   '-tr %.9f %.9f -r near ' ...
                   '-srcnodata %.17g -dstnodata %.17g ' ...
                   '%s %s'], ...
                   res1mInFoot, res1mInFoot, cfg.globalND, cfg.globalND, ...
                   q(elevMeter2ftVRT), q(out1mVRT));
    fprintf('\n[OR_MKRC] Create true 1m VRT. gdalwarp -tr is in feet.\n');
    runCmd(cmd, 'OR_MKRC true 1m VRT creation failed');
end
