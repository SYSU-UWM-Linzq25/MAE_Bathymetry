%% ============================================================
%  B002b_Fix_LCC_OR_Kewa_Only.m
%
%  Purpose:
%    Rebuild LCC_1m/3m/5m/10m for only:
%      1) OR_MKRC_Topobathy_2021
%      2) KewaFix2Null
%
%  Logic:
%    Use corrected Bathy_*m_FixND as target grid.
%    Warp existing binary LCC clip to exactly the same rows/cols/extent/proj.
%
%  Output:
%    LCC_1m/<river>/ESA_WorldCover_Resampleandclip_1m.vrt
%    LCC_3m/<river>/ESA_WorldCover_Resampleandclip_3m.vrt
%    LCC_5m/<river>/ESA_WorldCover_Resampleandclip_5m.vrt
%    LCC_10m/<river>/ESA_WorldCover_Resampleandclip_10m.vrt
%
%  Important:
%    - LCC is categorical/binary, use nearest neighbor.
%    - Do NOT use -tap.
%    - Use -te and -ts from corrected bathy grid.
% ============================================================

clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0');
GDALLoad();

addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rootPR = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

targetRivers = { ...
    'OR_MKRC_Topobathy_2021', ...
    'KewaFix2Null'};

targetRes = [1, 3, 5, 10];

globalND = -999999;

% Existing binary LCC clip folders.
% Check both possible locations because B002 has used both paths before.
clipRootCandidates = { ...
    fullfile(rootPR, 'LCC_10m_Transfer'), ...
    fullfile(rootPR, 'Transfer_to_Dr_Chen', 'LCC_10m_Transfer')};

for iRiver = 1:numel(targetRivers)

    river = targetRivers{iRiver};

    fprintf('\n============================================================\n');
    fprintf('Rebuild LCC for river: %s\n', river);
    fprintf('============================================================\n');

    % Find existing binary LCC clip
    lcc_clip = '';

    for k = 1:numel(clipRootCandidates)
        testFile = fullfile(clipRootCandidates{k}, river, 'ESA_WorldCover_Clip_WGS84.tif');
        if exist(testFile, 'file') == 2
            lcc_clip = testFile;
            break;
        end
    end

    if isempty(lcc_clip)
        error('Missing LCC binary clip for %s. Checked candidate roots.', river);
    end

    fprintf('Using LCC clip:\n%s\n', lcc_clip);

    for j = 1:numel(targetRes)

        res = targetRes(j);

        bathyRoot = fullfile(rootPR, sprintf('Bathy_%dm_FixND', res));
        bathy_vrt = fullfile(bathyRoot, river, sprintf('Bathy_%dm.vrt', res));

        if exist(bathy_vrt, 'file') ~= 2
            % For OR, 1m is VRT. For Kewa, also VRT. This should exist.
            error('Missing corrected bathy target grid: %s', bathy_vrt);
        end

        outRoot = fullfile(rootPR, sprintf('LCC_%dm', res));
        outSub  = fullfile(outRoot, river);

        if exist(outSub, 'dir') ~= 7
            mkdir(outSub);
        end

        outVrt = fullfile(outSub, sprintf('ESA_WorldCover_Resampleandclip_%dm.vrt', res));

        [~, rows, cols, geoTrans, proj, ~, ~] = RasterInfo(bathy_vrt);

        xmin = geoTrans(1);
        xres = geoTrans(2);
        ymax = geoTrans(4);
        yres = geoTrans(6);

        xmax = xmin + cols * xres;
        ymin = ymax + rows * yres;

        proj_arg = sprintf('''%s''', proj);

        if exist(outVrt, 'file') == 2
            delete(outVrt);
        end

        cmd = sprintf([ ...
            'gdalwarp -overwrite -of VRT ', ...
            '-ot Byte ', ...
            '-r near ', ...
            '-t_srs %s -te_srs %s ', ...
            '-te %.10f %.10f %.10f %.10f ', ...
            '-ts %d %d ', ...
            '-wo INIT_DEST=0 -wo SKIP_NOSOURCE=YES ', ...
            '"%s" "%s"' ], ...
            proj_arg, proj_arg, ...
            xmin, ymin, xmax, ymax, ...
            cols, rows, ...
            lcc_clip, outVrt);

        fprintf('\n[%s] LCC %dm -> target bathy grid\n', river, res);
        fprintf('Bathy grid: %s\n', bathy_vrt);
        fprintf('Output    : %s\n', outVrt);
        fprintf('rows=%d, cols=%d\n', rows, cols);
        fprintf('%s\n', cmd);

        status = system(cmd);

        if status ~= 0
            error('gdalwarp failed for %s LCC_%dm', river, res);
        end

        fprintf('Done: %s\n', outVrt);
    end
end

fprintf('\nAll OR/Kewa LCC repair done.\n');