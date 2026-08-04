function D002c_CheckTiles(varargin)
% D002c_CheckTiles
% Summarize the completed D001c AnyVisiblePatch training-tile QA before
% launching formal model training.
%
% This is a diagnostic only. It does not change or reject any D001c tile.
%
% Inputs are read from the D001c preprocessing dataset under Processed_Results.
% Outputs are written under the isolated downstream relax project results.
%
% Example:
%   D002c_CheckTiles;
%
% Optional:
%   D002c_CheckTiles('writeCombinedKept', true);

p = inputParser;
p.addParameter('tileDatasetRoot', ...
    '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_v2_D001c_AnyVisiblePatch', @ischar);
p.addParameter('outputRoot', ...
    '/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask/results/D002c_D001c_Tile_QA', @ischar);
p.addParameter('resolution', 1, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('thresholds', [1 16 64 128 256 512 1024], @isnumeric);
p.addParameter('writeCombinedKept', false, @(x)islogical(x)||ismember(x,[0 1]));
p.addParameter('makeFigures', true, @(x)islogical(x)||ismember(x,[0 1]));
p.parse(varargin{:});
cfg = p.Results;

resStr = resolutionString(cfg.resolution);
qaRoot = fullfile(cfg.tileDatasetRoot, 'QA');
listRoot = fullfile(cfg.tileDatasetRoot, 'Lists');
outRoot = cfg.outputRoot;
if exist(qaRoot, 'dir') ~= 7
    error('Missing D001c QA root: %s', qaRoot);
end
if exist(outRoot, 'dir') ~= 7
    mkdir(outRoot);
end

files = dir(fullfile(qaRoot, '*', sprintf('D001c_candidate_patch_QA_%s_*.csv', resStr)));
if isempty(files)
    error('No D001c candidate QA CSV files found under: %s', qaRoot);
end
[~, ord] = sort({files.folder});
files = files(ord);

required = { ...
    'kept', 'reject', 'hidden_patch_count', 'visible_patch_count', ...
    'known_patch_ratio', 'core_hidden_patch_count', ...
    'core_loss_pixel_count', 'core_loss_pixel_ratio', ...
    'effective_core_loss_pixel_count'};
qprob = [0 0.01 0.05 0.10 0.25 0.50 0.75 0.90 0.95 0.99 1.00];
qname = {'min','p01','p05','p10','p25','median','p75','p90','p95','p99','max'};

riverSummary = table();
quantileSummary = table();
thresholdSummary = table();
rejectSummary = table();
manifestSummary = table();
allKept = table();
riverData = struct('name', {}, 'effective', {}, 'coreHidden', {});

fprintf('\n============================================================\n');
fprintf('D002c D001c AnyVisiblePatch tile QA\n');
fprintf('Input : %s\n', cfg.tileDatasetRoot);
fprintf('Output: %s\n', outRoot);
fprintf('Rivers: %d\n', numel(files));
fprintf('============================================================\n');

for i = 1:numel(files)
    csvPath = fullfile(files(i).folder, files(i).name);
    [~, river] = fileparts(files(i).folder);
    T = readtable(csvPath, 'VariableNamingRule', 'preserve');
    assertColumns(T, required, csvPath);

    keptMask = double(T.kept) == 1;
    K = T(keptMask, :);
    if isempty(K)
        warning('No kept tiles in %s', river);
        continue;
    end

    e = finiteVector(K.effective_core_loss_pixel_count);
    hp = finiteVector(K.hidden_patch_count);
    vp = finiteVector(K.visible_patch_count);
    kr = finiteVector(K.known_patch_ratio);
    ch = finiteVector(K.core_hidden_patch_count);
    clp = finiteVector(K.core_loss_pixel_count);
    clr = finiteVector(K.core_loss_pixel_ratio);

    row = table(string(river), height(T), height(K), height(K)/max(1,height(T)), ...
        min(e), quantile(e,0.10), median(e), quantile(e,0.90), max(e), ...
        sum(e==0), sum(e<64), sum(e<128), sum(e<256), ...
        median(ch), median(hp), median(vp), median(kr), median(clp), median(clr), ...
        'VariableNames', {'river','candidate_count','kept_count','keep_fraction', ...
        'effective_min','effective_p10','effective_median','effective_p90','effective_max', ...
        'effective_eq0','effective_lt64','effective_lt128','effective_lt256', ...
        'core_hidden_patch_median','hidden_patch_median','visible_patch_median', ...
        'known_patch_ratio_median','core_loss_pixel_median','core_loss_pixel_ratio_median'});
    riverSummary = [riverSummary; row]; %#ok<AGROW>

    metrics = { ...
        'effective_core_loss_pixel_count', e; ...
        'core_hidden_patch_count', ch; ...
        'hidden_patch_count', hp; ...
        'visible_patch_count', vp; ...
        'known_patch_ratio', kr; ...
        'core_loss_pixel_count', clp; ...
        'core_loss_pixel_ratio', clr};
    for m = 1:size(metrics,1)
        vals = metrics{m,2};
        qv = quantile(vals, qprob);
        for j = 1:numel(qprob)
            qr = table(string(river), string(metrics{m,1}), string(qname{j}), qprob(j), qv(j), ...
                'VariableNames', {'river','metric','quantile_name','quantile_probability','value'});
            quantileSummary = [quantileSummary; qr]; %#ok<AGROW>
        end
    end

    for th = reshape(cfg.thresholds,1,[])
        tr = table(string(river), th, sum(e < th), sum(e <= th), ...
            sum(e < th)/numel(e), sum(e <= th)/numel(e), ...
            'VariableNames', {'river','threshold','n_below','n_at_or_below','fraction_below','fraction_at_or_below'});
        thresholdSummary = [thresholdSummary; tr]; %#ok<AGROW>
    end

    rej = string(T.reject);
    [u,~,g] = unique(rej);
    counts = accumarray(g,1);
    for j = 1:numel(u)
        rr = table(string(river), u(j), counts(j), counts(j)/height(T), ...
            'VariableNames', {'river','reject_reason','count','fraction_of_candidates'});
        rejectSummary = [rejectSummary; rr]; %#ok<AGROW>
    end

    manifestPath = fullfile(files(i).folder, sprintf('D001c_tile_manifest_%s_%s.csv', resStr, river));
    manifestRows = NaN; missingDEM = NaN; missingHidden = NaN; missingLoss = NaN; duplicateDEM = NaN;
    if exist(manifestPath, 'file') == 2
        M = readtable(manifestPath, 'VariableNamingRule', 'preserve');
        manifestRows = height(M);
        missingDEM = countMissingPaths(M, 'dem_path');
        missingHidden = countMissingPaths(M, 'hidden_mask_path');
        missingLoss = countMissingPaths(M, 'loss_pixel_mask_path');
        duplicateDEM = countDuplicates(M, 'dem_path');
    end
    mr = table(string(river), height(K), manifestRows, missingDEM, missingHidden, missingLoss, duplicateDEM, ...
        'VariableNames', {'river','kept_count','manifest_rows','missing_dem','missing_hidden','missing_loss','duplicate_dem_paths'});
    manifestSummary = [manifestSummary; mr]; %#ok<AGROW>

    if cfg.writeCombinedKept
        K.river = repmat(string(river), height(K), 1);
        allKept = [allKept; K]; %#ok<AGROW>
    end
    riverData(end+1).name = river; %#ok<AGROW>
    riverData(end).effective = e;
    riverData(end).coreHidden = ch;

    fprintf('%-44s kept=%6d  effective p10/med/p90=%8.1f / %8.1f / %8.1f\n', ...
        river, height(K), quantile(e,0.10), median(e), quantile(e,0.90));
end

% Aggregate ALL kept tiles without silently weighting rivers equally.
allEffective = [];
allCoreHidden = [];
for i = 1:numel(riverData)
    allEffective = [allEffective; riverData(i).effective(:)]; %#ok<AGROW>
    allCoreHidden = [allCoreHidden; riverData(i).coreHidden(:)]; %#ok<AGROW>
end
if ~isempty(allEffective)
    allRow = table("ALL", sum(riverSummary.candidate_count), sum(riverSummary.kept_count), ...
        sum(riverSummary.kept_count)/max(1,sum(riverSummary.candidate_count)), ...
        min(allEffective), quantile(allEffective,0.10), median(allEffective), quantile(allEffective,0.90), max(allEffective), ...
        sum(allEffective==0), sum(allEffective<64), sum(allEffective<128), sum(allEffective<256), ...
        median(allCoreHidden), NaN, NaN, NaN, NaN, NaN, ...
        'VariableNames', riverSummary.Properties.VariableNames);
    riverSummary = [riverSummary; allRow];
end

writetable(riverSummary, fullfile(outRoot, 'D002c_river_summary.csv'));
writetable(quantileSummary, fullfile(outRoot, 'D002c_quantile_summary.csv'));
writetable(thresholdSummary, fullfile(outRoot, 'D002c_effective_threshold_summary.csv'));
writetable(rejectSummary, fullfile(outRoot, 'D002c_reject_summary.csv'));
writetable(manifestSummary, fullfile(outRoot, 'D002c_manifest_integrity.csv'));
if cfg.writeCombinedKept && ~isempty(allKept)
    writetable(allKept, fullfile(outRoot, 'D002c_kept_tile_stats_all.csv'));
end

writeTextReport(outRoot, cfg, riverSummary, manifestSummary);
if cfg.makeFigures && ~isempty(riverData)
    makeFigures(outRoot, riverData);
end

fprintf('============================================================\n');
fprintf('D002c completed. Results: %s\n', outRoot);
fprintf('Primary table: D002c_river_summary.csv\n');
fprintf('============================================================\n');
end

function x = finiteVector(x)
x = double(x(:));
x = x(isfinite(x));
if isempty(x)
    x = NaN;
end
end

function assertColumns(T, required, path)
missing = setdiff(required, T.Properties.VariableNames);
if ~isempty(missing)
    error('Missing columns in %s: %s', path, strjoin(missing, ', '));
end
end

function n = countMissingPaths(T, varName)
if ~ismember(varName, T.Properties.VariableNames)
    n = height(T);
    return;
end
v = string(T.(varName));
n = 0;
for i = 1:numel(v)
    if strlength(v(i)) == 0 || exist(char(v(i)), 'file') ~= 2
        n = n + 1;
    end
end
end

function n = countDuplicates(T, varName)
if ~ismember(varName, T.Properties.VariableNames)
    n = NaN;
    return;
end
v = string(T.(varName));
n = numel(v) - numel(unique(v));
end

function makeFigures(outRoot, riverData)
fig = figure('Visible','off','Color','w','Position',[100 100 1200 760]);
hold on;
for i = 1:numel(riverData)
    x = sort(max(riverData(i).effective(:), 0.5));
    y = (1:numel(x))' / numel(x);
    semilogx(x, y, 'LineWidth', 1.4);
end
grid on; xlabel('Effective hidden core-loss pixels per kept tile'); ylabel('Empirical CDF');
title('D001c effective hidden supervision distribution');
legend({riverData.name}, 'Interpreter','none', 'Location','eastoutside');
saveFigure(fig, fullfile(outRoot, 'D002c_effective_core_loss_CDF.png'));
close(fig);

fig = figure('Visible','off','Color','w','Position',[100 100 1200 760]);
hold on;
for i = 1:numel(riverData)
    x = sort(riverData(i).coreHidden(:));
    y = (1:numel(x))' / numel(x);
    plot(x, y, 'LineWidth', 1.4);
end
grid on; xlabel('Core hidden patch count per kept tile'); ylabel('Empirical CDF');
title('D001c remaining hidden core patches');
legend({riverData.name}, 'Interpreter','none', 'Location','eastoutside');
saveFigure(fig, fullfile(outRoot, 'D002c_core_hidden_patch_CDF.png'));
close(fig);
end

function saveFigure(fig, path)
try
    exportgraphics(fig, path, 'Resolution', 180);
catch
    saveas(fig, path);
end
end

function writeTextReport(outRoot, cfg, S, M)
path = fullfile(outRoot, 'D002c_report.txt');
fid = fopen(path, 'w');
if fid < 0
    warning('Cannot write report: %s', path);
    return;
end
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'D002c D001c AnyVisiblePatch training-tile QA\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 31));
fprintf(fid, 'Tile dataset: %s\n', cfg.tileDatasetRoot);
fprintf(fid, 'This script is diagnostic only and does not alter the D001c dataset.\n\n');
for i = 1:height(S)
    fprintf(fid, '%s: candidates=%d kept=%d keep_fraction=%.6f effective p10/median/p90=%.3f/%.3f/%.3f\n', ...
        char(S.river(i)), S.candidate_count(i), S.kept_count(i), S.keep_fraction(i), ...
        S.effective_p10(i), S.effective_median(i), S.effective_p90(i));
end
fprintf(fid, '\nManifest integrity:\n');
for i = 1:height(M)
    fprintf(fid, '%s: kept=%d manifest=%g missing DEM/Hidden/Loss=%g/%g/%g duplicate DEM=%g\n', ...
        char(M.river(i)), M.kept_count(i), M.manifest_rows(i), M.missing_dem(i), ...
        M.missing_hidden(i), M.missing_loss(i), M.duplicate_dem_paths(i));
end
end

function s = resolutionString(res)
if abs(res-round(res)) < 1e-9
    s = sprintf('%dm', round(res));
else
    s = sprintf('%gm', res);
    s = strrep(s, '.', 'p');
end
end
