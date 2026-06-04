function RiverPanel_Pixel_Search_skel_CO_hugh(bathy_vrt_1m,bathy_vrt_3m,bathy_vrt_5m,bathy_vrt_10m, ...
    final_mix_vrt_1m,final_mix_vrt_3m,final_mix_vrt_5m,final_mix_vrt_10m, ...
    LCC_vrt_1m,LCC_vrt_3m,LCC_vrt_5m,LCC_vrt_10m, ...
    srcLine,ExtractPercent,winH,winW,TempFolder,OutFolder,BasinName)

%% Nov 24th, 2025
% 输入四种分辨率下的bathy,bathy_merge, LCC, srcline(河道中心的shape文件路径),ExtractPercent(随机取样的比例)
% winH和winW对应要取得tile的规格，目前使用的是336
% srcline直接来源于陈博士的算法，直接得到的shp包含了河道中心点的lineID（随机取样的样本), width（河宽）
% OutFolder 输出的路径
% BasinName 流域的名称
% 需要确保输入的坐标系一致

% %% 测试
% clear;clc;
%
% final_mix_vrt_1m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_1m/Kletzch_Combined_UpMax3Null/Combined_Bathy_Priority_1m.vrt';
% final_mix_vrt_3m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_3m/Kletzch_Combined_UpMax3Null/Combined_Bathy_Priority_3m.tif';
% final_mix_vrt_5m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_5m/Kletzch_Combined_UpMax3Null/Combined_Bathy_Priority_5m.tif';
% final_mix_vrt_10m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_10m/Kletzch_Combined_UpMax3Null/Combined_Bathy_Priority_10m.tif';
%
% bathy_vrt_1m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/Kletzch_Combined_UpMax3Null/Bathy_1m.vrt';
% bathy_vrt_3m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_3m_FixND/Kletzch_Combined_UpMax3Null/Bathy_3m.vrt';
% bathy_vrt_5m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_5m_FixND/Kletzch_Combined_UpMax3Null/Bathy_5m.vrt';
% bathy_vrt_10m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_10m_FixND/Kletzch_Combined_UpMax3Null/Bathy_10m.vrt';
%
% LCC_vrt_1m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_1m/Kletzch_Combined_UpMax3Null/ESA_WorldCover_Resampleandclip_1m.vrt';
% LCC_vrt_3m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_3m/Kletzch_Combined_UpMax3Null/ESA_WorldCover_Resampleandclip_3m.vrt';
% LCC_vrt_5m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_5m/Kletzch_Combined_UpMax3Null/ESA_WorldCover_Resampleandclip_5m.vrt';
% LCC_vrt_10m = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_10m/Kletzch_Combined_UpMax3Null/ESA_WorldCover_Resampleandclip_10m.vrt';
%
% srcLine = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/CenterRiverLine_skel/Reproj/Kletzch_Combined_UpMax3Null/ESA_WorldCover_Width_proj.shp';
% ExtractPercent = 0.7;
% TempFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/CenterRiverLine_skel/Sample_Extract/Kletzch_Combined_UpMax3Null/';
% BasinName = 'Kletzch_Combined_UpMax3Null';
%
% winH = 336;   % 行数（高度）
% winW = 336;   % 列数（宽度）
%
% OutFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE_Test/';

if exist(OutFolder,'dir') ~= 7
    mkdir(OutFolder);
end
if exist(TempFolder,'dir') ~= 7
    mkdir(TempFolder);
end

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');


%% 第一部分使用同一个抽样概率 p 做伯努利抽样
disp('Step1: Srcline Points Sample Extraction')

GT = shaperead(srcLine);

% 抽样比例（比如 10%）
sampleRate = ExtractPercent;
rng(2025);               % 固定随机种子，结果可复现

nPts    = numel(GT);
allLID  = vertcat(GT.line_ID);   % 假设是 numeric，如果是 cellstr 用 string/str2double 处理
uLID    = unique(allLID);

keep = false(nPts,1);

for k = 1:numel(uLID)
    idx = find(allLID == uLID(k));     % 这一条流线上的所有点

    % 这一条线内部做等概率抽样
    r = rand(numel(idx),1);
    lineKeep = r < sampleRate;

    % 保护：保证每条线至少有 1 个点被抽到
    if ~any(lineKeep)
        lineKeep(randi(numel(idx))) = true;
    end

    keep(idx) = lineKeep;
end

GT_sample = GT(keep);
% 导出为新的点 shp 便于后续使用
outShp = fullfile([TempFolder,'/centerline_pts_randompct.shp']);
shapewrite(GT_sample, outShp);
disp(['随机采样结果写出: ', outShp]);

% 2) 复制 .prj（保证坐标系一致）
[~,base,~] = fileparts(outShp);
[pthL,baseL,~] = fileparts(srcLine);
srcPrj = fullfile(pthL, [baseL '.prj']);
dstPrj = fullfile(fileparts(outShp), [base '.prj']);
if exist(srcPrj,'file')
    copyfile(srcPrj, dstPrj, 'f');
else
    warning('未找到源 .prj：%s（可用 ogr2ogr -a_srs 补写）', srcPrj);
end

% 注意：point shapefile 的 X/Y 对于每个要素一般是标量（可能带一个 NaN）
Xs    = vertcat(GT_sample.X);
Ys    = vertcat(GT_sample.Y);
LID   = vertcat(GT_sample.line_ID);
Width = vertcat(GT_sample.Width);

% 如果发现末尾多一堆 NaN，可以加一步清理：
nanMask = isnan(Xs) | isnan(Ys);
Xs(nanMask)    = [];
Ys(nanMask)    = [];
LID(nanMask)   = [];
Width(nanMask) = [];

P = table(Xs, Ys, LID, Width, ...
    'VariableNames', {'X','Y','LineID','Width'});

%% 可选部分，提取Fac

% % 根据这个的shp的point进行进一步的筛选
% % 读取FAC进行一个筛选判断
%
% fac = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/MD_PotomacRiver_Bathy_2019/NHDPlusV21_Data_Process/Range_fac_Proj.tif';
% [~,~,~,geoTrans,~,~,nodataval]=RasterInfo(fac);
%
% % 预分配
% P.fac = nan(height(P), 1);
%
% for i = 1 : height(P)
%     Points_Lon = P.X(i);
%     Points_Lat = P.Y(i);
%     [row0,col0]=Proj2RowCol(geoTrans,Points_Lat,Points_Lon);
%
%     % 读取 1x1 像元窗口（按你当前函数的参数顺序）
%     fac_Point = ReadRaster(fac,row0,col0,1,1); % 读取这个点的fac
%     % NoData 处理
%     if ~isempty(nodataval) && isfinite(nodataval) && isequal(fac_Point, double(nodataval))
%         P.fac(i) = nan;
%     else
%         P.fac(i) = fac_Point;
%     end
%     disp([num2str(i),' fac extraction is done!'])
% end

%% 第三部分 对采样点的提取分析
% 进一步跟bathmatry进行一个提取范围
% 计算bathmatry栅格数量和占比情况
% 同时可以结合LCC的河流mask进行有效的bathmatry+3DEP的提取

disp('Step2: srcline Points bathmatry extraction')

% 目标分辨率（单位：米）
% targetRes = [1, 3, 5, 10];   % 如果 1m 也要算就加上
targetRes = 1;   % 如果 1m 也要算就加上

% 取不同分辨率
mix_vrt.final_1  = final_mix_vrt_1m;
mix_vrt.final_3  = final_mix_vrt_3m;
mix_vrt.final_5  = final_mix_vrt_5m;
mix_vrt.final_10 = final_mix_vrt_10m;

bathy_map.final_1  = bathy_vrt_1m;
bathy_map.final_3  = bathy_vrt_3m;
bathy_map.final_5  = bathy_vrt_5m;
bathy_map.final_10 = bathy_vrt_10m;

LCC_map.final_1  = LCC_vrt_1m;
LCC_map.final_3  = LCC_vrt_3m;
LCC_map.final_5  = LCC_vrt_5m;
LCC_map.final_10 = LCC_vrt_10m;

hr = floor(winH/2);
hc = floor(winW/2);

for j = 1:length(targetRes)
    res = targetRes(j);    % 例如 1 / 3 / 5 / 10

    % ==== 1. 选择对应的 VRT 文件 ====
    key = sprintf('final_%d', res);   % 'final_1' / 'final_3' / ...
    final_mix_vrt = mix_vrt.(key);
    bathy_vrt     = bathy_map.(key);
    LCC_vrt       = LCC_map.(key);

    clear rows cols geoTrans proj dataType nodataval
    [~, rowsF, colsF, geoTrans, ~, ~, ~] = RasterInfo(final_mix_vrt);
    [~, rowsB, colsB, ~,       ~, ~, ~]  = RasterInfo(bathy_vrt);
    [~, rowsL, colsL, ~,       ~, ~, ~]  = RasterInfo(LCC_vrt);

    % rows = min([rowsF, rowsB, rowsL]);
    % cols = min([colsF, colsB, colsL]);

    if rowsF ~= rowsB || colsF ~= colsB || rowsF ~= rowsL || colsF ~= colsL
        error('Grid size mismatch for %s %dm: final=%d/%d, bathy=%d/%d, mask=%d/%d', ...
            BasinName, res, rowsF, colsF, rowsB, colsB, rowsL, colsL);
    end

    rows = rowsF;
    cols = colsF;

    % ==== 2. 为这一种分辨率预分配不同的列 ====
    suffix = sprintf('_%dm', res);    % '_1m', '_3m', '_5m', '_10m'

    var1 = ['bathamtry_Grid_num'        suffix];
    var2 = ['bathamtry_Grid_Ratio'      suffix];
    var3 = ['bathamtry_ValidGrid_num'   suffix];
    var4 = ['bathamtry_ValidGrid_Ratio' suffix];

    % 预分配列（每个分辨率对应自己的一组列）
    P.(var1) = nan(height(P), 1);
    P.(var2) = nan(height(P), 1);
    P.(var3) = nan(height(P), 1);
    P.(var4) = nan(height(P), 1);

    % ==== 3. 对每个采样点做提取 ====
    for i = 1 : height(P)
        Points_Lon = P.X(i);
        Points_Lat = P.Y(i);
        [row0,col0]=Proj2RowCol(geoTrans,Points_Lat,Points_Lon);

        % 以这个点为中心进行相应的提取
        % 3) 初始窗口左上角
        r1 = row0 - hr;
        c1 = col0 - hc;

        % 再计算右下角并反推实际尺寸（靠边时窗口会变小）
        r2 = r1 + winH - 1;
        c2 = c1 + winW - 1;
        h  = r2 - r1 + 1;
        w  = c2 - c1 + 1;

        % 判断是否超出bathmatry的范围
        if r1 < 1 || c1 < 1 || (r1 + h - 1) > rows || (c1 + w - 1) > cols
            P.(var1)(i) = NaN;
            P.(var2)(i) = NaN;
            P.(var3)(i) = NaN;
            P.(var4)(i) = NaN;
            fprintf('Res = %dm: %d Tile Out of Range!\n', res, i);
            continue
        end

        % 读取 这个tile的bathmatry
        tile = double(ReadRaster(bathy_vrt, r1, c1, h, w));
        tile_LCC_raw = double(ReadRaster(LCC_vrt, r1, c1, h, w));
        tileBathmatry3DEP = ReadRaster(final_mix_vrt, r1, c1, h, w);

        % bathy 有效区
        bathy_valid = isfinite(tile) & ~isnan(tile) & ...
            (tile ~= -999999) & (tile > -1e20);

        % simple final mask:
        % 1 = 需要预测
        % 0 = 不预测
        tile_finalMask = (tile_LCC_raw == 1) & bathy_valid;

        k_valid = find(bathy_valid);
        P.(var1)(i) = numel(k_valid);
        P.(var2)(i) = numel(k_valid) / (h * w);

        tileBathmatry3DEP_outRiver = tileBathmatry3DEP;
        tileBathmatry3DEP_outRiver(tile_finalMask == 1) = nan;

        k_valid_OutRiver = find(~isnan(tileBathmatry3DEP_outRiver));
        P.(var3)(i) = length(k_valid_OutRiver);
        P.(var4)(i) = length(k_valid_OutRiver)/(h*w);

        fprintf('Res = %dm: %d bathymetry extraction is done!\n', res, i);

        clear Points_Lon Points_Lat row0 col0 r1 c1 r2 c2 h w ...
            tile tile_LCC tile_3DEP tile_3DEP_inRiver ...
            k_valid k_valid_inRiver
    end
end

%% 生成目前统计分析的结果的shp
% 假设 P 是你得到的 table
% 1) 组装点要素 struct 并写出
% 生成目前统计分析的结果的shp

disp('Step3: srcline Points Output with table')
outPts = fullfile([TempFolder,'/centerline_pts_randompct_analysis.shp']);

% 1) 先构造一个“模板”struct，包含所有字段
S0 = struct( ...
    'Geometry', 'Point', ...
    'X', [], ...
    'Y', [], ...
    'LineID', [], ...
    'Width', [] );

% 为每个分辨率添加统计字段（短一点，避免 DBF 字段太长）
for r = targetRes

    fld_bnum  = sprintf('bnum_%dm',  r);
    fld_brat  = sprintf('brat_%dm',  r);
    fld_vbnum = sprintf('vbnum_%dm', r);
    fld_vbrat = sprintf('vbrat_%dm', r);

    S0.(fld_bnum)  = [];
    S0.(fld_brat)  = [];
    S0.(fld_vbnum) = [];
    S0.(fld_vbrat) = [];
end

% 2) 按模板扩展到每一个点
S = repmat(S0, height(P), 1);

% 3) 填值
for i = 1:height(P)
    S(i).X      = P.X(i);
    S(i).Y      = P.Y(i);
    S(i).LineID = P.LineID(i);
    S(i).Width  = P.Width(i);

    % 各分辨率的统计结果
    for r = targetRes
        suffix = sprintf('_%dm', r);

        % 对应 P 中的列名
        var_bnum   = ['bathamtry_Grid_num'        suffix];
        var_brat   = ['bathamtry_Grid_Ratio'      suffix];
        var_vbnum  = ['bathamtry_ValidGrid_num'   suffix];
        var_vbrat  = ['bathamtry_ValidGrid_Ratio' suffix];

        % 如果这些列存在，就写入 S(i) 的字段中
        if ismember(var_bnum, P.Properties.VariableNames)
            fld_bnum  = sprintf('bnum_%dm',  r);
            fld_brat  = sprintf('brat_%dm',  r);
            fld_vbnum = sprintf('vbnum_%dm', r);
            fld_vbrat = sprintf('vbrat_%dm', r);

            S(i).(fld_bnum)  = P.(var_bnum)(i);
            S(i).(fld_brat)  = P.(var_brat)(i);
            S(i).(fld_vbnum) = P.(var_vbnum)(i);
            S(i).(fld_vbrat) = P.(var_vbrat)(i);
        end
    end
end

% 4) 写出 shapefile
shapewrite(S, outPts);
disp(['Point shapefile saved: ', outPts]);

% 2) 复制 .prj（保证坐标系一致）
[~,base,~] = fileparts(outPts);
[pthL,baseL,~] = fileparts(srcLine);
srcPrj = fullfile(pthL, [baseL '.prj']);
dstPrj = fullfile(fileparts(outPts), [base '.prj']);
if exist(srcPrj,'file')
    copyfile(srcPrj, dstPrj, 'f');
else
    warning('未找到源 .prj：%s（可用 ogr2ogr -a_srs 补写）', srcPrj);
end

%% 进行条件筛选，输出满足要求的shp点图
% 为合适的point赋予一个PointID

nP = height(P);

% 1. 主河道筛选：1m 的 Grid_Ratio >= 20%
maskMain = isfinite(P.bathamtry_Grid_Ratio_1m) & ...
    ~isnan(P.bathamtry_Grid_Ratio_1m) & ...
    P.bathamtry_Grid_Ratio_1m >= 0.20;   % 大于等于 20%

% 也可以先得到主河道子集
% P_main = P(maskMain, :);

% 2. 为每个分辨率增加一个 0/1 列，表示 ValidGrid_Ratio > 80%

for r = targetRes
    suffix = sprintf('_%dm', r);   % '_1m', '_3m', ...

    % 对应 P 里的 ValidGrid_Ratio 列名
    varValidRatio = ['bathamtry_ValidGrid_Ratio' suffix];

    % 新建一个 0/1 列，默认全 0
    flagName = sprintf('ValidFlag_%dm', r);
    P.(flagName) = zeros(nP, 1);   % double 类型，用 0/1 表示

    % 先确定这个列确实存在（以防有的分辨率没算）
    if ~ismember(varValidRatio, P.Properties.VariableNames)
        warning('Column %s not found in P, skip %dm', varValidRatio, r);
        continue;
    end

    % 条件：主河道 + 有效比率 > 80%（如果你仍然想排除 1，可以加 < 1 条件）
    maskValid_r = maskMain & ...
        isfinite(P.(varValidRatio)) & ...
        ~isnan(P.(varValidRatio)) & ...
        P.(varValidRatio) > 0.80 & ...
        P.(varValidRatio) < 1.0;   % 如果不想排除全 1，可以去掉这一条

    % 满足条件的位置标记为 1
    P.(flagName)(maskValid_r) = 1;
end

P.BestRes = nan(nP, 1);

% 按从高到低分辨率依次赋值：
% 1m > 3m > 5m > 10m
for r = [1 3 5 10]   % 顺序很关键：先 1m，再 3m，再 5m，再 10m
    flagName = sprintf('ValidFlag_%dm', r);
    if ~ismember(flagName, P.Properties.VariableNames)
        continue;
    end

    mask = (P.(flagName) == 1) & isnan(P.BestRes);
    P.BestRes(mask) = r;
end

% 最终有效点：至少在一个分辨率下通过筛选
maskAnyValid = ~isnan(P.BestRes);
P_select = P(maskAnyValid, :);

% === 确保每个点有一个 PointID（方便命名）===
if ~ismember('PointID', P_select.Properties.VariableNames)
    P_select.PointID = (1:height(P_select)).';
end

fprintf('共有 %d/%d 个点在至少一个分辨率下有效。\n', ...
    sum(maskAnyValid), nP);

%% 生成符合条件的点
%% Step4: 按 BestRes 分辨率分别输出 shp

% 假设：
%   P_select 已包含：
%      X, Y, LineID, Width, BestRes
%      bathamtry_Grid_num_1m/_3m/_5m/_10m
%      bathamtry_ValidGrid_Ratio_1m/_3m/_5m/_10m 等
%   OutFolder 已存在
%   srcLine 为一个具有正确投影的线/点 shp，用来拷贝 .prj

outNameBase = [BasinName,'_Select_CenterPoints'];   % 基础文件名
disp('Step4: Qualified srcline Points Output by BestRes')

for r = targetRes
    mask_r = (P_select.BestRes == r);
    if ~any(mask_r)
        fprintf('  分辨率 %dm: 无有效点，跳过。\n', r);
        continue;
    end

    P_r = P_select(mask_r, :);
    nR  = height(P_r);

    % === 输出 shapefile 路径 ===
    outPts_r = fullfile(OutFolder, sprintf('%s_%dm.shp', outNameBase, r));

    % === 构造 struct 模板 ===
    S0 = struct( ...
        'Geometry', 'Point', ...
        'PointID',  [], ...
        'X',        [], ...
        'Y',        [], ...
        'LineID',   [], ...
        'Width',    [], ...
        'BestRes',  [] );

    % 为该分辨率添加简短字段名（避免 DBF 字段太长）
    % bnum  = Grid_num
    % brat  = Grid_Ratio
    % vbnum = ValidGrid_num
    % vbrat = ValidGrid_Ratio
    S0.bnum  = [];
    S0.brat  = [];
    S0.vbnum = [];
    S0.vbrat = [];

    % === 扩展到每个点 ===
    S_select = repmat(S0, nR, 1);

    % 该分辨率对应的列名（在 P_r 中）
    suffix = sprintf('_%dm', r);   % '_1m', '_3m', ...
    var_bnum  = ['bathamtry_Grid_num'        suffix];
    var_brat  = ['bathamtry_Grid_Ratio'      suffix];
    var_vbnum = ['bathamtry_ValidGrid_num'   suffix];
    var_vbrat = ['bathamtry_ValidGrid_Ratio' suffix];

    for i = 1:nR
        S_select(i).PointID = P_r.PointID(i);                 % 用原行号
        S_select(i).X       = P_r.X(i);
        S_select(i).Y       = P_r.Y(i);
        S_select(i).LineID  = P_r.LineID(i);
        S_select(i).Width   = P_r.Width(i);
        S_select(i).BestRes = P_r.BestRes(i);    % 应该等于 r

        % 对应分辨率的统计值
        S_select(i).bnum  = P_r.(var_bnum)(i);
        S_select(i).brat  = P_r.(var_brat)(i);
        S_select(i).vbnum = P_r.(var_vbnum)(i);
        S_select(i).vbrat = P_r.(var_vbrat)(i);
    end

    % === 写出 shp ===
    shapewrite(S_select, outPts_r);
    fprintf('  分辨率 %dm: 输出 %d 个点 -> %s\n', r, nR, outPts_r);

    % === 复制 .prj（保证坐标系一致）===
    [~,baseOut,~] = fileparts(outPts_r);
    [pthL,baseL,~] = fileparts(srcLine);
    srcPrj = fullfile(pthL, [baseL '.prj']);
    dstPrj = fullfile(fileparts(outPts_r), [baseOut '.prj']);
    if exist(srcPrj,'file')
        copyfile(srcPrj, dstPrj, 'f');
    else
        warning('未找到源 .prj：%s（可用 ogr2ogr -a_srs 补写）', srcPrj);
    end
    clear S_select mask_r P_r nR
end

%% 最后一部分，生成tile
%% Step5: 根据 BestRes 生成最终的 tiles（按分辨率分文件夹）

% 这里假设你已有这些变量（前面已经定义过）：
% final_mix_vrt_1m, final_mix_vrt_3m, final_mix_vrt_5m, final_mix_vrt_10m
% LCC_vrt_1m,      LCC_vrt_3m,      LCC_vrt_5m,      LCC_vrt_10m
% （如果还有 bathy_vrt_* 以后也可以加）

disp('Step5: Qualified srcline Points Tile Bathymetry & 3DEP Tiff output by BestRes')

for r = targetRes
    % ---- 1. 找到 BestRes == r 的点 ----
    mask_r = (P_select.BestRes == r);
    if ~any(mask_r)
        fprintf('Res = %dm: 没有点需要生成 tile，跳过。\n', r);
        continue;
    end

    P_r = P_select(mask_r, :);
    nR  = height(P_r);
    fprintf('Res = %dm: 需要生成 %d 个 tile。\n', r, nR);

    % ---- 2. 为该分辨率选择对应的栅格 ----
    switch r
        case 1
            final_mix_vrt = final_mix_vrt_1m;
            bathy_vrt     = bathy_vrt_1m;
            LCC_vrt       = LCC_vrt_1m;
        case 3
            final_mix_vrt = final_mix_vrt_3m;
            bathy_vrt     = bathy_vrt_3m;
            LCC_vrt       = LCC_vrt_3m;
        case 5
            final_mix_vrt = final_mix_vrt_5m;
            bathy_vrt     = bathy_vrt_5m;
            LCC_vrt       = LCC_vrt_5m;
        case 10
            final_mix_vrt = final_mix_vrt_10m;
            bathy_vrt     = bathy_vrt_10m;
            LCC_vrt       = LCC_vrt_10m;
        otherwise
            warning('未知分辨率 %d m，跳过。', r);
            continue;
    end

    % ---- 3. 为该分辨率创建输出文件夹 ----
    tileFolder = fullfile(OutFolder, sprintf('Tiles_%dm', r));
    if exist(tileFolder,'dir') ~= 7
        mkdir(tileFolder);
    end

    LCC_Mask_Folder = fullfile(tileFolder,sprintf('LCC_Mask'));
    if exist(LCC_Mask_Folder,'dir') ~= 7
        mkdir(LCC_Mask_Folder);
    end
    Train_tile_Folder = fullfile(tileFolder,sprintf('Train_tile'));
    if exist(Train_tile_Folder,'dir') ~= 7
        mkdir(Train_tile_Folder);
    end
    TileOutRiver_Folder = fullfile(tileFolder,sprintf('TileOutRiver'));
    if exist(TileOutRiver_Folder,'dir') ~= 7
        mkdir(TileOutRiver_Folder);
    end

    % ---- 4. 读取该分辨率的基本信息 ----
    clear rowsF colsF rowsL colsL rows cols geoTrans proj dataType nodataval

    [~, rowsF, colsF, geoTrans, proj, dataType, nodataval] = RasterInfo(final_mix_vrt);
    [~, rowsB, colsB, ~,       ~,    ~,        ~         ] = RasterInfo(bathy_vrt);
    [~, rowsL, colsL, ~,       ~,    ~,        ~         ] = RasterInfo(LCC_vrt);

    if rowsF ~= rowsB || colsF ~= colsB || rowsF ~= rowsL || colsF ~= colsL
        error('Grid size mismatch for %s %dm tile output: final=%d/%d, bathy=%d/%d, mask=%d/%d', ...
            BasinName, r, rowsF, colsF, rowsB, colsB, rowsL, colsL);
    end

    % % 以两者的交集作为安全范围
    % rows = min([rowsF, rowsL]);
    % cols = min([colsF, colsL]);
    rows = rowsF;
    cols = colsF;

    % ---- 5. 循环生成每个点的 tile ----
    for k = 1:nR
        % 当前点在原始 P_select 中的行号
        % 索引如果要用全局 PointID，可以直接用 P_r.PointID(k)
        PointID   = P_r.PointID(k);
        Points_Lon = P_r.X(k);
        Points_Lat = P_r.Y(k);

        % 中心像元
        [row0, col0] = Proj2RowCol(geoTrans, Points_Lat, Points_Lon);

        % 窗口左上角（注意这里是 1-based）
        r1 = row0 - hr;
        c1 = col0 - hc;
        h  = winH;
        w  = winW;

        % % ---- 边界检查：1-based，且保证窗口完整落在栅格内 ----
        % % 其实没有必要，前面都已经检查过了
        % if r1 < 1 || c1 < 1 || (r1 + h - 1) > rows || (c1 + w - 1) > cols
        %     fprintf('Res = %dm: PointID = %d 在边界附近，窗口越界，跳过该点。\n', ...
        %             r, PointID);
        %     continue;
        % end

        % ---- 读取这个 tile 的数据 ----
        tile_bathy = double(ReadRaster(bathy_vrt, r1, c1, h, w));
        tile_LCC_raw = double(ReadRaster(LCC_vrt, r1, c1, h, w));
        tileBathmatry3DEP = ReadRaster(final_mix_vrt, r1, c1, h, w);

        % bathy 有效区
        bathy_valid = isfinite(tile_bathy) & ~isnan(tile_bathy) & ...
            (tile_bathy ~= -999999) & (tile_bathy > -1e20);

        % simple final mask:
        % 1 = 需要预测 / 需要从输入中遮掉
        % 0 = 已知区域 / 不预测
        tile_finalMask = (tile_LCC_raw == 1) & bathy_valid;

        % 用 final mask 遮掉 bathy+3DEP 输入中的预测区域
        tileBathmatry3DEP_outRiver = tileBathmatry3DEP;
        tileBathmatry3DEP_outRiver(tile_finalMask == 1) = nan;

        % ---- 写出各类 tiff ----
        outFormat = 'GTiff';
        subgeoTrans = subTranscoef(geoTrans, r1, c1);

        % 1) LCC mask
        dataType_mask  = 1;    % Byte
        nodataval_mask = 255;
        fileRas_LCC_Mask = fullfile(LCC_Mask_Folder, ...
            sprintf('Select_tile_%dm_%s_ID%d_LCC_Mask.tif', ...
            r, BasinName, PointID));
        WriteRaster(fileRas_LCC_Mask, double(uint8(tile_finalMask)), subgeoTrans, proj, ...
            dataType_mask, outFormat, nodataval_mask);

        % 2) mask 后的 bathy+3DEP（只保留非 80 的部分）
        fileRas_Bath3DEP_outRiver = fullfile(TileOutRiver_Folder, ...
            sprintf('Select_tileOutRiver_%dm_%s_ID%d.tif', ...
            r, BasinName, PointID));
        WriteRaster(fileRas_Bath3DEP_outRiver, tileBathmatry3DEP_outRiver, ...
            subgeoTrans, proj, dataType, outFormat, nodataval);

        % 3) 完整的 bathy+3DEP（不 mask）
        fileRas_Bath3DEP_all = fullfile(Train_tile_Folder, ...
            sprintf('Select_tile_Basin_%dm_%s_ID%d.tif', ...
            r, BasinName, PointID));
        WriteRaster(fileRas_Bath3DEP_all, tileBathmatry3DEP, ...
            subgeoTrans, proj, dataType, outFormat, nodataval);

        fprintf('Res = %dm: PointID = %d Tile extraction is done!\n', r, PointID);
        % 清理临时变量
        clear subgeoTrans tile_bathy tile_LCC_raw tile_finalMask bathy_valid ...
            tileBathmatry3DEP tileBathmatry3DEP_outRiver
        clear Points_Lon Points_Lat row0 col0 r1 c1 h w
        clear PointID
    end
end


