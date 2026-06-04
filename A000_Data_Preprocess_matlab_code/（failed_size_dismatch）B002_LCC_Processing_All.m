%% 2025年11月4日

% 这部分后续提供给陈老师，用于提取河道中心线，经过校对的
%% 重新处理LCC，保持10m，但是按照bathmatry的坐标系，按照范围进行裁剪
% 先用 bathy 的范围 → 投到 LCC → 用行列号从 LCC.vrt 抠子栅格（仍是 LCC、像元=10m 不变）→ 再把这个子栅格重投到 WGS84。
% 与原来的tiff存在不一致，非常奇怪
% 分块写出最后是空值
% 确认了，/tank/data/SFS/xinyis/data/LCC/fileList.txt 指向的是另外的数据
% /tank/data/SFS/xinyis/data/LCC/ESA_WorldCover_10m_2021_V200_N30W114_Map.tif


% 重新创建LCC的vrt
clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

List_text='/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/filelist_new.txt';   % 确保是绝对路径 & Unix 换行
dstfile='/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_new.vrt';
cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
    'export PROJ_NETWORK=ON; ' ...                 % 可选
    'gdalbuildvrt -overwrite -input_file_list "%s" "%s"'], ...
    List_text, dstfile);
status = system(cmd);


% 补充下载后重新vrt构建
clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

List_text='/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/filelist_mix.txt';   % 确保是绝对路径 & Unix 换行
dstfile='/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_mix.vrt';
cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
    'export PROJ_NETWORK=ON; ' ...                 % 可选
    'gdalbuildvrt -overwrite -input_file_list "%s" "%s"'], ...
    List_text, dstfile);
status = system(cmd);


%% 进行裁剪 
% 裁剪使用到的范围是bathy的范围，然后向外扩充100个栅格，从而保证范围的覆盖

clear;clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

% LCC_vrt = '/tank/data/SFS/xinyis/data/LCC/LCC.vrt';
% LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_new.vrt';
LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_mix.vrt';
OutFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_10m_Transfer/';
if exist(OutFolder,'dir') ~= 7
    mkdir(OutFolder);
end

[~,rowsLCC,colsLCC,gtLCC,projLCC,dataTypeLCC,nodatavalLCC] = RasterInfo(LCC_vrt);  % projLCC 应该是 LCC 的 WKT
if ~contains(upper(projLCC),'WGS 84') && ~contains(upper(projLCC),'EPSG','ignorecase',true)
    error('当前 LCC_vrt 不是 WGS84，别用本脚本；用我上一版的“先 srcwin 后 warp”的方案。');
end

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
d = dir(Folder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];

% ===== 缺失/失败日志 =====
% logCSV = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_Supply/missing_or_fail.csv';
logCSV = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_Supply/missing_or_fail_2.csv';
if exist(logCSV,'file') ~= 2
    fidLog = fopen(logCSV,'w');
    fprintf(fidLog,'River,Reason,W,E,S,N,Note\n');
    fclose(fidLog);
end

nodataval = -99999;

for i = 1 : length(d)
    if contains(d(i).name,'NoNeed') || contains(d(i).name,'milwaukee_river_3DEP')
        continue
    end

    bathy_vrt = fullfile([Folder,d(i).name,'/Bathy.vrt']);
    dstfolder = fullfile([OutFolder,d(i).name,'/']);
    if exist(dstfolder,'dir') ~= 7
        mkdir(dstfolder);
    end

    dstFile = fullfile([dstfolder,'/ESA_WorldCover_Clip_WGS84.tif']);

    % 1) 读 bathy 的范围（在 bathy 自己的投影下）
    [~, rowsB, colsB, gtBathy, projBathy, ~, ~] = RasterInfo(bathy_vrt);
    % 计算四角（注意：一般北上图，gt = [x0, dx, 0, y0, 0, -dy]）
    x0 = gtBathy(1); y0 = gtBathy(4);
    x1 = gtBathy(1) + colsB*gtBathy(2) + rowsB*gtBathy(3);
    y1 = gtBathy(4) + colsB*gtBathy(5) + rowsB*gtBathy(6);
    xx = sort([x0, x1]); yy = sort([y0, y1]);
    x0 = xx(1); x1 = xx(2); y0 = yy(1); y1 = yy(2);

    % —— 2) 把四角从 bathy 投到 WGS84（EPSG:4326）
    wkttmp_src = [tempname, '.wkt']; fid = fopen(wkttmp_src,'w'); fwrite(fid, projBathy); fclose(fid);

    % 把 4 个点写到临时文件（每行 x y）
    ptstmp = [tempname, '.txt'];
    fid = fopen(ptstmp,'w');
    fprintf(fid, '%.12f %.12f\n', [x0 y0; x1 y0; x1 y1; x0 y1].');
    fclose(fid);

    % 用重定向把点喂给 gdaltransform —— 不再用 printf
    cmd_tr = sprintf('gdaltransform -s_srs "ESRI::%s" -t_srs EPSG:4326 < "%s"', wkttmp_src, ptstmp);
    [st_tr, out_tr] = system(cmd_tr);

    % delete(wkttmp_src); delete(ptstmp);

    if st_tr ~= 0
        error('gdaltransform 失败：\n%s', out_tr);
    end

    % 解析输出（可能有 z，也可能没有）
    C = textscan(out_tr, '%f %f %f');
    if isempty(C{1}) || numel(C{1}) < 4
        C2 = textscan(out_tr, '%f %f');
        if isempty(C2{1}) || numel(C2{1}) < 4
            error('gdaltransform 输出为空或不足四点：\n%s', out_tr);
        else
            lon = C2{1}; lat = C2{2};
        end
    else
        lon = C{1}; lat = C{2};
    end

    xminL = min(lon); xmaxL = max(lon);
    yminL = min(lat); ymaxL = max(lat);

    fprintf('Corner Lon/Lat transformed. [W E S N] = [%.8f %.8f %.8f %.8f]\n', ...
        xminL, xmaxL, yminL, ymaxL);

    % 得到在LCC投影下的行列号
    [rowLCC0,colLCC0]=Proj2RowCol(gtLCC,yminL,xminL);
    [rowLCC1,colLCC1]=Proj2RowCol(gtLCC,yminL,xmaxL);
    [rowLCC2,colLCC2]=Proj2RowCol(gtLCC,ymaxL,xminL);
    [rowLCC3,colLCC3]=Proj2RowCol(gtLCC,ymaxL,xmaxL);

    col0 = min([colLCC0,colLCC1,colLCC2,colLCC3]);
    col1 = max([colLCC0,colLCC1,colLCC2,colLCC3]);
    row0 = min([rowLCC0,rowLCC1,rowLCC2,rowLCC3]);
    row1 = max([rowLCC0,rowLCC1,rowLCC2,rowLCC3]);

    % 添加一个buff，用于确保范围的
    buff = 100;
    col0 = col0 - buff;
    col1 = col1 + buff;
    row0 = row0 - buff;
    row1 = row1 + buff;

    % === 简单判定：只要 LCC 没覆盖到（任一边越界）就记日志 ===
    if (col0 < 1) || (row0 < 1) || (col1 > colsLCC) || (row1 > rowsLCC)
        name = d(i).name;
        reason = 'LCC_not_fully_cover_bathy';
        W = xminL; E = xmaxL; S = yminL; N = ymaxL;   % 转到 WGS84 后的范围
        note = sprintf('req_win=[r%g:%g,c%g:%g]; LCC=[1..%d,1..%d]', ...
            row0, row1, col0, col1, rowsLCC, colsLCC);
        fidLog = fopen(logCSV,'a');
        fprintf(fidLog,'%s,%s,%.10f,%.10f,%.10f,%.10f,%s\n', ...
            name, reason, W, E, S, N, strrep(note, ',', ';'));
        fclose(fidLog);

        continue
    end

    % 保证不超出LCC大图的范围
    if col0 < 0
        col0 = 1;
    end
    if col1 > colsLCC
        col1 = colsLCC;
    end
    if row0 < 0
        row0 = 1;
    end
    if row1 > rowsLCC
        row1 = rowsLCC;
    end

    xSize = max(1, col1 - col0+1);
    ySize = max(1, row1 - row0+1);

    % 部分河流的非常大，这个时候只能尝试分块读取写出，最后拼合

    if xSize < 8000 && ySize < 8000
        LCC_range = ReadRaster(LCC_vrt,row0,col0,row1-row0+1,col1-col0+1); % 读取范围数据
        
        % 仅保留80（河道）为1，其余置为 0
        out = zeros(ySize, xSize);
        mask = (LCC_range == 80);
        out(mask) = 1;

        subgeoTrans = subTranscoef(gtLCC,row0,col0);
        outFormat = 'GTiff';
        WriteRaster(dstFile,out,subgeoTrans,projLCC,dataTypeLCC,outFormat,nodataval)
        clear out mask
    else
        tile = 2048;                                   % 可调

        % 全幅尺寸、仿射与投影（已从 RasterInfo 取到：rows, cols, geoTrans, proj）
        rows = ySize;
        cols = xSize;
        totalTiles = ceil(rows/tile) * ceil(cols/tile);
        tileCount = 0;

        subgeoTrans = subTranscoef(gtLCC,row0,col0);

        for rLocal = 1:tile:rows
            rr = min(tile, rows - rLocal + 1);
            for cLocal = 1:tile:cols
                cc = min(tile, cols - cLocal + 1);

                % ★ 读：全图绝对索引 = 子图左上 + 局部偏移
                absRow = row0 + rLocal - 1;
                absCol = col0 + cLocal - 1;
                block  = ReadRaster(LCC_vrt, absRow, absCol, rr, cc);

                % 仅保留80（河道）为1，其余置为 0
                out = zeros(rr, cc);
                mask = (block == 80);
                out(mask) = 1;

                % 分块写入（11 参调用；第一次会按 rows/cols 创建整幅，再写当前块）
                WriteRaster(dstFile, out, subgeoTrans, projLCC, dataTypeLCC, ...
                    'GTiff', nodataval, ...
                    rLocal, cLocal, rows, cols);

                % 极简进度
                tileCount = tileCount + 1;
                fprintf('\rProgress: %6.2f%%  (%d/%d)', ...
                    100*tileCount/totalTiles, tileCount, totalTiles);
                clear mask block out
            end
        end
        fprintf('\nDone. %s\n', dstFile);

        clear rows cols totalTiles
    end

    clear LCC_range subgeoTrans
    clear rowsB colsB gtBathy projBathy
    clear bathy_vrt dstFile dstfolder
    clear xminL xmaxL yminL ymaxL col0 col1 row0 row1 ...
        x0 x1 y0 y1 xx yy
    disp([num2str(i),' LCC Clip is done'])
end

% MIlwaukee river

clear;clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

% LCC_vrt = '/tank/data/SFS/xinyis/data/LCC/LCC.vrt';
% LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_new.vrt';
LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_mix.vrt';
OutFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_10m_Transfer/';
if exist(OutFolder,'dir') ~= 7
    mkdir(OutFolder);
end

[~,rowsLCC,colsLCC,gtLCC,projLCC,dataTypeLCC,nodatavalLCC] = RasterInfo(LCC_vrt);  % projLCC 应该是 LCC 的 WKT
if ~contains(upper(projLCC),'WGS 84') && ~contains(upper(projLCC),'EPSG','ignorecase',true)
    error('当前 LCC_vrt 不是 WGS84，别用本脚本；用我上一版的“先 srcwin 后 warp”的方案。');
end

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
Tifs = [
    dir(fullfile(Folder, '*.tif'));
    dir(fullfile(Folder, '*.vrt'));
    ];

Tifs = Tifs(~[Tifs.isdir]);

% ===== 缺失/失败日志 =====
% logCSV = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_Supply/missing_or_fail.csv';
logCSV = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_Origin/LCC_Supply/missing_or_fail_2.csv';
if exist(logCSV,'file') ~= 2
    fidLog = fopen(logCSV,'w');
    fprintf(fidLog,'River,Reason,W,E,S,N,Note\n');
    fclose(fidLog);
end

nodataval = -99999;

for i = 1 : length(Tifs)
    if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif') ||  contains(Tifs(i).name,'_DEM_1m_Proj')
        continue
    end

    bathy_vrt = fullfile([Folder,Tifs(i).name]);
    [~, baseName, ~] = fileparts(bathy_vrt);   % baseName 就是不含 .tif 的名字
    
    dstfolder = fullfile([OutFolder,baseName,'/']);
    if exist(dstfolder,'dir') ~= 7
        mkdir(dstfolder);
    end

    dstFile = fullfile([dstfolder,'/ESA_WorldCover_Clip_WGS84.tif']);

    % 1) 读 bathy 的范围（在 bathy 自己的投影下）
    [~, rowsB, colsB, gtBathy, projBathy, ~, ~] = RasterInfo(bathy_vrt);
    % 计算四角（注意：一般北上图，gt = [x0, dx, 0, y0, 0, -dy]）
    x0 = gtBathy(1); y0 = gtBathy(4);
    x1 = gtBathy(1) + colsB*gtBathy(2) + rowsB*gtBathy(3);
    y1 = gtBathy(4) + colsB*gtBathy(5) + rowsB*gtBathy(6);
    xx = sort([x0, x1]); yy = sort([y0, y1]);
    x0 = xx(1); x1 = xx(2); y0 = yy(1); y1 = yy(2);

    % —— 2) 把四角从 bathy 投到 WGS84（EPSG:4326）
    wkttmp_src = [tempname, '.wkt']; fid = fopen(wkttmp_src,'w'); fwrite(fid, projBathy); fclose(fid);

    % 把 4 个点写到临时文件（每行 x y）
    ptstmp = [tempname, '.txt'];
    fid = fopen(ptstmp,'w');
    fprintf(fid, '%.12f %.12f\n', [x0 y0; x1 y0; x1 y1; x0 y1].');
    fclose(fid);

    % 用重定向把点喂给 gdaltransform —— 不再用 printf
    cmd_tr = sprintf('gdaltransform -s_srs "ESRI::%s" -t_srs EPSG:4326 < "%s"', wkttmp_src, ptstmp);
    [st_tr, out_tr] = system(cmd_tr);

    % delete(wkttmp_src); delete(ptstmp);

    if st_tr ~= 0
        error('gdaltransform 失败：\n%s', out_tr);
    end

    % 解析输出（可能有 z，也可能没有）
    C = textscan(out_tr, '%f %f %f');
    if isempty(C{1}) || numel(C{1}) < 4
        C2 = textscan(out_tr, '%f %f');
        if isempty(C2{1}) || numel(C2{1}) < 4
            error('gdaltransform 输出为空或不足四点：\n%s', out_tr);
        else
            lon = C2{1}; lat = C2{2};
        end
    else
        lon = C{1}; lat = C{2};
    end

    xminL = min(lon); xmaxL = max(lon);
    yminL = min(lat); ymaxL = max(lat);

    fprintf('Corner Lon/Lat transformed. [W E S N] = [%.8f %.8f %.8f %.8f]\n', ...
        xminL, xmaxL, yminL, ymaxL);

    % 得到在LCC投影下的行列号
    [rowLCC0,colLCC0]=Proj2RowCol(gtLCC,yminL,xminL);
    [rowLCC1,colLCC1]=Proj2RowCol(gtLCC,yminL,xmaxL);
    [rowLCC2,colLCC2]=Proj2RowCol(gtLCC,ymaxL,xminL);
    [rowLCC3,colLCC3]=Proj2RowCol(gtLCC,ymaxL,xmaxL);

    col0 = min([colLCC0,colLCC1,colLCC2,colLCC3]);
    col1 = max([colLCC0,colLCC1,colLCC2,colLCC3]);
    row0 = min([rowLCC0,rowLCC1,rowLCC2,rowLCC3]);
    row1 = max([rowLCC0,rowLCC1,rowLCC2,rowLCC3]);

    % 添加一个buff，用于确保范围的
    buff = 100;
    col0 = col0 - buff;
    col1 = col1 + buff;
    row0 = row0 - buff;
    row1 = row1 + buff;


    % === 简单判定：只要 LCC 没覆盖到（任一边越界）就记日志 ===
    if (col0 < 1) || (row0 < 1) || (col1 > colsLCC) || (row1 > rowsLCC)
        name = d(i).name;
        reason = 'LCC_not_fully_cover_bathy';
        W = xminL; E = xmaxL; S = yminL; N = ymaxL;   % 转到 WGS84 后的范围
        note = sprintf('req_win=[r%g:%g,c%g:%g]; LCC=[1..%d,1..%d]', ...
            row0, row1, col0, col1, rowsLCC, colsLCC);
        fidLog = fopen(logCSV,'a');
        fprintf(fidLog,'%s,%s,%.10f,%.10f,%.10f,%.10f,%s\n', ...
            name, reason, W, E, S, N, strrep(note, ',', ';'));
        fclose(fidLog);

        continue
    end

    % 保证不超出LCC大图的范围
    if col0 < 0
        col0 = 1;
    end
    if col1 > colsLCC
        col1 = colsLCC;
    end
    if row0 < 0
        row0 = 1;
    end
    if row1 > rowsLCC
        row1 = rowsLCC;
    end

    xSize = max(1, col1 - col0+1);
    ySize = max(1, row1 - row0+1);

    % 部分河流的非常大，这个时候只能尝试分块读取写出，最后拼合

    if xSize < 8000 && ySize < 8000
        LCC_range = ReadRaster(LCC_vrt,row0,col0,row1-row0+1,col1-col0+1); % 读取范围数据
        
        % 仅保留80（河道）为1，其余置为 0
        out = zeros(ySize, xSize);
        mask = (LCC_range == 80);
        out(mask) = 1;

        subgeoTrans = subTranscoef(gtLCC,row0,col0);
        outFormat = 'GTiff';
        WriteRaster(dstFile,out,subgeoTrans,projLCC,dataTypeLCC,outFormat,nodataval)
        clear out mask
    else
        tile = 2048;                                   % 可调

        % 全幅尺寸、仿射与投影（已从 RasterInfo 取到：rows, cols, geoTrans, proj）
        rows = ySize;
        cols = xSize;
        totalTiles = ceil(rows/tile) * ceil(cols/tile);
        tileCount = 0;

        subgeoTrans = subTranscoef(gtLCC,row0,col0);

        for rLocal = 1:tile:rows
            rr = min(tile, rows - rLocal + 1);
            for cLocal = 1:tile:cols
                cc = min(tile, cols - cLocal + 1);

                % ★ 读：全图绝对索引 = 子图左上 + 局部偏移
                absRow = row0 + rLocal - 1;
                absCol = col0 + cLocal - 1;
                block  = ReadRaster(LCC_vrt, absRow, absCol, rr, cc);

                % 仅保留80（河道）为1，其余置为 0
                out = zeros(rr, cc);
                mask = (block == 80);
                out(mask) = 1;

                % 分块写入（11 参调用；第一次会按 rows/cols 创建整幅，再写当前块）
                WriteRaster(dstFile, out, subgeoTrans, projLCC, dataTypeLCC, ...
                    'GTiff', nodataval, ...
                    rLocal, cLocal, rows, cols);

                % 极简进度
                tileCount = tileCount + 1;
                fprintf('\rProgress: %6.2f%%  (%d/%d)', ...
                    100*tileCount/totalTiles, tileCount, totalTiles);
                clear mask block out
            end
        end
        fprintf('\nDone. %s\n', dstFile);

        clear rows cols totalTiles
    end

    clear LCC_range subgeoTrans
    clear rowsB colsB gtBathy projBathy
    clear bathy_vrt dstFile dstfolder
    clear xminL xmaxL yminL ymaxL col0 col1 row0 row1 ...
        x0 x1 y0 y1 xx yy
    disp([num2str(i),' LCC Clip is done'])
end


%% 使用新的裁剪后的LCC重新生成河流vrt对应的LCC_1m, LCC_3m, LCC_5m

clear;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

LCC_ClipFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Transfer_to_Dr_Chen/LCC_10m_Transfer/';

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
d = dir(Folder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];

% 目标分辨率（单位：米）
targetRes = [1, 3, 5, 10]; % 10m相当于重投影的
for j = 1 : length(targetRes)
    res = targetRes(j);
    OutFolder = fullfile(['/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_',num2str(res),'m/']);
    if exist(OutFolder,'dir') ~= 7
        mkdir(OutFolder);
    end

    for i = 1 : length(d)
        if contains(d(i).name,'NoNeed') || contains(d(i).name,'milwaukee_river_3DEP')
            continue
        end

        LCC_vrt = fullfile([LCC_ClipFolder,d(i).name,'/ESA_WorldCover_Clip_WGS84.tif']);

        bathy_vrt = fullfile([Folder,d(i).name,'/Bathy.vrt']);
        dstfolder = fullfile([OutFolder,d(i).name]);
        if exist(dstfolder,'dir') ~= 7
            mkdir(dstfolder);
        end

        dstFile = fullfile([dstfolder,'/ESA_WorldCover_Resampleandclip_',num2str(res),'m.vrt']);

        [~,rows,cols,geoTrans,proj,~,~]=RasterInfo(bathy_vrt);

        % 已有：geoTrans, proj(WKT), cols, rows, dep3_vrt, dstFile
        xmin = geoTrans(1);
        xres1 = geoTrans(2);      % 1 m（正）
        ymax = geoTrans(4);
        yres1 = geoTrans(6);      % -1 m（负）
        xmax = xmin + cols * xres1;
        ymin = ymax + rows * yres1;

        % === 目标改为 res m ===
        xres = xres1 * res;                      % 
        yres = abs(yres1) * res;                 % 


        % 用单引号包住 WKT（注意 MATLAB 里单引号要写成两个：''）
        proj_arg = sprintf('''%s''', proj);

        % 像元分辨率（与 bathy 一致），-tap 需要 -tr
        tr_arg = sprintf('-tr %.10f %.10f', xres, abs(yres));

        nd_arg = '';  % <= 关键：不传 -srcnodata/-dstnodata

        % 用最近邻；分类数据保持 Byte
        cmd = sprintf([ ...
            'gdalwarp -of VRT -ot Byte -r near -multi -wo NUM_THREADS=ALL_CPUS -wm 2048 ' ...
            '-t_srs %s -te_srs %s %s -te %.10f %.10f %.10f %.10f -tap ' ...
            '%s' ... % 可选的 nodata 片段
            '-overwrite "%s" "%s"' ], ...
            proj_arg, proj_arg, tr_arg, ...
            xmin, ymin, xmax, ymax, ...
            nd_arg, LCC_vrt, dstFile);

        status = system(cmd);
        if status ~= 0
            error('gdalwarp 生成 VRT 失败（检查 WKT 引号、-tr/-tap、nodata）。');
        end

        clear LCC_vrt
        clear bathy_vrt dstFile Output_folder
        clear geoTrans proj cols rows
        disp([num2str(res),'m ',num2str(i),' LCC vrt Resampleandclip is done'])
    end
end

% Milwaukee River

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
Tifs = [
    dir(fullfile(Folder, '*.tif'));
    dir(fullfile(Folder, '*.vrt'));
    ];

Tifs = Tifs(~[Tifs.isdir]);

LCC_ClipFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Transfer_to_Dr_Chen/LCC_10m_Transfer/';
% 目标分辨率（单位：米）
targetRes = [1, 3, 5, 10]; % 10m相当于重投影的
for j = 1 : length(targetRes)
    res = targetRes(j);
    OutFolder = fullfile(['/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_',num2str(res),'m/']);
    if exist(OutFolder,'dir') ~= 7
        mkdir(OutFolder);
    end

    for i = 1 : length(Tifs)
        if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif') ||  contains(Tifs(i).name,'_DEM_1m_Proj')
            continue
        end

        bathy_vrt = fullfile([Folder,Tifs(i).name]);
        [~, baseName, ~] = fileparts(bathy_vrt);   % baseName 就是不含 .tif 的名字

        LCC_vrt = fullfile([LCC_ClipFolder,baseName,'/ESA_WorldCover_Clip_WGS84.tif']);

        dstfolder = fullfile([OutFolder,baseName]);
        if exist(dstfolder,'dir') ~= 7
            mkdir(dstfolder);
        end

        dstFile = fullfile([dstfolder,'/ESA_WorldCover_Resampleandclip_',num2str(res),'m.vrt']);

        [~,rows,cols,geoTrans,proj,~,~]=RasterInfo(bathy_vrt);

        % 已有：geoTrans, proj(WKT), cols, rows, dep3_vrt, dstFile
        xmin = geoTrans(1);
        xres1 = geoTrans(2);      % 1 m（正）
        ymax = geoTrans(4);
        yres1 = geoTrans(6);      % -1 m（负）
        xmax = xmin + cols * xres1;
        ymin = ymax + rows * yres1;

        % === 目标改为 res m ===
        xres = xres1 * res;                      % 
        yres = abs(yres1) * res;                 % 

        % 用单引号包住 WKT（注意 MATLAB 里单引号要写成两个：''）
        proj_arg = sprintf('''%s''', proj);

        % 像元分辨率（与 bathy 一致），-tap 需要 -tr
        tr_arg = sprintf('-tr %.10f %.10f', xres, abs(yres));

        nd_arg = '';  % <= 关键：不传 -srcnodata/-dstnodata

        % 用最近邻；分类数据保持 Byte
        cmd = sprintf([ ...
            'gdalwarp -of VRT -ot Byte -r near -multi -wo NUM_THREADS=ALL_CPUS -wm 2048 ' ...
            '-t_srs %s -te_srs %s %s -te %.10f %.10f %.10f %.10f -tap ' ...
            '%s' ... % 可选的 nodata 片段
            '-overwrite "%s" "%s"' ], ...
            proj_arg, proj_arg, tr_arg, ...
            xmin, ymin, xmax, ymax, ...
            nd_arg, LCC_vrt, dstFile);

        status = system(cmd);
        if status ~= 0
            error('gdalwarp 生成 VRT 失败（检查 WKT 引号、-tr/-tap、nodata）。');
        end

        clear LCC_vrt
        clear bathy_vrt dstFile Output_folder
        clear geoTrans proj cols rows
        disp([num2str(res),'m ',num2str(i),' LCC vrt Resampleandclip is done'])
    end
end




%% 对陈老师的结果进行重投影


% 批量将 Width.shp 重投影为与各河流 Bathy 相同的投影
% 读取 Bathy 投影用你现有的 RasterInfo；源 Width.shp 默认是 WGS84（EPSG:4326）
% 输出到 Reproj 目录：.../CenterRiverLine_skel/Reproj/<River>/ESA_WorldCover_Width_proj.shp

clear;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

bathyRoot = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
skelOriginRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/CenterRiverLine_skel/Origin';
skelOutRoot    = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/CenterRiverLine_skel/Reproj';
if exist(skelOutRoot,'dir') ~= 7, mkdir(skelOutRoot); end

% 扫描所有河流（以 Bathy 目录为准）
d = dir(bathyRoot); d = d([d.isdir]); d(1:2) = [];

% 日志
logCSV = fullfile(skelOutRoot,'reproj_log.csv');
fid = fopen(logCSV,'w');
fprintf(fid,'River,Status,Message\n');
fclose(fid);

for i = 1:numel(d)
    river = d(i).name;
    if contains(river,'NoNeed') || contains(river,'milwaukee_river_3DEP')
        continue
    end

    % 输入 shapefile（WGS84）
    inDir = fullfile(skelOriginRoot, river);
    inSHP = fullfile(inDir, 'ESA_WorldCover_Clip_WGS84_Width.shp');
    if exist(inSHP,'file') ~= 2
        % 该河流没有 Width.shp，记日志跳过
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,SKIP,No Width.shp\n', river);
        fclose(fid);
        fprintf('[%s] Skip: no Width.shp\n', river);
        continue
    end

    % 目标投影：Bathy.vrt 的投影
    bathyVRT = fullfile(bathyRoot, river, 'Bathy.vrt');
    if exist(bathyVRT,'file') ~= 2
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,SKIP,No Bathy.vrt\n', river);
        fclose(fid);
        fprintf('[%s] Skip: no Bathy.vrt\n', river);
        continue
    end

    % 直接用 RasterInfo 取 WKT（proj 是一整段 WKT）
    try
        [~,~,~,~,projWKT,~,~] = RasterInfo(bathyVRT);
    catch ME
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,FAIL,Read RasterInfo error: %s\n', river, ME.message);
        fclose(fid);
        fprintf('[%s] FAIL: RasterInfo error: %s\n', river, ME.message);
        continue
    end

    % 输出路径
    outDir = fullfile(skelOutRoot, river);
    if exist(outDir,'dir') ~= 7, mkdir(outDir); end
    outSHP = fullfile(outDir, 'ESA_WorldCover_Width_proj.shp');

    % 构造 ogr2ogr 命令
    % 源坐标：若 .prj 不确定，显式给 EPSG:4326 更保险；t_srs 用从 Bathy 读出的 WKT
    srcPrj = fullfile(inDir,'ESA_WorldCover_Clip_WGS84_Width.prj');
    if exist(srcPrj,'file') == 2
        s_srs_arg = '';              % shapefile 自带 .prj 就让 ogr2ogr 自动识别
    else
        s_srs_arg = '-s_srs EPSG:4326';  % 没 .prj 就显式指定
    end

    % WKT 里有空格和括号，用单引号包起来；在 sprintf 里用两个单引号转义
    t_srs_arg = sprintf('''%s''', projWKT);

    cmd = sprintf(['ogr2ogr -f "ESRI Shapefile" -overwrite -skipfailures ' ...
                   '%s -t_srs %s "%s" "%s"'], ...
                   s_srs_arg, t_srs_arg, outSHP, inSHP);

    % 执行
    [st, msg] = system(cmd);
    if st == 0 && exist(outSHP,'file') == 2
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,OK,Reprojected\n', river);
        fclose(fid);
        fprintf('[%s] OK -> %s\n', river, outSHP);
    else
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,FAIL,%s\n', river, strrep(strtrim(msg), ',', ';'));
        fclose(fid);
        fprintf('[%s] FAIL: %s\n', river, msg);
    end
end

disp('All done. See log:'); disp(logCSV);

% Milwaukee river

clear;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
Tifs = [
    dir(fullfile(Folder, '*.tif'));
    dir(fullfile(Folder, '*.vrt'));
    ];

Tifs = Tifs(~[Tifs.isdir]);

skelOriginRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/CenterRiverLine_skel/Origin';
skelOutRoot    = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/CenterRiverLine_skel/Reproj';
if exist(skelOutRoot,'dir') ~= 7, mkdir(skelOutRoot); end

% 日志
logCSV = fullfile(skelOutRoot,'reproj_log.csv');
fid = fopen(logCSV,'w');
fprintf(fid,'River,Status,Message\n');
fclose(fid);

for i = 1 : length(Tifs)
    if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif') ||  contains(Tifs(i).name,'_DEM_1m_Proj')
        continue
    end

    bathyVRT = fullfile([Folder,Tifs(i).name]);
    [~, baseName, ~] = fileparts(bathyVRT);   % baseName 就是不含 .tif 的名字
    river = baseName;

    % 输入 shapefile（WGS84）
    inDir = fullfile(skelOriginRoot, river);
    inSHP = fullfile(inDir, 'ESA_WorldCover_Clip_WGS84_Width.shp');
    if exist(inSHP,'file') ~= 2
        % 该河流没有 Width.shp，记日志跳过
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,SKIP,No Width.shp\n', river);
        fclose(fid);
        fprintf('[%s] Skip: no Width.shp\n', river);
        continue
    end

    % 目标投影：Bathy.vrt 的投影
    if exist(bathyVRT,'file') ~= 2
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,SKIP,No Bathy.vrt\n', river);
        fclose(fid);
        fprintf('[%s] Skip: no Bathy.vrt\n', river);
        continue
    end

    % 直接用 RasterInfo 取 WKT（proj 是一整段 WKT）
    try
        [~,~,~,~,projWKT,~,~] = RasterInfo(bathyVRT);
    catch ME
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,FAIL,Read RasterInfo error: %s\n', river, ME.message);
        fclose(fid);
        fprintf('[%s] FAIL: RasterInfo error: %s\n', river, ME.message);
        continue
    end

    % 输出路径
    outDir = fullfile(skelOutRoot, river);
    if exist(outDir,'dir') ~= 7, mkdir(outDir); end
    outSHP = fullfile(outDir, 'ESA_WorldCover_Width_proj.shp');

    % 构造 ogr2ogr 命令
    % 源坐标：若 .prj 不确定，显式给 EPSG:4326 更保险；t_srs 用从 Bathy 读出的 WKT
    srcPrj = fullfile(inDir,'ESA_WorldCover_Clip_WGS84_Width.prj');
    if exist(srcPrj,'file') == 2
        s_srs_arg = '';              % shapefile 自带 .prj 就让 ogr2ogr 自动识别
    else
        s_srs_arg = '-s_srs EPSG:4326';  % 没 .prj 就显式指定
    end

    % WKT 里有空格和括号，用单引号包起来；在 sprintf 里用两个单引号转义
    t_srs_arg = sprintf('''%s''', projWKT);

    cmd = sprintf(['ogr2ogr -f "ESRI Shapefile" -overwrite -skipfailures ' ...
                   '%s -t_srs %s "%s" "%s"'], ...
                   s_srs_arg, t_srs_arg, outSHP, inSHP);

    % 执行
    [st, msg] = system(cmd);
    if st == 0 && exist(outSHP,'file') == 2
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,OK,Reprojected\n', river);
        fclose(fid);
        fprintf('[%s] OK -> %s\n', river, outSHP);
    else
        fid = fopen(logCSV,'a');
        fprintf(fid,'%s,FAIL,%s\n', river, strrep(strtrim(msg), ',', ';'));
        fclose(fid);
        fprintf('[%s] FAIL: %s\n', river, msg);
    end
end

disp('All done. See log:'); disp(logCSV);






























%% 处理LCC，重采样并裁剪到对应河流段上

% clear;clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
%
% LCC_vrt = '/tank/data/SFS/xinyis/data/LCC/LCC.vrt';
%
% Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
% d = dir(Folder);                 % 列出该目录下的所有条目
% d = d([d.isdir]);                % 只保留文件夹
% d(1:2) = [];
%
% for i = 12 %: length(d)
%     if contains(d(i).name,'MD_PotomacRiver') || contains(d(i).name,'NoNeed') || contains(d(i).name,'milwaukee_river_3DEP')
%         continue
%     end
%
%     bathy_vrt = fullfile([Folder,d(i).name,'/Bathy.vrt']);
%     Output_folder = fullfile([Folder,d(i).name,'/LCC']);
%     if exist(Output_folder,'dir') ~= 7
%         mkdir(Output_folder);
%     end
%
%     dstFile = fullfile([Output_folder,'/ESA_WorldCover_Resampleandclip.tif']);
%
%     [~,rows,cols,geoTrans,proj,~,~]=RasterInfo(bathy_vrt);
%
%     ResampleAndClip(geoTrans, proj, cols,...
%         rows, LCC_vrt, dstFile, 'GTiff', 1, 1); % use 1 nearest
%
%     clear bathy_vrt dstFile Output_folder
%     clear geoTrans proj cols rows
%     disp([num2str(i),' LCC vrt Resampleandclip is done'])
% end
%
% % MIlwaukee river
% clear; clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
%
% Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
% Tifs = [
%     dir(fullfile(Folder, '*.tif'));
%     dir(fullfile(Folder, '*.vrt'));
%     ];
%
% Tifs = Tifs(~[Tifs.isdir]);
% LCC_vrt = '/tank/data/SFS/xinyis/data/LCC/LCC.vrt';
%
% Output_folder = fullfile([Folder,'/LCC/']);
% if exist(Output_folder,'dir') ~= 7
%     mkdir(Output_folder);
% end
%
% for i = 1 : length(Tifs)
%     if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif') ||  contains(Tifs(i).name,'_DEM_1m_Proj')
%         continue
%     end
%
%     bathy_vrt = fullfile([Folder,Tifs(i).name]);
%     [~, baseName, ~] = fileparts(bathy_vrt);   % baseName 就是不含 .tif 的名字
%
%     dstFile = fullfile([Output_folder,baseName,'_ESA_WorldCover_Resampleandclip.vrt']);
%
%     [~,rows,cols,geoTrans,proj,~,~]=RasterInfo(bathy_vrt);
%
%     ResampleAndClip(geoTrans, proj, cols,...
%         rows, LCC_vrt, dstFile, 'GTiff', 1, 1); % use 1
%
%     clear bathy_vrt dstFile baseName
%     clear geoTrans proj cols rows
%     disp([num2str(i),' LCC vrt Resampleandclip is done'])
% end
%
%
% % 全部完成


%% 提取并转换,仅保留80这个河道的
% 分块读写失败，暂时不需要了，只要在统计的时候强行取80进行计算就可以

% clear;clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
%
% LCC_vrt = '/tank/data/SFS/xinyis/data/LCC/LCC.vrt';
% Output_folder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_1m/';
% if exist(Output_folder,'dir') ~= 7
%     mkdir(Output_folder);
% end
%
% Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
% d = dir(Folder);                 % 列出该目录下的所有条目
% d = d([d.isdir]);                % 只保留文件夹
% d(1:2) = [];
%
% for i = 2 : length(d)
%     if contains(d(i).name,'MD_PotomacRiver') || contains(d(i).name,'NoNeed') || contains(d(i).name,'milwaukee_river_3DEP')
%         continue
%     end
%
%     LCC_Folder = fullfile([Folder,d(i).name,'/LCC']);
%     LCC_File = fullfile([LCC_Folder,'/ESA_WorldCover_Resampleandclip.tif']);
%
%     [nbands,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(LCC_File);
%
%     % LCC = ReadRaster(LCC_File, 1, 1, rows, cols); % 过大不能一次读入
%
%     % 分块读取并写出
%     % 设定
%     tile = 2048;                                   % 可调
%     dataType = 1;                                  % GDT_Byte = 1（分类数据）
%     ndv = double(nodataval);                       % NoData 值（保持 double）
%     dstFolder = fullfile(Output_folder, d(i).name);
%     if exist(dstFolder,'dir') ~= 7, mkdir(dstFolder); end
%     dstFile = fullfile(dstFolder, 'ESA_WorldCover_1m_River_Proj.tif');
%
%     % 全幅尺寸、仿射与投影（已从 RasterInfo 取到：rows, cols, geoTrans, proj）
%     totalTiles = ceil(rows/tile) * ceil(cols/tile);
%     tileCount = 0;
%
%     for r = 1:tile:rows
%         rr = min(tile, rows - r + 1);
%         for c = 1:tile:cols
%             cc = min(tile, cols - c + 1);
%
%             % 分块读取（你的 ReadRaster 是 1-based: xOff=c, yOff=r, xSize=cc, ySize=rr）
%             block = ReadRaster(LCC_File, r, c, rr, cc);
%
%             % 仅保留80（河道），其余置为 NaN（WriteRaster 会把 NaN 换成 NoData）
%             out = nan(rr, cc);
%             mask = (block == 80);
%             out(mask) = 80;
%
%             % 分块写入（11 参调用；第一次会按 rows/cols 创建整幅，再写当前块）  ! 这里应该用subgeoTrans
%             WriteRaster(dstFile, out, geoTrans, proj, dataType, 'GTiff', ndv, ...
%                 r, c, rows, cols);
%
%             % 极简进度
%             tileCount = tileCount + 1;
%             fprintf('\rProgress: %6.2f%%  (%d/%d)', ...
%                 100*tileCount/totalTiles, tileCount, totalTiles);
%             clear mask block
%         end
%     end
%     fprintf('\nDone. %s\n', dstFile);
%
%
%     % mask = false(rows, cols);
%     % for r = 1:tile:rows
%     %     rr = min(tile, rows - r + 1);
%     %     for c = 1:tile:cols
%     %         cc = min(tile, cols - c + 1);
%     %         mask(r:r+rr-1, c:c+cc-1) = true;
%     %     end
%     % end
%     % assert(all(mask(:)), '有像元未被覆盖到！');
%
%     clear LCC_Folder dstFile LCC_File
%     clear geoTrans proj cols rows dataType nodataval
%     clear dstFolder dstFile
%     disp([num2str(i),' LCC River Extract is done'])
% end
%
%
