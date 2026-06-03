%% 2025年10月28日
% 经过前面的处理流程，完成了所有的河流bathmatry的处理
% 包括：1. 合并vrt; 2. 融合bathmatry和3DEP
% 增加一个升尺度的操作，从1m升到3m，5m，10m为了适应宽河段
% 提取合适的中心点-超过80%的有效数据推导河道内
% 根据会议讨论，后续会使用新的算法从LCC得到中心线和河宽

% 首先构建vrt，然后全面检查nodata，摸清楚情况后，统一nodata

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

SrcRoot = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
DstRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m/';

if exist(DstRoot,'dir') ~= 7; mkdir(DstRoot); end

d = dir(SrcRoot);
d = d([d.isdir]); d(1:2) = [];

for i = 1:numel(d)
    river = d(i).name;
    if contains(river,'NoNeed') || contains(river,'milwaukee_river_3DEP')
        continue;
    end

    srcFolder = fullfile(SrcRoot, river);
    listFile  = fullfile(srcFolder, 'Filelist.txt');
    if ~exist(listFile,'file')
        fprintf('[WARN] %s: Filelist.txt not found, skip.\n', river);
        continue;
    end

    dstFolder = fullfile(DstRoot, river);
    if exist(dstFolder,'dir') ~= 7; mkdir(dstFolder); end
    dstVRT = fullfile(dstFolder, 'Bathy_1m.vrt');

    cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
                   'export PROJ_NETWORK=ON; ' ...
                   'gdalbuildvrt -overwrite ' ...
                   '-input_file_list "%s" "%s"'], ...
                   listFile, dstVRT);

    status = system(cmd);
    if status ~= 0
        error('gdalbuildvrt failed for %s:\n%s', river, cmd);
    end

    [~,~,~,~,~,~,nd_vrt] = RasterInfo(dstVRT);
    fprintf('[%s] VRT built (merge only). VRT nodata(meta) = %g\n', river, nd_vrt);
end

% milwaukee 部分的
% Milwaukee river 的三个独立 bathymetry：只建 VRT，方便后续统一处理

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

rawFolder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
dstRoot   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m/';

if exist(dstRoot,'dir') ~= 7
    mkdir(dstRoot);
end

bathyNames = {'BadgerFinNull','Estabrook_Combined','KewaFix2Null'};

for k = 1:numel(bathyNames)
    srcTif    = fullfile(rawFolder, [bathyNames{k}, '.tif']);
    outFolder = fullfile(dstRoot, bathyNames{k});
    if exist(outFolder,'dir') ~= 7
        mkdir(outFolder);
    end
    outVrt = fullfile(outFolder, 'Bathy_1m.vrt');

    lst = tempname;
    fid = fopen(lst, 'w');
    fprintf(fid, '%s\n', srcTif);
    fclose(fid);

    cmd = sprintf('gdalbuildvrt -overwrite -input_file_list "%s" "%s"', lst, outVrt);
    assert(system(cmd) == 0, 'gdalbuildvrt failed for %s', bathyNames{k});
    % delete(lst);

    fprintf('[Milwaukee] %s -> %s\n', srcTif, outVrt);
    [~,~,~,~,~,~,nd_vrt] = RasterInfo(outVrt);
    fprintf('[%s] VRT built (merge only). VRT nodata(meta) = %g\n', bathyNames{k}, nd_vrt);

end


% 将其中的Kletzch.tif和UpMax3Null.tiff合并出一个vrt
% 以UpMax3Null.tiff为主
clear;clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Kletzch_proj = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/Kletzch_proj.tif';
UpMax3Null   = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/UpMax3Null.tif';
outdir = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m/Kletzch_Combined_UpMax3Null/';
if exist(outdir,'dir') ~= 7; mkdir(outdir); end
out_vrt      = [outdir,'/Bathy_1m.vrt'];

cd('/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/')
lst = tempname; 
fid = fopen(lst,'w');
fprintf(fid, '%s\n', Kletzch_proj);   % 先 Kletzch_proj
fprintf(fid, '%s\n', UpMax3Null);     % 后 UpMax3Null（优先）
fclose(fid);

% 只 merge：不设置 -srcnodata/-vrtnodata，不做 -r，不做 -resolution
cmd = sprintf('gdalbuildvrt -overwrite -input_file_list "%s" "%s"', lst, out_vrt);

assert(system(cmd)==0, 'gdalbuildvrt failed');
% delete(lst);

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
[~,~,~,~,~,~,nd_vrt] = RasterInfo(out_vrt);
fprintf('VRT built (merge only). VRT nodata(meta) = %g\n', nd_vrt);


%% 经过检查，只有milwaukee河流以及CA_KlamathRiver_TopoBathy_2018_D18的和别的河流的不相同
% 经过检查，所有的milwaukee的，和CA的nodata与其他人不一样，统一到-999999
% 用A000_bathymetry_VRT_mergeOnly/fix_nd.sh在像元上修改，然后重新生成vrt
% 其余的可以直接生成vrt

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

SrcRoot = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
DstRoot = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/';

if exist(DstRoot,'dir') ~= 7; mkdir(DstRoot); end

d = dir(SrcRoot);
d = d([d.isdir]); d(1:2) = [];

for i = 1:numel(d)
    river = d(i).name;
    if contains(river,'NoNeed') || contains(river,'milwaukee_river_3DEP') || contains(river,'CA_KlamathRiver')
        continue;
    end

    srcFolder = fullfile(SrcRoot, river);
    listFile  = fullfile(srcFolder, 'Filelist.txt');
    if ~exist(listFile,'file')
        fprintf('[WARN] %s: Filelist.txt not found, skip.\n', river);
        continue;
    end

    dstFolder = fullfile(DstRoot, river);
    if exist(dstFolder,'dir') ~= 7; mkdir(dstFolder); end
    dstVRT = fullfile(dstFolder, 'Bathy_1m.vrt');

    cmd = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
                   'export PROJ_NETWORK=ON; ' ...
                   'gdalbuildvrt -overwrite ' ...
                   '-input_file_list "%s" "%s"'], ...
                   listFile, dstVRT);

    status = system(cmd);
    if status ~= 0
        error('gdalbuildvrt failed for %s:\n%s', river, cmd);
    end

    [~,~,~,~,~,~,nd_vrt] = RasterInfo(dstVRT);
    fprintf('[%s] VRT built (merge only). VRT nodata(meta) = %g\n', river, nd_vrt);
end


%% 生成vrt的范围用于下载3DEP
% 统一输出位置了

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/A001_bathymetry_VRT/';
d = dir(Folder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];

for i = 1 : length(d)

    Tiff_folder = [Folder,d(i).name];
    vrtfile   = fullfile(Tiff_folder, 'Bathy.vrt');   % 确保是绝对路径 & Unix 换行

    [nbands,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(vrtfile);

    % 找到坐标的范围
    [Lat0,Lon0]=RowCol2Proj(geoTrans,1,1);
    [Lat1,Lon1]=RowCol2Proj(geoTrans,rows,cols);

    XMIN = min([Lon0 Lon1]);
    XMAX = max([Lon0 Lon1]);
    YMIN = min([Lat0 Lat1]);
    YMAX = max([Lat0 Lat1]);


    % 写四角点（NW, NE, SE, SW），一行一个“X Y”
    fn = fullfile(Tiff_folder,'utm_corners.txt');
    fid = fopen(fn,'w');
    fprintf(fid,'%.3f %.3f\n', XMIN, YMAX);  % NW
    fprintf(fid,'%.3f %.3f\n', XMAX, YMAX);  % NE
    fprintf(fid,'%.3f %.3f\n', XMAX, YMIN);  % SE
    fprintf(fid,'%.3f %.3f\n', XMIN, YMIN);  % SW
    fclose(fid);

    fprintf('写出角点：%s\n', fn);
end

% 下面这部分就不用了
% %% milwaukee river 对应的tiff的3DEP的下载
% 
% % 读取范围
% clear; clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
% 
% Folder = '/tank/data/SFS/xinyis/data/bathymetry/milwaukee_river_3DEP/';
% 
% Tifs = [
%     dir(fullfile(Folder, '**', '*.tif'));
%     dir(fullfile(Folder, '**', '*.vrt'));
%     ];
% 
% Tifs = Tifs(~[Tifs.isdir]);
% 
% Output_folder = '/tank/data/SFS/xinyis/data/bathymetry/milwaukee_river_3DEP/LonLat_range/';
% if exist(Output_folder,'dir') ~= 7
%     mkdir(Output_folder);
% end
% 
% for i = 1 : length(Tifs)
% 
%     Tiff_path = fullfile(Tifs(i).folder, Tifs(i).name);
%     [~, baseName, ~] = fileparts(Tiff_path);   % baseName 就是不含 .tif 的名字
% 
%     [nbands,rows,cols,geoTrans,proj,dataType,nodataval]=RasterInfo(Tiff_path);
% 
%     % 找到坐标的范围
%     [Lat0,Lon0]=RowCol2Proj(geoTrans,1,1);
%     [Lat1,Lon1]=RowCol2Proj(geoTrans,rows,cols);
% 
%     XMIN_1 = min([Lon0 Lon1]);
%     XMAX_1 = max([Lon0 Lon1]);
%     YMIN_1 = min([Lat0 Lat1]);
%     YMAX_1 = max([Lat0 Lat1]);
% 
%     if i == 1
%         XMIN = XMIN_1;
%         XMAX = XMAX_1;
%         YMIN = YMIN_1;
%         YMAX = YMAX_1;
%     else
%         XMIN = min([XMIN_1 XMIN]);
%         XMAX = max([XMAX_1 XMAX]);
%         YMIN = min([YMIN_1 YMIN]);
%         YMAX = max([YMAX_1 YMAX]);
%     end
% 
% end
% 
% % 写四角点（NW, NE, SE, SW），一行一个“X Y”
% fn = fullfile(Output_folder,'utm_corners.txt');
% fid = fopen(fn,'w');
% fprintf(fid,'%.3f %.3f\n', XMIN, YMAX);  % NW
% fprintf(fid,'%.3f %.3f\n', XMAX, YMAX);  % NE
% fprintf(fid,'%.3f %.3f\n', XMAX, YMIN);  % SE
% fprintf(fid,'%.3f %.3f\n', XMIN, YMIN);  % SW
% fclose(fid);
% 
% fprintf('写出角点：%s\n', fn);





%% 将3DEP的数据整理成vrt，并统一 nodata（不再追踪原始 nodata）
% milwaukee部分的是整合成一个的milwaukee_river_3DEP/DEM_3DEP_1m.vrt

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
d = dir(Folder);
d = d([d.isdir]);
d(1:2) = [];

DstFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/3DEP_1m_VRT/';
if exist(DstFolder,'dir') ~= 7
    mkdir(DstFolder);
end

globalND = -999999;   % 统一的 NoData

for i = 1 : length(d)
    if contains(d(i).name,'NoNeed')
        continue
    end

    Tiff_folder = fullfile(Folder, d(i).name, 'DEM_1m_raw');
    List_text   = fullfile(Tiff_folder, 'Filelist.txt');

    dstfolder = fullfile(DstFolder, d(i).name);
    if exist(dstfolder,'dir') ~= 7
        mkdir(dstfolder);
    end

    vrt_raw   = fullfile(dstfolder, 'DEM_3DEP_1m.vrt');          % 原始拼接 VRT
    vrt_fixnd = fullfile(dstfolder, 'DEM_3DEP_1m_FixND.vrt');    % 统一 nodata 的 Warp VRT

    % ---- 1) build raw VRT (不乱设 srcnodata) ----
    cmd1 = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
        'export PROJ_NETWORK=ON; ' ...
        'gdalbuildvrt -overwrite ' ...
        '-input_file_list "%s" "%s"'], ...
        List_text, vrt_raw);

    cd(Tiff_folder);
    status = system(cmd1);
    if status ~= 0
        error('gdalbuildvrt failed:\n%s', cmd1);
    end

    % ---- 2) warp to a VRT that unifies dst nodata ----
    % 关键：只用 -dstnodata，让 GDAL 根据源数据自带 nodata/alpha 识别无效区
    cmd2 = sprintf(['export PROJ_USE_PROJ4_INIT_RULES=YES; ' ...
        'export PROJ_NETWORK=ON; ' ...
        'gdalwarp -overwrite -of VRT -multi -wo NUM_THREADS=ALL_CPUS ' ...
        '-dstnodata %g "%s" "%s"'], ...
        globalND, vrt_raw, vrt_fixnd);

    status2 = system(cmd2);
    if status2 ~= 0
        error('gdalwarp(VRT) failed:\n%s', cmd2);
    end

    fprintf('\n[%s] DEM_3DEP_1m_FixND.vrt nodata:\n', d(i).name);
    system(sprintf('gdalinfo "%s" | grep -i "NoData Value"', vrt_fixnd));

    disp([num2str(i),' 3DEP vrt build + FixND is done'])
end


%% 进行resample和合并的操作
% 第一步是将3DEP的先重采样到同bathmetry相同的网格

% % 融合Bathmatry和3DEP
% % 以前者为准
% 
% % 这一步先将3DEP的栅格重采样到bathmatry的栅格上面
% 
% % 首先进行投影和重采样
% 
% clear; clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
% 
% Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
% d = dir(Folder);                 % 列出该目录下的所有条目
% d = d([d.isdir]);                % 只保留文件夹
% d(1:2) = [];
% 
% for i = 3 %:length(d)
%     if contains(d(i).name,'MD_PotomacRiver') || contains(d(i).name,'NoNeed')
%         continue
%     end
% 
%     bathy_vrt = fullfile([Folder,d(i).name,'/Bathy.vrt']);
%     dep3_vrt = fullfile([Folder,d(i).name,'/DEM_1m_raw/DEM_3DEP_1m.vrt']);
%     dstFile = fullfile([Folder,d(i).name,'/DEM_1m_raw/DEM_1m_Proj_3.vrt']);
% 
%     [~,rows,cols,geoTrans,proj,~,~]=RasterInfo(bathy_vrt);
% 
%     ResampleAndClip(geoTrans, proj, cols,...
%         rows, dep3_vrt, dstFile, 'VRT', 1, 2); % use 2 Bilinear
% 
%     clear bathy_vrt dep3_vrt dstFile
%     clear geoTrans proj cols rows
%     disp([num2str(i),' 3DEP vrt Resampleandclip is done'])
% end

% 对于3号而言resampleandclip的速度非常慢，所以尝试用分块的方法提速
% 但是要验证和之前的结果是否相同
% 验证通过，全面采用

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

% ---- 1) 路径与目录 ----
Folder    = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/';
DEMFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/3DEP_1m_VRT/';

d  = dir(Folder);                 
d  = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];                      % 去掉 . 和 ..

globalND = -999999;   % 统一的 NoData

% Milwaukee 底下四个子块，对应同一个 DEM_3DEP_1m.vrt
milwaukee_children = { ...
    'BadgerFinNull', ...
    'Estabrook_Combined', ...
    'KewaFix2Null', ...
    'Kletzch_Combined_UpMax3Null'};

DstFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/3DEP_1m_ResampleClip/';
if exist(DstFolder,'dir') ~= 7
    mkdir(DstFolder);
end

for i = 1 : length(d)
    bname = d(i).name;   % 当前 bathymetry 子目录名

    % ---- 找到对应的 DEM VRT 名（Milwaukee 特例）----
    if any(strcmp(bname, milwaukee_children))
        demName = 'milwaukee_river_3DEP';
    else
        demName = bname;
    end

    bathy_vrt = fullfile(Folder, bname, 'Bathy_1m.vrt');
    dep3_vrt  = fullfile(DEMFolder, demName, 'DEM_3DEP_1m_FixND.vrt');

    if exist(bathy_vrt,'file') ~= 2
        warning('跳过：找不到 Bathy.vrt: %s', bathy_vrt);
        continue;
    end
    if exist(dep3_vrt,'file') ~= 2
        error('找不到 DEM_3DEP_1m.vrt: %s', dep3_vrt);
    end

    % 输出目录：存在则复用，不存在则创建
    out_subdir = fullfile(DstFolder, bname);
    if exist(out_subdir,'dir') ~= 7
        mkdir(out_subdir);
    end
    dstFile = fullfile(out_subdir, 'DEM_3DEP_1m_ResampleandClip.vrt');

    % ---- 读取 bathy 的网格信息，作为目标网格 ----
    [~, rows, cols, geoTrans, proj, ~, ~] = RasterInfo(bathy_vrt);

    xmin = geoTrans(1);
    xres = geoTrans(2);      % 正
    ymax = geoTrans(4);
    yres = geoTrans(6);      % 负
    xmax = xmin + cols * xres;
    ymin = ymax + rows * yres;

    % 注意 WKT 里可能有单引号，这里用两个单引号转义
    proj_arg = sprintf('''%s''', proj);

    % 分辨率与 bathy 一致
    tr_arg = sprintf('-tr %.10f %.10f', xres, abs(yres));

    % 统一的 nodata
    srcnodata = globalND;
    dstnodata = globalND;

    % ---- gdalwarp: 把 DEM_3DEP_1m.vrt 重采样到 bathy 网格 ----
    % 用 -te + -ts cols rows，强制 DEM 输出网格尺寸与 bathy 完全一致
    cmd = sprintf([ ...
        'gdalwarp -of VRT -r near -multi -wo NUM_THREADS=ALL_CPUS -wm 2048 ' ...
        '-t_srs %s -te_srs %s ' ...
        '-te %.10f %.10f %.10f %.10f ' ...
        '-ts %d %d ' ...
        '-srcnodata %g -dstnodata %g -wo INIT_DEST=NO_DATA -wo SKIP_NOSOURCE=YES -overwrite ' ...
        '"%s" "%s"' ], ...
        proj_arg, proj_arg, ...
        xmin, ymin, xmax, ymax, ...
        cols, rows, ...             % 注意顺序：cols rows
        srcnodata, dstnodata, dep3_vrt, dstFile);

    status = system(cmd);
    if status ~= 0
        error('gdalwarp 生成 VRT 失败（检查 WKT 引号、-tr/-tap、nodata）。\nCMD = %s', cmd);
    end

    fprintf('[%d/%d] %s <- %s  : DEM_1m_ResampleandClip.vrt done.\n', ...
            i, length(d), bname, demName);
end

% 测试是否重采样好了
[~, rB, cB, gB, pB] = RasterInfo(bathy_vrt);
[~, rD, cD, gD, pD] = RasterInfo(bathy_vrt);

disp('rows/cols bathy vs dem:'); disp([rB cB; rD cD]);
disp('geoTrans bathy vs dem:'); disp([gB(:) gD(:)]);
fprintf('proj same? %d\n', strcmp(pB,pD));


% % MIlwaukee River
% clear; clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
% 
% Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
% OutFolder = [Folder,'/3DEP_Resampleandclip/'];
% if exist(OutFolder,'dir') ~= 7
%     mkdir(OutFolder);
% end
% 
% Tifs = [
%     dir(fullfile(Folder, '*.tif'));
%     dir(fullfile(Folder, '*.vrt'));
%     ];
% 
% Tifs = Tifs(~[Tifs.isdir]);
% dep3_vrt = fullfile([Folder,'/DEM_1m_raw/DEM_3DEP_1m.vrt']);
% 
% globalND = -3.4028235e+38;   % 和 Bathy 那边保持一致
% 
% for i = 1 : length(Tifs)
%     if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif')
%         continue
%     end
% 
%     bathy_vrt = fullfile([Folder,Tifs(i).name]);
%     [~, baseName, ~] = fileparts(bathy_vrt);   % baseName 就是不含 .tif 的名字
%     dstfolder = fullfile([OutFolder,baseName]);
%     if exist(dstfolder,'dir') ~= 7
%         mkdir(dstfolder);
%     end
% 
%     dstFile = fullfile([dstfolder,'/DEM_1m_ResampleandClip.vrt']);
% 
%     [~,rows,cols,geoTrans,proj,~,~]=RasterInfo(bathy_vrt);
% 
%     % 已有：geoTrans, proj(WKT), cols, rows, dep3_vrt, dstFile
%     xmin = geoTrans(1);
%     xres = geoTrans(2);      % 正
%     ymax = geoTrans(4);
%     yres = geoTrans(6);      % 负
%     xmax = xmin + cols * xres;
%     ymin = ymax + rows * yres;
% 
%     % 用单引号包住 WKT（注意 MATLAB 里单引号要写成两个：''）
%     proj_arg = sprintf('''%s''', proj);
% 
%     % 像元分辨率（与 bathy 一致），-tap 需要 -tr
%     tr_arg = sprintf('-tr %.10f %.10f', xres, abs(yres));
% 
%     % 如果你知道真实 nodata 就填；不确定可以先删这两个参数
%     srcnodata = globalND; dstnodata = globalND;
% 
%     cmd = sprintf([ ...
%         'gdalwarp -of VRT -r bilinear -multi -wo NUM_THREADS=ALL_CPUS -wm 2048 ' ...
%         '-t_srs %s -te_srs %s %s -te %.10f %.10f %.10f %.10f -tap ' ...
%         '-srcnodata %g -dstnodata %g -wo INIT_DEST=NO_DATA -wo SKIP_NOSOURCE=YES -overwrite ' ...
%         '"%s" "%s"' ], ...
%         proj_arg, proj_arg, tr_arg, ...
%         xmin, ymin, xmax, ymax, ...
%         srcnodata, dstnodata, dep3_vrt, dstFile);
% 
%     status = system(cmd);
%     if status ~= 0
%         error('gdalwarp 生成 VRT 失败（检查 WKT 引号、-tr/-tap、nodata）。');
%     end
% 
%     disp([num2str(i),' 3DEP vrt Resampleandclip is done'])
% end
% 

% % 进行验证
% clear;clc;
% 
% ResampleResults = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/OR_SantiamRiverTB_Topobathy_1_D23/DEM_1m_raw/DEM_1m_Proj.vrt';
% gdalwarpResults = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/OR_SantiamRiverTB_Topobathy_1_D23/DEM_1m_raw/DEM_1m_ResampleandClip.vrt';
% 
% % ==== 读取基础信息 ====
% [nb1, rows1, cols1, g1, p1, dt1, nd1] = RasterInfo(ResampleResults);
% [nb2, rows2, cols2, g2, p2, dt2, nd2] = RasterInfo(gdalwarpResults);
% 
% % 基本一致性快速检查
% assert(rows1==rows2 && cols1==cols2, '行列不一致：rows/cols');
% if ~strcmp(p1, p2)
%     warning('投影 WKT 不一致（可能仍然配准一致，但建议检查）');
% end
% if any(abs([g1(1)-g2(1), g1(2)-g2(2), g1(4)-g2(4), g1(6)-g2(6)]) > 1e-9)
%     warning('GeoTransform 不完全一致（像元对齐可能有差异）');
% end
% 
% tileW = 1000;           % tile 的边长（像元）
% Nsamples = 10;          % 抽样次数
% tol = 1e-6;             % 数值容差（双线性/线程差异可适当放大，如1e-4）
% rng('shuffle');
% 
% % 防止 tile 超出边界
% maxRow0 = rows1 - tileW;
% maxCol0 = cols1 - tileW;
% assert(maxRow0 > 0 && maxCol0 > 0, 'tileW 超出影像尺寸');
% 
% for k = 1:Nsamples
%     % 随机左上角（行列索引，1-based）
%     row0 = randi([1, maxRow0]);
%     col0 = randi([1, maxCol0]);
% 
%     % 读取各自 tile（注意 ReadRaster 的参数顺序：列、行、宽、高）
%     t1 = ReadRaster(ResampleResults,row0, col0, tileW, tileW);
%     t2 = ReadRaster(gdalwarpResults, row0, col0, tileW, tileW);
% 
%     % 转 double 计算
%     t1 = double(t1);
%     t2 = double(t2);
%     t1_line = reshape(t1,[],1);
%     t2_line = reshape(t2,[],1);
% 
%     vaild = find(~isnan(t1_line));
%     noSame = find(t1_line(vaild) ~= t2_line(vaild));
% 
%     if ~isempty(noSame)
%         disp(['Total have ',num2str(length(noSame)),' grids are different!'])
% 
%         if length(noSame) > 10
%             for j = 1 : 10
%                 disp(['ResampleResults: ',num2str(t1_line(noSame(j)))])
%                 disp(['gdalwarpResults: ',num2str(t2_line(noSame(j)))])
%             end
%         else
%             for j = 1 : length(noSame)
%                 disp(['ResampleResults: ',num2str(t1_line(noSame(j)))])
%                 disp(['gdalwarpResults: ',num2str(t2_line(noSame(j)))])
%             end
%         end
% 
%     else
%         disp('No different!')
%     end
% 
% end




%% 将3DEP同bathyj进行融合
% 对任意一个像元 (x, y)：

% 如果 bathy(x,y) 有值（≠ globalND）
% → Combined_Bathy_Priority(x,y) = bathy(x,y)；

% 如果 bathy(x,y) 是 nodata（= globalND），但 3DEP(x,y) 有值
% → Combined_Bathy_Priority(x,y) = 3DEP(x,y)；

% 如果两者都是 nodata
% → Combined_Bathy_Priority(x,y) = globalND。

% 必须使用像元计算，然后保存为tiff，最后再输出成vrt（可选）

%% ====== 合并 bathy + 3DEP：bathy 优先，用 3DEP 补洞，输出 TIF ======
% clear; clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
% 
% % ------ 1) 先对单条河测试 ------
% name = 'Kletzch_Combined_UpMax3Null';   % 改成你要合并的子目录名
% 
% Folder_bathy = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/';
% Folder_dem   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/3DEP_1m_ResampleClip/';
% OutFolder    = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_1m/';
% 
% if exist(OutFolder,'dir') ~= 7
%     mkdir(OutFolder);
% end
% 
% bathy_vrt = fullfile(Folder_bathy, name, 'Bathy_1m.vrt');
% dem_vrt   = fullfile(Folder_dem,   name, 'DEM_3DEP_1m_ResampleandClip.vrt');
% outTif    = fullfile(OutFolder, [name '_Merged_1m.tif']);
% 
% assert(exist(bathy_vrt,'file')==2, 'Bathy_1m.vrt not found');
% assert(exist(dem_vrt,'file')  ==2, 'DEM_3DEP_1m_ResampleandClip.vrt not found');
% 
% % 如果已经有旧的结果，先删掉，防止残留
% if exist(outTif,'file') == 2
%     delete(outTif);
% end
% 
% % ------ 2) 用 bathy 的空间参考作为“母版” ------
% [~, rows, cols, geoTrans, proj, dataType_bathy, nodataval] = RasterInfo(bathy_vrt);
% fprintf('Bathy size: rows=%d, cols=%d\n', rows, cols);
% 
% % 统一的 nodata（你已经在前面所有数据中统一过）
% globalND = -999999;   % 统一的 NoData
% 
% % ------ 3) 分块读写设置 ------
% tile = 2048;    % 可调；内存充足可以 4096 甚至更大
% totalTiles = ceil(rows/tile) * ceil(cols/tile);
% tileCount  = 0;
% 
% fprintf('Start merging %s ...\n', name);
% 
% for rLocal = 1:tile:rows
%     rr = min(tile, rows - rLocal + 1);
%     for cLocal = 1:tile:cols
%         cc = min(tile, cols - cLocal + 1);
% 
%         % ★ 这里没有子窗口，所以绝对行/列就是 rLocal/cLocal
%         absRow = rLocal;
%         absCol = cLocal;
% 
%         % ------ 4) 分块读取 bathy / 3DEP ------
%         B = double(ReadRaster(bathy_vrt, absRow, absCol, rr, cc));
%         D = double(ReadRaster(dem_vrt,   absRow, absCol, rr, cc));
% 
%         % 这里的融合读取还是出现问题，写出很容易就得到一个非常小的填充值
%         % ------ 6) 按规则合并：bathy优先，bathy洞用3DEP补 ------
%         C = B;
% 
%         isHoleB  = isnan(B) | (B == globalND);
%         isValidD = isfinite(D) & ~isnan(D) & (D ~= globalND); % 3DEP有效的地方
% 
%         mask_fill = isHoleB & isValidD;
%         C(mask_fill) = D(mask_fill);
% 
%         C(~isfinite(C) | isnan(C) | (C == globalND)) = globalND;
% 
%         % ------ 7) 分块写入输出 TIF ------
%         % 注意 geoTrans 是整幅图的仿射；rows/cols 也是全图大小
%         WriteRaster(outTif, C, geoTrans, proj, dataType_bathy, ...
%                     'GTiff', globalND, ...
%                     rLocal, cLocal, rows, cols);
%         % WriteRaster(outTif, C, geoTrans, proj, dataType_bathy, ...
%         %             'GTiff', nodataval);
% 
%         % 输出用于验证的
%         subgeoTrans = subTranscoef(geoTrans,absRow,absCol);
%         WriteRaster(outTif1, C, subgeoTrans, proj, dataType_bathy, 'GTiff', nodataval); % 融合后的结果
%         WriteRaster(outTif2, B, subgeoTrans, proj, dataType_bathy, 'GTiff', nodataval); % 融合前的bathy
%         WriteRaster(outTif3, D, subgeoTrans, proj, dataType_bathy, 'GTiff', nodataval); % 融合前的3DEP
% 
%         % ------ 8) 进度信息 ------
%         tileCount = tileCount + 1;
%         fprintf('\rProgress: %6.2f%%  (%d/%d)', ...
%             100*tileCount/totalTiles, tileCount, totalTiles);
% 
%         clear B D C mask_fill
%     end
% end
% fprintf('\nDone. Output = %s\n', outTif);


%% ====== Batch merge bathy + 3DEP (bathy priority, DEM fill holes) ======
clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder_bathy = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_1m_FixND/';
Folder_dem   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/3DEP_1m_ResampleClip/';
OutFolder    = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_1m/';
VerifyRoot   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Z001_Verify_Merge/';

if exist(OutFolder,'dir') ~= 7; mkdir(OutFolder); end
if exist(VerifyRoot,'dir') ~= 7; mkdir(VerifyRoot); end

globalND = -999999;         % unified NoData
tile     = 2048;            % tile size
% ---- (在每条河开始处加这两个参数) ----
nVerifyTiles = 2;        % 每条河输出 1-2 个样本
minFillPix   = 1000;     % 至少补洞像元数阈值（可调：200/1000/5000）
verifySaved  = 0;        % 已输出样本数

% ---- enumerate rivers by bathy folders ----
d = dir(Folder_bathy);
d = d([d.isdir]);
d(1:2) = [];  % remove . ..

fprintf('Found %d rivers under Bathy folder.\n', numel(d));

for iRiver = 1:numel(d)
    name = d(iRiver).name;

    bathy_vrt = fullfile(Folder_bathy, name, 'Bathy_1m.vrt');
    dem_vrt   = fullfile(Folder_dem,   name, 'DEM_3DEP_1m_ResampleandClip.vrt');

    if exist(bathy_vrt,'file') ~= 2
        warning('[%d/%d] Skip %s: missing bathy vrt: %s', iRiver, numel(d), name, bathy_vrt);
        continue;
    end
    if exist(dem_vrt,'file') ~= 2
        warning('[%d/%d] Skip %s: missing dem vrt: %s', iRiver, numel(d), name, dem_vrt);
        continue;
    end

    % ---- output folders ----
    out_subdir = fullfile(OutFolder, name);
    if exist(out_subdir,'dir') ~= 7; mkdir(out_subdir); end
    % outTif = fullfile(out_subdir, sprintf('%s_Merged_1m.tif', name)); % 最后整体出个vrt就行

    tilesDir = fullfile(out_subdir, '_tiles');
    if exist(tilesDir,'dir') ~= 7; mkdir(tilesDir); end

    verifyDir = fullfile(VerifyRoot, name);
    if exist(verifyDir,'dir') ~= 7; mkdir(verifyDir); end

    % ---- master grid from bathy ----
    [~, rows, cols, geoTrans, proj, dataType_bathy, nodataval] = RasterInfo(bathy_vrt);
    fprintf('\n[%d/%d] Start merging: %s  (rows=%d cols=%d)\n', iRiver, numel(d), name, rows, cols);

    % ---- tiling loops ----
    totalTiles = ceil(rows/tile) * ceil(cols/tile);
    tileCount  = 0;

    for rLocal = 1:tile:rows
        rr = min(tile, rows - rLocal + 1);

        for cLocal = 1:tile:cols
            cc = min(tile, cols - cLocal + 1);

            absRow = rLocal;
            absCol = cLocal;

            % ---- read ----
            B = double(ReadRaster(bathy_vrt, absRow, absCol, rr, cc));
            D = double(ReadRaster(dem_vrt,   absRow, absCol, rr, cc));

            % ---- merge ----
            C = B;

            % 你现在这版成功了就保持：NaN 或 globalND 判洞
            isHoleB  = isnan(B) | (B == globalND);
            isValidD = isfinite(D) & ~isnan(D) & (D ~= globalND);

            mask_fill = isHoleB & isValidD;
            C(mask_fill) = D(mask_fill);
            C(~isfinite(C) | isnan(C) | (C == globalND)) = globalND;


            % ---- write main merged (tile write into one big tif) ----
            % WriteRaster(outTif, C, geoTrans, proj, dataType_bathy, ...
            %             'GTiff', globalND, ...
            %             rLocal, cLocal, rows, cols);

            % 写出每一个tile的tif
            subgeoTrans = subTranscoef(geoTrans, absRow, absCol);
            tileTif = fullfile(tilesDir, sprintf('tile_r%06d_c%06d.tif', rLocal, cLocal));
            WriteRaster(tileTif, C, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);

            % ---- write verify tiles: only tiles that actually have "fill" ----
            if verifySaved < nVerifyTiles
                nFill = nnz(mask_fill);

                % 只有当补洞数量足够，才输出样本
                if nFill >= minFillPix

                    tag = sprintf('Tile_r%04d_c%04d_fill%d', rLocal, cLocal, nFill);

                    outTif1 = fullfile(verifyDir, sprintf('%s_%s_Merged.tif', name, tag));
                    outTif2 = fullfile(verifyDir, sprintf('%s_%s_Bathy.tif',  name, tag));
                    outTif3 = fullfile(verifyDir, sprintf('%s_%s_3DEP.tif',   name, tag));

                    % 用 nodataval 或 globalND 二选一：
                    % - 如果 ReadRaster 已经把 nodata 读成 NaN，那么 nodataval 写回去影响不大
                    % - 为了和大图一致，也可以写 globalND
                    WriteRaster(outTif1, C, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);
                    WriteRaster(outTif2, B, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);
                    WriteRaster(outTif3, D, subgeoTrans, proj, dataType_bathy, 'GTiff', globalND);

                    verifySaved = verifySaved + 1;

                    fprintf('\n  [VERIFY %d/%d] nFill=%d (r=%d c=%d)\n    %s\n    %s\n    %s\n', ...
                        verifySaved, nVerifyTiles, nFill, rLocal, cLocal, outTif1, outTif2, outTif3);
                end
            end

            % ---- progress ----
            tileCount = tileCount + 1;
            fprintf('\r  Progress: %6.2f%% (%d/%d)', 100*tileCount/totalTiles, tileCount, totalTiles);

            clear B D C mask_fill
        end
    end

    % 最后输出一个vrt
    outVrt = fullfile(out_subdir, sprintf('%s_Merged_1m.vrt', name));
    listTxt = fullfile(out_subdir, 'tile_list.txt');

    % 1) 生成文件列表（绝对路径，一行一个）
    cmdList = sprintf('find "%s" -maxdepth 1 -type f -name "tile_*.tif" | sort > "%s"', ...
        tilesDir, listTxt);
    status = system(cmdList);
    if status ~= 0
        error('Failed to build tile list. CMD=%s', cmdList);
    end

    % 2) 检查列表是否为空
    info = dir(listTxt);
    if isempty(info) || info.bytes == 0
        error('No tiles found in %s (tile_list.txt empty).', tilesDir);
    end

    % 3) 用 input_file_list build vrt
    cmdV = sprintf('gdalbuildvrt -overwrite -vrtnodata %g -input_file_list "%s" "%s"', ...
        globalND, listTxt, outVrt);

    status = system(cmdV);
    if status ~= 0
        error('gdalbuildvrt failed: %s', cmdV);
    end

    fprintf('[%s] VRT mosaic done: %s\n', name, outVrt);

end

fprintf('\nALL RIVERS DONE.\n');



%%% 原来合并的操作
% % Milwaukee
% 
% clear; clc;
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% addpath('/tank/data/SFS/xinyis/src/CREST_Prep');
% 
% Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
% ResampleFolder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/3DEP_Resampleandclip/';
% OutFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_Merge_3DEP_1m/';
% if exist(OutFolder,'dir') ~= 7
%     mkdir(OutFolder);
% end
% 
% Tifs = [
%     dir(fullfile(Folder, '*.tif'));
%     dir(fullfile(Folder, '*.vrt'));
%     ];
% 
% Tifs = Tifs(~[Tifs.isdir]);
% 
% for i = 1 : length(Tifs)
%     if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif') 
%         continue
%     end
% 
%     bathy_vrt = fullfile([Folder,Tifs(i).name]);
%     [~, baseName, ~] = fileparts(bathy_vrt);   % baseName 就是不含 .tif 的名字
% 
%     dep3_Resample = fullfile([ResampleFolder,baseName,'/DEM_1m_ResampleandClip.vrt']);
% 
%     dstfolder = fullfile([OutFolder,baseName]);
%     if exist(dstfolder,'dir') ~= 7
%         mkdir(dstfolder);
%     end
%     dstFile = fullfile([dstfolder,'/Combined_Bathy_Priority.vrt']);
% 
%     cd(Folder)
%     lst = tempname; fid = fopen(lst,'w');
%     fprintf(fid, '%s\n', dep3_Resample);   % 先 3DEP
%     fprintf(fid, '%s\n', bathy_vrt);  % 后 Bathy（优先）
%     fclose(fid);
% 
%     cmd = sprintf(['gdalbuildvrt -overwrite -resolution highest -r bilinear -hidenodata ' ...
%         '-input_file_list "%s" "%s"'], lst, dstFile);
%     assert(system(cmd)==0, 'gdalbuildvrt failed');
% 
%     disp([num2str(i),' 3DEP merge Bathy is done'])
% end


%% 2025年11月23日
% 将之前处理的结果进行升尺度
% 使用ResampleandClip
% 写出为tiff
% 这里同样对CA和Milwaukee的进行nodata的统一操作，使用
% 用fix_nd.sh在像元上修改，然后重新生成vrt
% 在已经升好尺度的结果上面操作

clear; 
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';

d = dir(Folder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];

% 目标分辨率（单位：米）
targetRes = [3, 5, 10];
for j = 2 : length(targetRes)
    res = targetRes(j);
    outFolder = fullfile(['/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_',num2str(res),'m/']);
    if exist(outFolder, 'dir') ~= 7
        mkdir(outFolder);
    end

    for i = 1 %: length(d)
        if contains(d(i).name,'NoNeed') || contains(d(i).name,'milwaukee_river_3DEP')
            continue
        end

        bathy_vrt = fullfile([Folder,d(i).name,'/Bathy.vrt']);
        [~, rows, cols, geoTransSrc, projSrc, ~, ~] = RasterInfo(bathy_vrt);

        outSubFolder = fullfile([outFolder,d(i).name]);
        if exist(outSubFolder, 'dir') ~= 7
            mkdir(outSubFolder);
        end
        outRaster = fullfile(outSubFolder, sprintf('Bathy_%dm_gdalwarp.tif', res));

        % % 目标的 geotransform：保持左上角不变，只改像元大小
        % geoTransTar       = geoTransSrc;
        % geoTransTar(2)    = geoTransSrc(2) * res;  % 像元宽度（一般为正）
        % geoTransTar(6)    = geoTransSrc(6) * res;  % 像元高度（一般为负）
        % % 其他 [3]、[5] 项保持不变
        % 
        % % 目标行列数（取整，避免越界）
        % tarXSize = floor(cols / res);
        % tarYSize = floor(rows / res);
        % 
        % % bandSrc = 1; resampleAlg = 3 表示 Average
        % ResampleAndClip(geoTransTar, projSrc, tarXSize, tarYSize, ...
        %     bathy_vrt, outRaster, 'GTiff', 1, 3);
        % 
        % fprintf('✅ %dm (Average) upscaling done: %s\n', res, outTif);

        % 用 gdalwarp 重采样：
        % -tr res res      : 目标像元大小
        % -r average       : 用平均值重采样（可以改成 bilinear 看需求）
        % -multi/NUM_THREADS: 多线程加速
        % -overwrite       : 覆盖已有文件
        cmd = sprintf(['gdalwarp -tr %d %d -r average ', ...
            '-multi -wo NUM_THREADS=ALL_CPUS ', ...
            '-overwrite "%s" "%s"'], ...
            res, res, bathy_vrt, outRaster);

        fprintf('Running: %s\n', cmd);
        status = system(cmd);
        if status ~= 0
            error('gdalwarp failed for resolution %d m', res);
        else
            fprintf('✅ %dm resampling done: %s\n', res, outRaster);
        end

    end
end

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

% 目标分辨率（单位：米）
targetRes = [3, 5, 10];
for j = 1 : length(targetRes)
    res = targetRes(j);
    outFolder = fullfile(['/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_',num2str(res),'m/']);
    if exist(outFolder, 'dir') ~= 7
        mkdir(outFolder);
    end

    for i = 1 : length(Tifs)
        if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif')
            continue
        end

        bathy_vrt = fullfile([Folder,Tifs(i).name]);
        [~, baseName, ~] = fileparts(bathy_vrt);   % baseName 就是不含 .tif 的名字
        [~, rows, cols, geoTransSrc, projSrc, ~, ~] = RasterInfo(bathy_vrt);

        outSubFolder = fullfile([outFolder,baseName]);
        if exist(outSubFolder, 'dir') ~= 7
            mkdir(outSubFolder);
        end
        outRaster = fullfile(outSubFolder, sprintf('Bathy_%dm.tif', res));

        % % 目标的 geotransform：保持左上角不变，只改像元大小
        % geoTransTar       = geoTransSrc;
        % geoTransTar(2)    = geoTransSrc(2) * res;  % 像元宽度（一般为正）
        % geoTransTar(6)    = geoTransSrc(6) * res;  % 像元高度（一般为负）
        % % 其他 [3]、[5] 项保持不变
        % 
        % % 目标行列数（取整，避免越界）
        % tarXSize = floor(cols / res);
        % tarYSize = floor(rows / res);
        % 
        % % bandSrc = 1; resampleAlg = 3 表示 Average
        % ResampleAndClip(geoTransTar, projSrc, tarXSize, tarYSize, ...
        %     bathy_vrt, outRaster, 'GTiff', 1, 3);
        % 
        % fprintf('✅ %dm (Average) upscaling done: %s\n', res, outTif);

        % 用 gdalwarp 重采样：
        % -tr res res      : 目标像元大小
        % -r average       : 用平均值重采样（可以改成 bilinear 看需求）
        % -multi/NUM_THREADS: 多线程加速
        % -overwrite       : 覆盖已有文件
        cmd = sprintf(['gdalwarp -tr %d %d -r average ', ...
            '-multi -wo NUM_THREADS=ALL_CPUS ', ...
            '-overwrite "%s" "%s"'], ...
            res, res, bathy_vrt, outRaster);

        fprintf('Running: %s\n', cmd);
        status = system(cmd);
        if status ~= 0
            error('gdalwarp failed for resolution %d m', res);
        else
            fprintf('✅ %dm resampling done: %s\n', res, outRaster);
        end

    end
end

% 补充，对Kletzch_Combined_UpMax3Null从最新的A000_bathymetry_VRT_mergeOnly进行升尺度，原来的vrt构建中没有带上nodata，后续可能会有错误
% Milwaukee rivers upscaling from mergeOnly VRT
clear; clc;

cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

% ===== Input root: already merged 1m VRT =====
inRoot  = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/A000_bathymetry_VRT_mergeOnly';

% ===== Output root: Bathy_3m / Bathy_5m / Bathy_10m =====
outBase = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

% Milwaukee rivers
rivers = { ...
    'BadgerFinNull', ...
    'Estabrook_Combined', ...
    'KewaFix2Null', ...
    'Kletzch_Combined_UpMax3Null' ...
    };

% target resolutions (m)
targetRes = [3, 5, 10];

for j = 1:numel(targetRes)
    res = targetRes(j);

    outFolder = fullfile(outBase, sprintf('Bathy_%dm', res));
    if exist(outFolder, 'dir') ~= 7
        mkdir(outFolder);
    end

    for i = 1:numel(rivers)
        river = rivers{i};

        inVRT = fullfile(inRoot, river, 'Bathy_mergeOnly.vrt');
        if exist(inVRT, 'file') ~= 2
            warning('[SKIP] Missing input VRT: %s', inVRT);
            continue;
        end

        outSubFolder = fullfile(outFolder, river);
        if exist(outSubFolder, 'dir') ~= 7
            mkdir(outSubFolder);
        end

        outTif = fullfile(outSubFolder, sprintf('Bathy_%dm.tif', res));
        % outVrt = fullfile(outSubFolder, sprintf('Bathy_%dm.vrt', res));

        % ---- gdalwarp upscaling ----
        % average: 适合升尺度做面平均；若你想更平滑可改 bilinear
        cmd = sprintf([ ...
            'gdalwarp -overwrite -of GTiff ', ...
            '-tr %d %d -r average ', ...
            '-multi -wo NUM_THREADS=ALL_CPUS ', ...
            '-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES ', ...
            '"%s" "%s"' ], ...
            res, res, inVRT, outTif);

        fprintf('\n[%s] %dm: Running gdalwarp...\n%s\n', river, res, cmd);
        status = system(cmd);
        if status ~= 0
            error('gdalwarp failed: river=%s, res=%dm', river, res);
        end
        fprintf('✅ Done: %s\n', outTif);

        % % ---- build a small VRT wrapper for convenience ----
        % cmd2 = sprintf('gdalbuildvrt -overwrite "%s" "%s"', outVrt, outTif);
        % status2 = system(cmd2);
        % if status2 ~= 0
        %     warning('gdalbuildvrt failed: %s', outVrt);
        % else
        %     fprintf('✅ VRT: %s\n', outVrt);
        % end
    end
end

disp('ALL DONE.');



%% 融合后的bathmatry+3DEP
% 这部分相当于在1m融合后的结果再升尺度
% 需要重新做，因为这个时候的1m融合结果是错误的

clear; 
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_1m/';

d = dir(Folder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];

% 目标分辨率（单位：米）
targetRes = [3, 5, 10];
for j = 1 : length(targetRes)
    res = targetRes(j);
    outFolder = fullfile(['/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_',num2str(res),'m/']);
    if exist(outFolder, 'dir') ~= 7
        mkdir(outFolder);
    end

    for i = 15 %: length(d)
        bathy_vrt = fullfile([Folder,d(i).name,'/Combined_Bathy_Priority_1m.vrt']);

        outSubFolder = fullfile([outFolder,d(i).name]);
        if exist(outSubFolder, 'dir') ~= 7
            mkdir(outSubFolder);
        end
        outRaster = fullfile(outSubFolder, sprintf('Combined_Bathy_Priority_%dm.tif', res));

        % [~, rows, cols, geoTransSrc, projSrc, ~, ~] = RasterInfo(bathy_vrt);
        % % 目标的 geotransform：保持左上角不变，只改像元大小
        % geoTransTar       = geoTransSrc;
        % geoTransTar(2)    = geoTransSrc(2) * res;  % 像元宽度（一般为正）
        % geoTransTar(6)    = geoTransSrc(6) * res;  % 像元高度（一般为负）
        % % 其他 [3]、[5] 项保持不变
        % 
        % % 目标行列数（取整，避免越界）
        % tarXSize = floor(cols / res);
        % tarYSize = floor(rows / res);
        % 
        % % bandSrc = 1; resampleAlg = 3 表示 Average
        % ResampleAndClip(geoTransTar, projSrc, tarXSize, tarYSize, ...
        %     bathy_vrt, outRaster, 'GTiff', 1, 3);
        % 
        % fprintf('✅ %dm (Average) upscaling done: %s\n', res, outTif);

        % 用 gdalwarp 重采样：
        % -tr res res      : 目标像元大小
        % -r average       : 用平均值重采样（可以改成 bilinear 看需求）
        % -multi/NUM_THREADS: 多线程加速
        % -overwrite       : 覆盖已有文件
        cmd = sprintf(['gdalwarp -tr %d %d -r average ', ...
            '-multi -wo NUM_THREADS=ALL_CPUS ', ...
            '-overwrite "%s" "%s"'], ...
            res, res, bathy_vrt, outRaster);

        fprintf('Running: %s\n', cmd);
        status = system(cmd);
        if status ~= 0
            error('gdalwarp failed for resolution %d m', res);
        else
            fprintf('✅ %dm resampling done: %s\n', res, outRaster);
        end

    end
end