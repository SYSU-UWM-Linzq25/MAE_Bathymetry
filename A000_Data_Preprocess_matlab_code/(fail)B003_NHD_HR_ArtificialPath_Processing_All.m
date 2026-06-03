%% 2025年11月4日
% 处理NHD的河道中心线
% 后续可能用别的中心线数据
% 取样点的间隔后续可能根据河宽和分辨率改变


%% 预处理：类型转换-脱离maplineshape类型 + 投影转换并裁剪到相应的河流
% % 第一步转换类型
% clear;clc
% 
% cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
% 
% inShp  = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHD_HR_AritificalPath_origin/NHDPlus_HR_ArtificialPath.shp';
% outDir = fileparts(inShp);
% out2D  = fullfile(outDir, 'NHDPlus_HR_ArtificialPath_2D.shp');
% 
% % 源层名（Shapefile 层名=文件名不含扩展名）
% [~, srcLayer] = fileparts(inShp);
% 
% % 只保留这7列 + 2D LINESTRING；其余一律丢弃
% % 说明：
% %  - 用 OGRSQL（默认方言，不需要 SQLite）做列选择
% %  - -dim XY / -nlt LINESTRING 把 PolyLineZ 降成 2D
% %  - -mapFieldType 把 64位整型/日期写成字符串，避免 DBF 限制
% %  - -lco RESIZE=YES 自动扩宽 DBF 字段宽度
% cmd = sprintf(['ogr2ogr -f "ESRI Shapefile" "%s" "%s" ' ...
%                '-nln NHDPlus_HR_ArtificialPath_2D ' ...
%                '-nlt LINESTRING -dim XY ' ...
%                '-sql "SELECT OBJECTID FROM %s" ' ...
%                '-lco RESIZE=YES ' ...
%                '-mapFieldType Integer64=String,DateTime=String ' ...
%                '-overwrite -skipfailures'], ...
%                out2D, inShp, srcLayer);
% 
% [st,msg] = system(cmd); disp(msg);
% assert(st==0, 'ogr2ogr 2D/精简列 失败');
% 
% % （可选）看下摘要确认几何&字段
% [~,chk] = system(sprintf('ogrinfo -so -al "%s"', out2D)); disp(chk);

% HR的数据非常大，而且转换不顺利，在HPC上使用少量属性表提取


%% 第二步投影转换和bathy一样，并裁剪到对应的范围
% 全美的结果，非常大，得先做裁剪，再都进来shp

clear;clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

OutFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHDPlus_HR_ArtificialPath_Processed/';
if exist(OutFolder,'dir') ~= 7
    mkdir(OutFolder);
end

% 全美 ArtificialPath（2D，NAD83，经纬度）
in2D  = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHDPlus_HR_AritificalPath_origin/NHDPlus_HR_ArtificialPath_core.shp';

% ❌ 这一行完全没必要，极慢，还会占很多内存
% GT = shaperead(in2D);

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
d = dir(Folder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];                     % 去掉 '.' 和 '..'

for i = 1 : length(d)
    if contains(d(i).name,'NoNeed') || contains(d(i).name,'milwaukee_river_3DEP')
        continue
    end

    bathy_vrt = fullfile([Folder,d(i).name,'/Bathy.vrt']);
    dstfolder = fullfile([OutFolder,d(i).name]);
    if exist(dstfolder,'dir') ~= 7
        mkdir(dstfolder);
    end
    
    % 读取 bathy 的坐标系 EPSG
    [st, epsgOut] = system(sprintf('gdalsrsinfo -o epsg "%s"', bathy_vrt));
    assert(st==0, 'gdalsrsinfo 失败：%s', epsgOut);
    tok = regexp(epsgOut,'EPSG:\s*(\d+)','tokens','once');
    assert(~isempty(tok), '未解析到 EPSG：%s', epsgOut);
    tgtEPSG = sprintf('EPSG:%s', tok{1});
 
    % === (2) 计算 bathy 范围 ===
    [~,rows,cols,geoTrans,~,~,~]=RasterInfo(bathy_vrt);
    xMin = geoTrans(1);
    xMax = geoTrans(1) + cols*geoTrans(2);
    yMax = geoTrans(4);
    yMin = geoTrans(4) + rows*geoTrans(6);   % 注意 GT(6) 多为负数

    % === (3) 输出路径 ===
    shpProj   = fullfile(dstfolder, 'NHDPlus_ArtificialPath_proj.shp');
    shpClip   = fullfile(dstfolder, 'NHDPlus_ArtificialPath_Resampleandclip.shp');

    % === (4) 先重投影到与 bathy 相同坐标系 ===
    cmdProj = sprintf('ogr2ogr -f "ESRI Shapefile" -t_srs %s "%s" "%s"', ...
                      tgtEPSG, shpProj, in2D);
    [st,msg] = system(cmdProj);
    assert(st==0, 'ogr2ogr 重投影失败：\n%s', msg);

    % === (5) 再按 bathy 范围裁剪（spatial filter） ===
    % 使用 -spat (minX minY maxX maxY)；此时 shpProj 与 bathy 已同一坐标系
    cmdClip = sprintf('ogr2ogr -f "ESRI Shapefile" -spat %.3f %.3f %.3f %.3f "%s" "%s"', ...
                      xMin, yMin, xMax, yMax, shpClip, shpProj);
    [st,msg] = system(cmdClip);
    assert(st==0, 'ogr2ogr 裁剪失败：\n%s', msg);

    fprintf('[%d/%d] %s -> 完成（EPSG=%s，范围=[%.1f %.1f %.1f %.1f]）\n', ...
        i, numel(d), d(i).name, tgtEPSG, xMin, yMin, xMax, yMax);

    clear bathy_vrt dstfolder
    clear epsgOut tok
    clear xMin xMax yMax yMin
    clear geoTrans cols rows
    clear shpProj shpClip
    clear cmdProj cmdClip
    disp([num2str(i),' NHDPlus HR vrt Resampleandclip is done'])
end

%% Milwaukee river

clear; clc;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';
Tifs = [
    dir(fullfile(Folder, '*.tif'));
    dir(fullfile(Folder, '*.vrt'));
    ];
Tifs = Tifs(~[Tifs.isdir]);

in2D  = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHDPlus_HR_AritificalPath_origin/NHDPlus_HR_ArtificialPath_core.shp';
OutFolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHDPlus_HR_ArtificialPath_Processed/';
if exist(OutFolder,'dir') ~= 7
    mkdir(OutFolder);
end

for i = 1 : length(Tifs)
    if strcmpi(Tifs(i).name,'Kletzch_proj.tif') || strcmpi(Tifs(i).name,'UpMax3Null.tif') ||  contains(Tifs(i).name,'_DEM_1m_Proj')
        continue
    end

    bathy_vrt = fullfile(Folder, Tifs(i).name);
    [~, baseName, ~] = fileparts(bathy_vrt);   % baseName 就是不含 .tif 的名字

    dstfolder = fullfile(OutFolder, baseName);
    if exist(dstfolder,'dir') ~= 7
        mkdir(dstfolder);
    end

    % 读取bathy的坐标系
    [st, epsgOut] = system(sprintf('gdalsrsinfo -o epsg "%s"', bathy_vrt));
    assert(st==0, 'gdalsrsinfo 失败：%s', epsgOut);
    tok = regexp(epsgOut,'EPSG:\s*(\d+)','tokens','once');
    assert(~isempty(tok), '未解析到 EPSG：%s', epsgOut);
    tgtEPSG = sprintf('EPSG:%s', tok{1});
 
    % === (2) 计算 bathy 范围 ===
    [~,rows,cols,geoTrans,~,~,~]=RasterInfo(bathy_vrt);
    xMin = geoTrans(1);
    xMax = geoTrans(1) + cols*geoTrans(2);
    yMax = geoTrans(4);
    yMin = geoTrans(4) + rows*geoTrans(6);   % 注意 GT(6) 多为负数

    % === (3) 输出路径 ===
    shpProj   = fullfile(dstfolder, 'NHDPlus_ArtificialPath_proj.shp');
    shpClip   = fullfile(dstfolder, 'NHDPlus_ArtificialPath_Resampleandclip.shp');

    % === (4) 先重投影到与 bathy 相同坐标系 ===
    cmdProj = sprintf('ogr2ogr -f "ESRI Shapefile" -t_srs %s "%s" "%s"', ...
                      tgtEPSG, shpProj, in2D);
    [st,msg] = system(cmdProj);
    assert(st==0, 'ogr2ogr 重投影失败：\n%s', msg);

    % === (5) 再按 bathy 范围裁剪
    cmdClip = sprintf('ogr2ogr -f "ESRI Shapefile" -spat %.3f %.3f %.3f %.3f "%s" "%s"', ...
                      xMin, yMin, xMax, yMax, shpClip, shpProj);
    [st,msg] = system(cmdClip);
    assert(st==0, 'ogr2ogr 裁剪失败：\n%s', msg);

    fprintf('[%d/%d] %s -> 完成（EPSG=%s，范围=[%.1f %.1f %.1f %.1f]）\n', ...
        i, numel(Tifs), Tifs(i).name, tgtEPSG, xMin, yMin, xMax, yMax);

    clear bathy_vrt dstfolder
    clear epsgOut tok
    clear xMin xMax yMax yMin
    clear geoTrans cols rows
    clear shpProj shpClip
    clear cmdProj cmdClip
    % disp([num2str(i),' NHDPlus vrt Resampleandclip is done'])
end

% 至此三步准备都完成了


