%% 2025年11月4日
% 处理陈博士提取的河道中心线
% 取样点的间隔后续根据河宽和分辨率改变
% 利用Width的shp点，然后取样，里面有每一段shp的ID
% 可以直接进入取点，然后开始取tile
% 但是在这之前需要先转换DEM尺度

%% 预处理：类型转换-脱离maplineshape类型 + 投影转换并裁剪到相应的河流

clear;
cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

Folder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy3DEP_Merged_Tiff_1m/';
d = dir(Folder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];

for i = 3
    BasinName = d(i).name;

    fprintf('\n========== Start basin: %s ==========\n', BasinName);

    cd('/tank/data/SFS/xinyis/src/MEX_2.3.0'); GDALLoad();
    addpath('/tank/data/SFS/xinyis/src/CREST_Prep');

    % ==== 根路径统一放这里，方便改 ====
    rootPR   = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results';

    % ==== 你的测试用那几行变量定义，改成用 BasinName 拼 ====
    % 3DEP+bathy 合并结果
    final_mix_vrt_1m  = fullfile(rootPR, 'Bathy3DEP_Merged_Tiff_1m',  BasinName, 'Combined_Bathy_Priority_1m.vrt');
    final_mix_vrt_3m  = fullfile(rootPR, 'Bathy3DEP_Merged_Tiff_3m',  BasinName, 'Combined_Bathy_Priority_3m.tif');
    final_mix_vrt_5m  = fullfile(rootPR, 'Bathy3DEP_Merged_Tiff_5m',  BasinName, 'Combined_Bathy_Priority_5m.tif');
    final_mix_vrt_10m = fullfile(rootPR, 'Bathy3DEP_Merged_Tiff_10m', BasinName, 'Combined_Bathy_Priority_10m.tif');

    % Bathy 单独（如果只在前面统计时用，可以保留）
    % 这里假定 bathy_1m 都在一个以 BasinName 命名的 tif/vrt 里，
    bathy_vrt_1m  = fullfile(rootPR, 'Bathy_1m_FixND',  BasinName, 'Bathy_1m.vrt');
    bathy_vrt_3m  = fullfile(rootPR, 'Bathy_3m_FixND',  BasinName, 'Bathy_3m.vrt');
    bathy_vrt_5m  = fullfile(rootPR, 'Bathy_5m_FixND',  BasinName, 'Bathy_5m.vrt');
    bathy_vrt_10m = fullfile(rootPR, 'Bathy_10m_FixND', BasinName, 'Bathy_10m.vrt');

    % LCC (1/3/5/10 m)
    LCC_vrt_1m  = fullfile(rootPR, 'LCC_1m',  BasinName, 'ESA_WorldCover_Resampleandclip_1m.vrt');
    LCC_vrt_3m  = fullfile(rootPR, 'LCC_3m',  BasinName, 'ESA_WorldCover_Resampleandclip_3m.vrt');
    LCC_vrt_5m  = fullfile(rootPR, 'LCC_5m',  BasinName, 'ESA_WorldCover_Resampleandclip_5m.vrt');
    LCC_vrt_10m = fullfile(rootPR, 'LCC_10m', BasinName, 'ESA_WorldCover_Resampleandclip_10m.vrt');

    % 中心线点（已经 reproject 成 Raster 投影的）
    srcLine = fullfile(rootPR, 'CenterRiverLine_skel', 'Reproj', BasinName, 'ESA_WorldCover_Width_proj.shp');

    % 中间结果 & 输出
    TempFolder = fullfile(rootPR, '/CenterRiverLine_skel/', '/Sample_Extract/', BasinName);
    if exist(TempFolder,'dir') ~= 7
        mkdir(TempFolder);
    end

    OutFolder  = fullfile(rootPR, 'Tiles_for_MAE_CO');
    if exist(OutFolder,'dir') ~= 7
        mkdir(OutFolder);
    end

    if contains(d(i).name,'BadgerFinNull') || contains(d(i).name,'Estabrook_Combined') || contains(d(i).name,'KewaFix2Null') || contains(d(i).name,'Kletzch_Combined_UpMax3Null') 
        ExtractPercent = 0.7;  % 你之前用的
    else
        ExtractPercent = 0.4;  % 你之前用的
    end
    winH = 336;
    winW = 336;

    cd('/tank/data/SFS/xinyis/data/bathymetry/Processed_Results')
    RiverPanel_Pixel_Search_skel(bathy_vrt_1m,bathy_vrt_3m,bathy_vrt_5m,bathy_vrt_10m, ...
        final_mix_vrt_1m,final_mix_vrt_3m,final_mix_vrt_5m,final_mix_vrt_10m, ...
        LCC_vrt_1m,LCC_vrt_3m,LCC_vrt_5m,LCC_vrt_10m, ...
        srcLine,ExtractPercent,winH,winW,TempFolder,OutFolder,BasinName)

end



