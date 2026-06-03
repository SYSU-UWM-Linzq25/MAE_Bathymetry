
clear;

Folder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/';
subfolder = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_Merge_3DEP_1m/';
d = dir(subfolder);                 % 列出该目录下的所有条目
d = d([d.isdir]);                % 只保留文件夹
d(1:2) = [];

Bathy_Folder1 = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/';
Bathy_Folder2 = '/tank/data/SFS/xinyis/data/bathymetry/USGS_3DEP_bathymetry_DEM/milwaukee_river_3DEP/';

for i = 15 %: length(d)

    % i == 1 || i == 4 || i == 6 || i == 14 || i == 15
    % final_mix_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_Merge_3DEP_1m/MD_PotomacRiver_Bathy_2019/Combined_Bathy_Priority.vrt';
    % LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_1m/MD_PotomacRiver_Bathy_2019/ESA_WorldCover_Resampleandclip.vrt';
    % srcLine = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHDPlus_ArtificialPath_Processed/MD_PotomacRiver_Bathy_2019/NHDPlus_ArtificialPath_Resampleandclip.shp';
    % final_mix_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Bathy_Merge_3DEP_1m/Kletzch_Combined_UpMax3Null/Combined_Bathy_Priority.vrt';
    % LCC_vrt = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/LCC_1m/Kletzch_Combined_UpMax3Null/ESA_WorldCover_Resampleandclip.vrt';
    % srcLine = '/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/NHDPlus_ArtificialPath_Processed/Kletzch_Combined_UpMax3Null/NHDPlus_ArtificialPath_Resampleandclip.shp';

    % 依次尝试三种可能的位置/命名
    candidates = {
        fullfile(Bathy_Folder1, d(i).name, 'Bathy.vrt')           % 1) Folder1/<subdir>/Bathy.vrt
        fullfile(Bathy_Folder2, [d(i).name, '.vrt'])              % 2) Folder2/<name>.vrt
        fullfile(Bathy_Folder2, [d(i).name, '.tif'])              % 3) Folder2/<name>.tif
        };

    bathy_vrt = '';
    for k = 1:numel(candidates)
        if exist(candidates{k}, 'file') == 2   % 或使用 isfile(candidates{k})
            bathy_vrt = candidates{k};
            break
        end
    end

    if isempty(bathy_vrt)
        error('未找到对应的 Bathy 文件：%s（在 %s 或 %s）', d(i).name, Bathy_Folder1, Bathy_Folder2);
    end

    final_mix_vrt = fullfile([Folder,'/Bathy_Merge_3DEP_1m/',d(i).name,'/Combined_Bathy_Priority.vrt']);
    LCC_vrt = fullfile([Folder,'/LCC_1m/',d(i).name,'/ESA_WorldCover_Resampleandclip.vrt']);
    srcLine = fullfile([Folder,'/NHDPlus_HR_ArtificialPath_Processed/',d(i).name,'/NHDPlus_ArtificialPath_Resampleandclip.shp']);

    SearchStep = 100;
    winH = 336;
    winW = 336;

    OutFolder = fullfile(['/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/Tiles_for_MAE/',d(i).name,'/']);
    if exist(OutFolder,'dir') ~= 7
        mkdir(OutFolder);
    end

    cd('/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/')
    RiverPanel_Pixel_Search(bathy_vrt,final_mix_vrt,LCC_vrt,srcLine,SearchStep,winH,winW,OutFolder,d(i).name)
    clear final_mix_vrt LCC_vrt srcLine OutFolder bathy_vrt
end