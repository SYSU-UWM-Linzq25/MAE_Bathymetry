function S = getGdalInfoSummary(rasterFile)

    S = emptyInfo();

    cmd = sprintf('gdalinfo -approx_stats "%s"', rasterFile);
    [status, txt] = system(cmd);

    if status ~= 0
        warning('gdalinfo failed: %s', rasterFile);
        return;
    end

    % Size
    tok = regexp(txt, 'Size is\s+(\d+),\s*(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        S.xsize = str2double(tok{1});
        S.ysize = str2double(tok{2});
    end

    % Pixel Size
    tok = regexp(txt, 'Pixel Size = \(([-+0-9.eE]+),([-+0-9.eE]+)\)', 'tokens', 'once');
    if ~isempty(tok)
        S.pixx = str2double(tok{1});
        S.pixy = str2double(tok{2});
    end

    % NoData
    tok = regexp(txt, 'NoData Value=([-+0-9.eE]+)', 'tokens', 'once');
    if ~isempty(tok)
        S.nodata = str2double(tok{1});
    end

    % Stats line:
    % Minimum=..., Maximum=..., Mean=..., StdDev=...
    tok = regexp(txt, ...
        'Minimum=([-+0-9.eE]+),\s*Maximum=([-+0-9.eE]+),\s*Mean=([-+0-9.eE]+),\s*StdDev=([-+0-9.eE]+)', ...
        'tokens', 'once');

    if ~isempty(tok)
        S.minv = str2double(tok{1});
        S.maxv = str2double(tok{2});
        S.mean = str2double(tok{3});
        S.stdv = str2double(tok{4});
    end

    % Horizontal unit heuristic
    lowerTxt = lower(txt);

    if contains(lowerTxt, 'unit["us survey foot"') || ...
       contains(lowerTxt, 'unit["foot_us"') || ...
       contains(lowerTxt, 'unit["foot",0.3048') || ...
       contains(lowerTxt, 'unit["foot",0.304800')
        S.horizUnit = 'foot';
    elseif contains(lowerTxt, 'unit["metre",1') || ...
           contains(lowerTxt, 'unit["meter",1') || ...
           contains(lowerTxt, 'unit["metre",1.0') || ...
           contains(lowerTxt, 'unit["meter",1.0')
        S.horizUnit = 'meter';
    else
        S.horizUnit = 'unknown';
    end
end