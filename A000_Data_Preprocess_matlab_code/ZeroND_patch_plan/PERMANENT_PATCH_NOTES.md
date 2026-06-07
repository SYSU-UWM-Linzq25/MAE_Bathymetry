# Permanent patches after the staged Santiam test succeeds

## 1. B001_01_Bathmatry_3DEP_Processing_All.m

In the bathy + 3DEP merge block, do not use only:

```matlab
C = B;
isHoleB = isnan(B) | (B == globalND);
```

Use a river-specific zero-fill policy and explicit initialization:

```matlab
zeroFillRivers = {'OR_SantiamRiverTB_Topobathy_1_D23'};
zeroTol = 1e-8;
zeroIsNoData = ismember(name, zeroFillRivers);

validB = isfinite(B) & ~isnan(B) & (B ~= globalND);
if zeroIsNoData
    validB = validB & abs(B) > zeroTol;
end
validD = isfinite(D) & ~isnan(D) & (D ~= globalND);

C = ones(size(B)) * globalND;
C(validB) = B(validB);
mask_fill = ~validB & validD;
C(mask_fill) = D(mask_fill);
C(~isfinite(C) | isnan(C)) = globalND;
```

Also:

- clear old `_tiles/tile_*.tif` before rebuilding;
- move `verifySaved = 0` inside each river loop;
- build the final VRT with both `-srcnodata -999999` and `-vrtnodata -999999`;
- use the downstream-consistent name `Combined_Bathy_Priority_1m.vrt`.

## 2. B003s_Build_SimpleFinalMask_LCC_BathyValid_ForRiver.m

For confirmed zero-fill rivers, exclude zero from `bathy_valid`:

```matlab
bathy_valid = isfinite(B) & ~isnan(B) & (B ~= bathyND) & (B > -1e20);
if zeroIsNoData
    bathy_valid = bathy_valid & abs(B) > zeroTol;
end
```

Prefer reading the canonical bathy root (`Bathy_1m_FixND_ZeroFixed`) so this is only a fallback.

## 3. RiverPanel_Pixel_Search_skel.m, Step 2

Do not count values with only `~isnan(tile)`. Use explicit valid masks:

```matlab
validBathy = isfinite(tile) & ~isnan(tile) & tile ~= bathyND & tile > -1e20;
k_valid = find(validBathy);

validMix = isfinite(tileBathmatry3DEP_outRiver) & ...
           ~isnan(tileBathmatry3DEP_outRiver) & ...
           tileBathmatry3DEP_outRiver ~= finalND & ...
           tileBathmatry3DEP_outRiver > -1e20;
k_valid_OutRiver = find(validMix);
```

## 4. RiverPanel_Pixel_Search_skel.m, Step 5

- restore the full window boundary check;
- write Float32 output with a forced `globalND = -999999`;
- canonicalize invalid values before write;
- use `globalND`, not `NaN`, for `TileOutRiver`;
- for a confirmed zero-fill river, throw an error if any zero survives the fixed merge.

```matlab
globalND = -999999;
tileBathmatry3DEP(~isfinite(tileBathmatry3DEP) | ...
                  tileBathmatry3DEP == globalND) = globalND;

if zeroIsNoData && any(abs(tileBathmatry3DEP(:)) <= zeroTol & ...
                       tileBathmatry3DEP(:) ~= globalND)
    error('Unexpected zero remains in fixed merged input.');
end

tileBathmatry3DEP_outRiver = tileBathmatry3DEP;
tileBathmatry3DEP_outRiver(tile_LCC == 1) = globalND;

WriteRaster(..., 6, 'GTiff', globalND); % Float32
```
