from pathlib import Path
import pandas as pd
from collections import defaultdict
import math
import numpy as np

scales = defaultdict(lambda: 11.9)
# scales['BR017-206111 001'] = 7.87

folder = Path('/home/kieran/final_model_outputs_2/')
barcodes = pd.read_csv(folder / '_barcodes.csv')

def get_name(image_name):
    if image_name[:5].upper() == 'BR017':
        vals = barcodes[barcodes['barcode'] == image_name[:9]]['Name']
    elif image_name[:3].upper() == 'BR9':
        vals = barcodes[barcodes['barcode'] == image_name[:10]]['Name']
    elif image_name[:4].upper() == 'BR11':
        vals = barcodes[barcodes['barcode'] == image_name[:11]]['Name']
    
    if len(vals) > 0:
        return vals.iloc[0]
    else:
        print(f'Unknown: {image_name}')
        return 'Unknown'


files = list(folder.glob('b*.csv'))
for p in files:
    print(p)
    ds = pd.read_csv(p, dtype={'genotype':str})
    out = folder / 'out'
    out.mkdir(exist_ok=True)
    
    # remove bad masks and lengths
    ds = ds[ds['multiple_masks'] == False]
    ds = ds[ds['length'] > 1]

    scale_array = ds['image_id'].apply(lambda name: scales[name]).to_numpy()
    print(np.unique(scale_array))

    ds['length_px'] = ds['length']
    ds['perimeter_px'] = ds['perimeter']
    ds['area_px'] = ds['area']
    ds['length_mm'] = ds['length_px'] / scale_array
    ds['perimeter_mm'] = ds['perimeter_px'] / scale_array
    ds['area_mm2'] = ds['area_px'] / scale_array
    ds['name'] = ds['image_id'].apply(get_name)
    ds = ds[['image_id', 'name', 'length_mm', 'perimeter_mm', 'area_mm2', 'treatment', 'genotype']]
    
    ds = ds.reset_index(drop=True)

    ds.to_csv(out / f'{p.stem}_objects.csv')

    averaged_output = defaultdict(list)
    for ind, ind_group in ds.groupby('image_id'):
        averaged_output['image_id'].append(ind)
        averaged_output['name'].append(ind_group.iloc[0]['name'])
        averaged_output['treatment'].append(ind_group.iloc[0]['treatment'])
        averaged_output['genotype'].append(ind_group.iloc[0]['genotype'])
        averaged_output['count'].append(len(ind_group))

        for metric in ['length_mm', 'area_mm2']:
            data = ind_group[metric].sort_values()
            averaged_output[f'{metric}_mean'].append(data.mean())
            averaged_output[f'{metric}_top10_mean'].append(data.nlargest(n=math.ceil(len(data)*0.1)).mean())
            averaged_output[f'{metric}_top25_mean'].append(data.nlargest(n=math.ceil(len(data)*0.25)).mean())
            averaged_output[f'{metric}_top50_mean'].append(data.nlargest(n=math.ceil(len(data)*0.5)).mean())
            averaged_output[f'{metric}_lower50_mean'].append(data.nsmallest(n=math.ceil(len(data)*0.5)).mean())
            upper_c80_i, lower_c80_i = math.ceil((len(data)-1)*0.9), math.floor((len(data)-1)*0.1)
            upper_c50_i, lower_c50_i = math.ceil((len(data)-1)*0.75), math.floor((len(data)-1)*0.25)
            averaged_output[f'{metric}_central80_mean'].append(data.iloc[lower_c80_i:upper_c80_i].mean())
            averaged_output[f'{metric}_central50_mean'].append(data.iloc[lower_c50_i:upper_c50_i].mean())

            averaged_output[f'{metric}_median'].append(data.median())
            averaged_output[f'{metric}_top10_median'].append(data.nlargest(n=math.ceil(len(data)*0.1)).median())
            averaged_output[f'{metric}_top25_median'].append(data.nlargest(n=math.ceil(len(data)*0.25)).median())
            averaged_output[f'{metric}_top50_median'].append(data.nlargest(n=math.ceil(len(data)*0.5)).median())
            averaged_output[f'{metric}_lower50_median'].append(data.nsmallest(n=math.ceil(len(data)*0.5)).median())

    averaged_output = pd.DataFrame(averaged_output)
    averaged_output.to_csv(out / f'{p.stem}_averages.csv')
        


