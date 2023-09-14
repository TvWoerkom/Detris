import pandas as pd
import numpy as np

import DETRIS

#%% hypothetical data
typ_pdf = pd.read_csv('typ_pdf.csv', index_col = [0,1])
slope_cdf = pd.read_csv('slope_cdf.csv', index_col = [0,1])
height_cdf = pd.read_csv('height_cdf.csv', index_col = [0,1])

evolutionshow = True
finalshow = True
    
time_dikesize = {1500: np.array([[5,0],[10,3],[15,3],[20,0.]]),
                 2200: np.array([[0,0],[10,5],[15,5],[25,0]])}

all_pdfs = {'typ_pdf': typ_pdf, 
        'slope_cdf': slope_cdf,
        'height_cdf': height_cdf}

maxtries = 10
maxtotal = 50

core = None
core = pd.read_csv('test_core.csv', index_col = 0)

colordict = {}
colordict['nan'] = '#ffffff'
colordict['v'] = '#9e4f41'
colordict['vz1'] = '#9e7141'
colordict['vk1'] = '#6b4f28'
colordict['vk3'] = '#87755b'
colordict['g'] = '#d9a521'
colordict['gz2'] = '#dfb80a'
colordict['gz3'] = '#e3c500'
colordict['gz4'] = '#e6c50f'
colordict['z'] = '#e8c521'
colordict['zs1'] = '#e8c521'
colordict['zs2'] = '#f2de05'
colordict['zs3'] = '#fae603'
colordict['zs4'] = '#ffff00'
colordict['zk'] = '#f2de05'
colordict['zkx'] = '#f2de05'
colordict['zk2'] = '#d5e61e'
colordict['lz1'] = '#4bc490'
colordict['lz3'] = '#4bc471'
colordict['kz1'] = '#81ba3c'
colordict['kz2'] = '#a1c44b'
colordict['kz3'] = '#c1cf5b'
colordict['ks1'] = '#009100'
colordict['ks2'] = '#209b0f'
colordict['ks3'] = '#40a51e'
colordict['ks4'] = '#61b02d'
colordict['k'] = '#009100'
colordict['kh1'] = '#009100'
colordict['h'] = '#4a3105'
colordict['st'] = '#9c1616'
colordict['none'] = '#ffffff'

for i in range(1):
#     print(i)
#     core = pd.DataFrame()
#     #xcoords = np.random.uniform(2, 23, 4)
#     xcoords = [6,10,15,19]
#     for x in xcoords:
#         layers = np.random.randint(8,12)
#         materials  = np.random.choice(typ_pdf.columns, layers)
#         tops = np.r_[sorted(np.random.uniform(0, 4.98, layers-1)), 5].round(2)
#         bots = np.r_[[0], tops[:-1]]
#         coredf = pd.DataFrame([np.zeros(layers)+x, tops, bots, materials], index = ['x', 'top', 'bot', 'typ']).T[::-1]
#         core = pd.concat([core, coredf])
#     core.reset_index(inplace = True, drop = True)
#     core = None
    
    #%%
    Detris = DETRIS.Detris(time_dikesize = time_dikesize, 
                           all_pdfs = all_pdfs, 
                           core = core,
                           colordict = colordict,
                           evolutionshow = True,
                           maxtries = maxtries,
                           maxtotal = maxtotal)
    
    Detris.simulate()
