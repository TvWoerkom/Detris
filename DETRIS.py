import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from more_itertools import collapse

from matplotlib.colors import to_hex

import shapely.geometry as gmt
from shapely import ops, affinity

import warnings
warnings.filterwarnings('ignore')

class Detris:
    def __init__(self, time_dikesize, all_pdfs, core = None, colordict = None, maxtries = 10, maxtotal = 50, evolutionshow = False, finalshow = True):
        '''
        Initialize the DETRIS model and set the initial parameters.
        
        Parameters
        ----------
        time_dikesize : dictionary
            Dictionary containing the time (as key) and surface profile (as item) of the dike for various time periods.
        all_pdfs : dictionary of pd.DataFrames
            Dictionary containing the type ('typ_pdf', 'slope_cdf', 'height_cdf') as keys and the probability density functions as item.
        core : pd.DataFrame, optional
            Dataframe containing the information on local data, if present. Required columns are x, top, bot and typ. The default is None.
        colordict : dict, optional
            Dictionary containing the plotting colors for each simulation material typ. Has the material type as key and the #hex-colorcode as item. The default is None.
        maxtries : int, optional
            Maximum number of inner loops before error. The default is 10.
        maxtotal : int, optional
            Maximum number of outer loops before error. The default is 50.
        evolutionshow : bool, optional
            Boolean to indicate if the dike construction evolution is plotted after each new layer is added. The default is False.
        finalshow : bool, optional
            Boolean to indicate if the final dike buildup is plotted after the simulation has finished. The default is True.

        Returns
        -------
        None.

        '''
        
        self.time_dikesize = time_dikesize
        self.all_pdfs = all_pdfs
        self.maxtries = maxtries
        self.maxtotal = maxtotal
        self.trycounter = 0
        self.totalcounter = 0
        self.blocks = ['left', 'top', 'right']
        
        self.evolutionshow = evolutionshow
        self.finalshow = finalshow

        self.below_zero = checks.z_coords_below_zero(self.time_dikesize) 
        self.maxtries, self.maxtotal = checks.positive_trybreaks(self.maxtries, self.maxtotal)
      
        self.times_present = np.unique(list(self.time_dikesize.keys()))

        self.finsimgeo = np.array(self.time_dikesize[self.times_present[-1]])*100
        self.finsimgeo[:,1]-=self.finsimgeo[:,1].min()
        self.q = 0
        
        self.plotting = plotting(self, colordict)
        self.ground_truth = ground_truth(core)
        
    def simulate(self):
        '''
        Simulates a DETRIS dike and saves a final dataframe of the simulated dike as "self.dikedf"

        Returns
        -------
        None.

        '''
        
        #loop over time periods
        for it, time in enumerate(self.times_present):  
            self.simgeo = np.array(self.time_dikesize[time])*100
            self.baseline = np.array([[self.simgeo[0,0]-self.simgeo[:,1].max()*30,-1],[self.simgeo[0,0]-self.simgeo[:,1].max()*30,0],[self.simgeo[-1,0]+self.simgeo[:,1].max()*30,0]])

            intpol = gmt.Polygon(self.simgeo)
            if self.ground_truth.cores is not None:
                self.ground_truth.cores = gpd.overlay(self.ground_truth.basecores, gpd.GeoDataFrame(geometry = [intpol]))

            #create a small initial dike to start the simulation
            if it == 0:       
                t, pdfs = self.new_layer_params(time)
                topw = 100
                startx = self.simgeo[:,0].mean()
                
                startgeo = False
                while not startgeo:
                    starttyp, h, s = self.get_typ_h_s(pdfs)
                    startgeo, startcoords = self.make_start_shape(startx, topw, h, s)
                    
                    #condition initial starting geometry to available ground truth data
                    if self.ground_truth.cores is not None:
                        startgeo, startcoords, starttyp = self.ground_truth.conditioning(startgeo, starttyp, t, h, s, self.simgeo, startcoords, self.baseline)
                
                self.geos = [startgeo]          
                self.typs = [starttyp]
                
                
                if self.evolutionshow:
                    self.plotting.fill_plot(self)

            else:
                startgeo = np.array(self.time_dikesize[self.times_present[it-1]])*100
                startcoords = startgeo.copy()
            
            self.update_baseline(startcoords)
            self.baseline = self.baseline[1:]
           
            self.finished = False
            self.stuck = False
            if self.below_zero:
                self.stuck = True
                print('User defined dike coordinates have z-levels below 0, aborting simulation..')
                break
            
            #start filling the dike simulation space
            self.totalcounter = 0
            while not self.finished and not self.stuck:     
                
                #select a new reinforcement block and corresponding pdfs
                t, pdfs = self.new_layer_params(time)
                
                #create an initial polygon and condition initial polygon to available ground truth data
                pol = False
                self.trycounter = 0
                while not pol:
                    typ, h, s = self.get_typ_h_s(pdfs)    
                    pol, topcoords = self.add_slice(self.baseline, t, h, s, self.simgeo)
                    
                    if self.ground_truth.cores is not None and pol is not False:
                        pol, topcoords, typ = self.ground_truth.conditioning(pol, typ, t, h, s, self.simgeo, topcoords, self.baseline)
                    
                    self.update_trycounter()
                    if self.trycounter == self.maxtries:
                        break
                           
                if checks.polygons_not_valid(self.baseline, topcoords):
                    continue
                
                #add created polygon to list, and update the dike surface coordinates
                self.geos.append(pol)
                self.typs.append(typ)
                self.update_baseline(topcoords)
                self.q+=1
                if self.evolutionshow:
                    self.plotting.fill_plot(self)
            
                self.finished = checks.finished(self.simgeo, self.baseline)
                if self.finished: 
                   self.typs, self.geos = self.plotting.clip_geos(self.typs, self.geos, intpol)
              
            if self.stuck:
                print('Simulation algorithm is stuck, aborting simulation...\nTry increasing "maxtries" or "maxtotal" and check for very small spacing between ground truth datapoints.')
                break
            
        if not self.stuck:    
            if self.finalshow:
                self.plotting.fill_plot(self)
            
            self.make_final_dike_df()
            
    def make_start_shape(self, startx, topw, h, s):
        '''
        Function to make an initial top shape to start the simulation with.
        Parameters
        ----------
        startx : float
            Center x-coordinate for the start shape.
        topw : float
            Width of the initial dike shape crest.
        h : float
            Height of the initial dike shape crest.
        s : float, positive.
            Slope (m/m) of the initial dike slope.

        Returns
        -------
        startgeo : shapely.geometry.Polygon
            Polygon of the initial dike shape.
        startcoords : numpy.array
            Array of the coordinates of the intial dike shape polygon.

        '''
        startgeo = gmt.Polygon([[startx-topw/2-h/s,0],[startx-topw/2,h],[startx+topw/2, h], [startx+topw/2+h/s,0]])  
        startcoords = np.array(startgeo.boundary.coords[:-1])
        return startgeo, startcoords

    def new_layer_params(self, time):
        '''
        Function to select the appropriate probabilities for dike layer orientation, dike layer thickness and dike layer material from the larger database, 
        given the construction period and the pre-constructed dike geometry.
        
        Parameters
        ----------
        time : float
            Endtime of current construction period, as present in the time_dikesize dictionary.

        Returns
        -------
        t : str
            Type of next reinforcement block, either "left", "right", or "top".
        pdfs : dict of pd.Series
            Dictionairy with the "typ_pdf", "slope_cdf", and "height_cdf" as keys, containing a pd.Series of the probabilities for the given "t" and "time" .

        '''
        probs = self.define_probabilities()
        t = np.random.choice(self.blocks, p = probs)
        
        pdfs = {}
        for key, pdf in self.all_pdfs.items():
            pdf_times = pdf.index.get_level_values('period').unique()
            if time >= pdf_times.min():
                get_times = pdf_times[pdf_times<=time]
            else:
                get_times = pdf_times[pdf_times==pdf_times.min()]
                
            if key == 'typ_pdf':
                pdfs[key] = pdf.loc[get_times, t, :].groupby('side').mean().loc[t]
            else: 
                new_pdf = pdf.loc[get_times, t, :].groupby('side').mean().loc[t]
                new_pdf.index = new_pdf.index.astype(float)
                pdfs[key] = new_pdf
        return t, pdfs  

    def get_typ_h_s(self, pdfs):
        '''
        Function to sample a material type, layer height and layer slope from the probability functions.

        Parameters
        ----------
        pdfs : dict of pd.Series
            Dictionairy with the "typ_pdf", "slope_cdf", and "height_cdf" as keys, containing a pd.Series of the probabilities.

        Returns
        -------
        typ : str
            Material type for the next DETRIS layer.
        h : float
            Layer thickness for the next DETRIS layer.
        s : float
            Layer slope for the next DETRIS layer.
        '''
        typ_pdf = pdfs['typ_pdf']
        typ = np.random.choice(typ_pdf.index, p = typ_pdf.values.astype(float))
        
        h_pdf = pdfs['height_cdf']
        rand_hx = np.random.uniform(0, 1)
        h = np.interp(rand_hx, h_pdf.values, h_pdf.index)*100

        s_pdf = pdfs['slope_cdf']
        rand_sx = np.random.uniform(0, 1)
        s = np.interp(rand_sx, s_pdf.values, s_pdf.index)
        s = abs(s)
        return typ, h, s

    def define_probabilities(self):
        '''
        Function to calculate probabilities for the likelyhood of the next reinforcement location. 

        Returns
        -------
        probs : numpy.array
            Array containing the probability of the next reinforcement location. The array is for a [left. top, right] DETRIS block.

        '''
        leftside = self.baseline[:np.where(self.baseline[:,1]==self.baseline[:,1].max())[0][0]+1]
        leftxs = np.arange(leftside[0,0], leftside[-1,0],1)
        intline = np.interp(leftxs, leftside[:,0], leftside[:,1])
        topline = np.interp(leftxs, self.simgeo[:,0], self.simgeo[:,1])
        diff = np.maximum(topline-intline, 0)
        if (diff.size != 0) & (topline.sum()!=0) :
            probleft = diff.sum()/topline.sum()
        else:
            probleft = 0.
        
        rightside = self.baseline[np.where(self.baseline[:,1]==self.baseline[:,1].max())[0][-1]:]
        rightxs = np.arange(rightside[0,0], rightside[-1,0],1)
        intline = np.interp(rightxs, rightside[:,0], rightside[:,1])
        topline = np.interp(rightxs, self.simgeo[:,0], self.simgeo[:,1])
        diff = np.maximum(topline-intline, 0)
        if (diff.size != 0) & (diff.sum() !=0):
            probright = diff.sum()/topline.sum()

        else:
            probright = 0.
        
        topps = self.baseline[np.where(self.baseline[:,1]==self.baseline[:,1].max())[0]]
        testxs = np.arange(topps[0,0],topps[-1,0],1)
        topys = np.interp(testxs, topps[:,0], topps[:,1])
        simys = np.interp(testxs, self.simgeo[:,0], self.simgeo[:,1])
        topxs = testxs[topys<simys]
        if topxs.size == 0:
            probtop = 0.1*max(probright, probleft)
        else:
            topwidth = topxs.max()-topxs.min()
            topspace = np.maximum(simys-topys, 0)
            probtop = max(0, topwidth/200-0.9)*(topspace.sum()/simys.sum())
            probtop = max(0.01, probtop)

        probs = np.array([probleft, probtop, probright])
        probs/=np.nansum(probs)
        probs[np.isnan(probs)] = 0
        #print(probs)
        if (probs==0).all():
            probs[1] = 1
        return probs

    def mini(self, step, side, dh, s, dxbase, dxtop, getarea, t, intpol, final):
        '''
        Function that optimizes the new DETRIS layer shape. While taking into account the geometry of the preconstructed DETRIS dike,
        it can change the polygon shape while as much as possible retaining the original reinforcement surface area.

        Parameters
        ----------
        step : TYPE
            DESCRIPTION.
        side : numpy.array
             Coordinates of the current dike surface, at the location where the new DETRIS layer will be added.
        dh : float
            Height difference between the highest and lowest point the new DETRIS layer.
        s : float
            Layer slope for the next DETRIS layer.
        dxbase : float
            Largest x-coordinate difference between the points the horizontal bottom part of the new DETRIS layer.
        dxtop : float
            Largest x-coordinate difference between the points the horizontal top part of the new DETRIS layer.
        getarea : float
            Area of the constructed initial polygon.
        t : str
            Type of next reinforcement block, either "left", "right", or "top".
        intpol : shapely.geometry.Polygon
            Polygon of the preconstructed DETRIS dike surface geometry.
        final : bool
            Boolean to indicate whether this function is run in the optimization phase (False) or when finalizing the layer (True).

        Returns
        -------
        diffarea: float
            Difference between the inipol area (getarea) and the new optimized polygon area (neware).

        '''
        if t == 'left':
            polcoords = [[side[-1,0]+dxtop+step, side[-1,1]], [side[1,0]+dh/s+dxbase, side[-1,1]], [side[1,0]+dxbase, side[1,1]], [side[-1,0]+dxtop-dh/s+step, side[1,1]]]
        if t == 'right':
            polcoords = [[side[-1,0]+dxtop+step, side[-1,1]], [side[1,0]+dh/s+dxbase, side[-1,1]], [side[1,0]+dxbase, side[1,1]], [side[-1,0]+dxtop-dh/s+step, side[1,1]]]

        pol = gmt.Polygon(polcoords)
        sideline = gmt.LineString(side)
        
        if not pol.is_valid:
            return np.random.uniform(getarea*1000,getarea*10000)
        
        newpol = pol.difference(intpol)
        if type(newpol) != gmt.Polygon:
            newpol = list(collapse(newpol))
            newpol = newpol[np.argmin([sideline.distance(n) for n in newpol])]
        
        if newpol.distance(sideline) > 1E-3:
            return np.random.uniform(getarea*1000,getarea*10000)
        
        if final:
            return newpol
        newarea = newpol.area
        diffarea = abs(getarea-newarea)
        return diffarea

    def get_inipol(self, side, h, s):
        '''
        Function to create an initial shape for the next DETRIS layer.

        Parameters
        ----------
        side : numpy.array
            Coordinates of the current dike surface, at the location where the new DETRIS layer will be added.
        h : float
            Layer thickness for the next DETRIS layer.
        s : float
            Layer slope for the next DETRIS layer.

        Returns
        -------
        dh : float
            Height difference between the highest and lowest point the new DETRIS layer.
        dxbase : float
            Largest x-coordinate difference between the points the horizontal bottom part of the new DETRIS layer.
        dxtop : float
            Largest x-coordinate difference between the points the horizontal top part of the new DETRIS layer.
        getarea : float
            Area of the constructed initial polygon.

        '''
        topslopes = abs(np.divide(*(side[-1]-side).T[::-1]))
        if ((topslopes[1:][~np.isnan(topslopes[1:])]<s).any()):# & (not (topslopes[1:][~np.isnan(topslopes[1:])]>s).all()):
            topslopeloc = np.nanargmax(topslopes)
            topp = side[topslopeloc]
            dytop = side[:,1].max()-topp[1]
            dxtop = dytop/topslopes[topslopeloc]-dytop/s
            dybot = side[:,1].max()-side[1,1]
            dxbot = dybot/s-dybot/topslopes[1]
            dxs = np.array([dxbot, dxtop])
            dxtop = dxs[np.argmax(abs(dxs))]
        else:
            dxtop = 0
        
        #print(side)
        baseslopes = abs(np.divide(*(side[1]-side[1:]).T[::-1]))
        #print(baseslopes)
        if ((baseslopes[~np.isnan(baseslopes)]<s).any()):# & (not (baseslopes[1:][~np.isnan(baseslopes[1:])]>s).all()):
            baseslopeloc = np.nanargmin(baseslopes)
            basep= side[baseslopeloc+1]
            dybase = basep[1]-side[1,1]
            dxbase = dybase/baseslopes[baseslopeloc]-dybase/s
            dycrest = side[:,1].max()-side[1,1]
            dxcrest = dycrest/baseslopes[-1]-dycrest/s
            dxs = abs(np.array([dxbase, dxcrest]))
            dxbase = dxs[np.argmax(abs(dxs))]
        else:
            dxbase = 0.
        dh = side[:,1].max()-side[:,1].min()
        sidex = np.sqrt(h**2 + (h/baseslopes[-1])**2)
        getarea = dh*sidex
        return dh, dxbase, dxtop, getarea


    def add_slice(self, baseline, t, h, s, simgeo):
        '''
        Function to add a new DETRIS layer to the existing geometry. Ground truth data is not taken into account, and conditioning is done in a later step.

        Parameters
        ----------
        baseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.
        t : str
            Type of next reinforcement block, either "left", "right", or "top".
        h : float
            Layer thickness for the next DETRIS layer.
        s : float
            Layer slope for the next DETRIS layer.
        simgeo : numpy.array
            Array of [[x1, z1], [x2, z2]] containing the final geometry of the current construction time period.

        Returns
        -------
        pol : shapely.geometry.Polygon or False
            Polygon of the new DETRIS reinforcement layer. "False" if no suitable Polygon can be determined.
        topcoords : numpy.array or False
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer. "False" if no suitable Polygon can be determined.

        '''
        intpol = gmt.Polygon(baseline).buffer(0)
        if t == 'left':
            side = baseline[:np.where(baseline[:,1]==baseline[:,1].max())[0][0]+1]
            
            dh, dxbase, dxtop, getarea = self.get_inipol(side, h, s)
            
            done = False
            trycounter = 0
            while not done:
                startsimp =  [np.random.uniform(-800,0), np.random.uniform(0, 800)]
                step = minimize_scalar(self.mini, 
                                       startsimp, 
                                       args = (side, dh, s, dxbase, dxtop, getarea, t, intpol, False), 
                                       tol = 1E-3, 
                                       options = {'maxiter':1000})
                done = step.fun<5
                trycounter+=1
                if trycounter > 100:
                    return False, False
              
            pol = self.mini(step.x, side, dh, s, dxbase, dxtop, getarea, t, intpol, True)

        if t == 'right':
            side = baseline[np.where(baseline[:,1]==baseline[:,1].max())[0][-1]:][::-1]

            dh, dxbase, dxtop, getarea = self.get_inipol(side, h, s)
           
            invs = -s
            dxbase = -dxbase
            dxtop = -dxtop
            # side = side[]
           
            done = False
            trycounter = 0
            while not done:
                startsimp =  [np.random.uniform(-800,0), np.random.uniform(0, 800)]
                step = minimize_scalar(self.mini, 
                                       startsimp, 
                                       args = (side, dh, invs, dxbase, dxtop, getarea, t, intpol, False), 
                                       tol = 1E-3, 
                                       options = {'maxiter':1000})
                done = step.fun<5
                trycounter+=1
                if trycounter > 100:
                    return False, False

            pol = self.mini(step.x, side, dh, invs, dxbase, dxtop, getarea, t, intpol, True)
            
        if t == 'top':
            baseline = baseline.round(6)
            toploc = np.where(baseline[:,1]==baseline[:,1].max())[0]
            # print(baseline[:,0])
            topps = baseline[[toploc[0], toploc[-1]]]
            slopes = np.array([s,-s])
            h = min(h, np.diff(topps[:,0])[0]*np.product(abs(slopes))/abs(slopes).sum()*0.7)
            topcoords = [topps[0],[topps[0,0]+h/slopes[0], topps[0,1]+h], [topps[1,0]+h/slopes[1], topps[1,1]+h], topps[1]]    
            pol = gmt.Polygon(topcoords)
            
        if pol is False:
            return False, False
        if not pol.is_empty and (type(pol)!=gmt.MultiPolygon):
            gmtbase = gmt.LineString(baseline)
            topcoords = np.array([p for p in pol.exterior.coords[:] if (gmt.Point(p).distance(gmtbase)>1E-5) | (p[0] in pol.bounds[::2])])
            topcoords = np.unique(topcoords, axis = 0)
            #print(baseline, topcoords)
            basecoords = baseline[(baseline[:,0]>topcoords[:,0].min()) & (baseline[:,0] < topcoords[:,0].max())]
            polcoords = np.r_[topcoords, basecoords[::-1]]
            if polcoords.shape[0] < 3:
                return False, False
            pol = gmt.Polygon(polcoords).buffer(0)
        else:
            pol = False
            topcoords = False
        return pol, topcoords
    
    def make_final_dike_df(self):    
        '''
        Function to construct a pandas.DataFrame of the final dike geometry.

        Returns
        -------
        None.

        '''
        dikedf = gpd.GeoDataFrame(self.typs, columns = ['texture'], geometry = self.geos)
        dikedf['geometry'] = dikedf.scale(0.01,0.01, origin = (0,0))
        self.dikedf = dikedf[dikedf.geometry.type == 'Polygon']
    
    def update_trycounter(self):
        '''
        Helper function to update the iteration count when trying to add a new DETRIS layer. Will provoke an error if self.stuck becomes True.

        Returns
        -------
        None.

        '''
        self.trycounter+=1
        if self.trycounter == self.maxtries:
            self.totalcounter+=1
            if self.totalcounter == self.maxtotal:
                self.stuck = True
    
    def update_baseline(self, topcoords):
        '''
        Function to update the baseline array (containing the current DETRIS dike surface geometry) with the coordinates of the new DETRIS layer.
        
        Parameters
        ----------
        topcoords : numpy.array
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer.

        Returns
        -------
        None.

        '''
        self.baseline = self.baseline[(self.baseline[:,0]<topcoords[:,0].min()) | (self.baseline[:,0] > topcoords[:,0].max())]
        self.baseline = np.r_[self.baseline,topcoords]
        self.baseline = np.unique(self.baseline, axis = 0)
        self.baseline = self.baseline.round(6)
    
class ground_truth:
    def __init__(self, basecores):
        '''
        Initialize the ground truth data, if available. Creates a ground truth geopandas.GeoDataFrame.

        Parameters
        ----------
        basecores : pandas.DataFrame or None
            Dataframe containing the information on local data, if present. Required columns are x, top, bot and typ. The default is None.

        Returns
        -------
        None.

        '''
        if basecores is not None:
            basecores['geometry'] = [gmt.LineString([[row.x, row.top], [row.x, row.bot]]) for i, row in basecores.iterrows()]
            cores = gpd.GeoDataFrame(basecores.copy())
            cores = cores.dissolve('typ').reset_index()
            cores = cores[~cores.is_empty]
            #cores['geometry'] = [ops.linemerge(collapse(c)) for c in cores.geometry]
            cores = cores.explode().reset_index(drop=True)
            cores['top'] = cores.bounds.maxy
            cores['bot'] = cores.bounds.miny
            cores['x'] = cores.bounds.minx
            cores = cores.sort_values(['x', 'top'], ascending = [True, True]).reset_index(drop=True)
            cores['geometry'] = cores.scale(100, 100, origin = (0,0))
            cores['change'] = [g.bounds[3] for g in cores.geometry]
            cores['realx'] = cores.bounds.minx
            self.basecores = cores
            self.cores = self.basecores.copy()
        else:
            self.cores = None
        
    def conditioning(self, pol, typ, t, h, s, simgeo, topcoords, tbaseline):
        '''
        Function that conditions a preconstructed DETRIS layer polygon to the available ground truth data.

        Parameters
        ----------
        pol : shapely.geometry.Polygon
            A polygon of the initial DETRIS reinforcement layer.
        typ : str
            Material type for the next DETRIS layer.
        t : str
            Type of next reinforcement block, either "left", "right", or "top".
        h : float
            Layer thickness for the next DETRIS layer.
        s : float
            Layer slope for the next DETRIS layer.
        simgeo : numpy.array
            Array of [[x1, z1], [x2, z2]] containing the final geometry of the current construction time period.
        topcoords : numpy.array
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer. 
        tbaseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.

        Returns
        -------
        pol : shapely.geometry.Polygon or False
            The polygon of the next DETRIS reinforcement, now conditioned to the available ground truth data.    
            "False" if no suitable Polygon can be determined.
        topcoords: numpy.array or False
            Array of the top (surface) coordinates of the next DETRIS reinforcement, now conditioned to the available ground truth data.
            "False" if no suitable Polygon can be determined.
        typ: str or False
            The material type of the next DETRIS reinforcement, now conditioned to the available ground truth data.    
            "False" if no suitable Polygon can be determined.
        '''
        tbaseline = self.make_temp_baseline(tbaseline, topcoords)
        top_line = gmt.LineString(topcoords)
      
        corexs, hard_data = self.get_hard_data(pol)
        n_typs = hard_data.typ.nunique()
        baselinepol = gmt.Polygon(np.r_[[[tbaseline[0,0], -100]], tbaseline, [[tbaseline[-1,0], -100]]])
      
        if n_typs == 0:
            pass
        elif n_typs == 1:
            typ = hard_data.typ.iloc[0]
        else:
            newpol_vert = self.fit_vertical(hard_data, pol, topcoords, t, s, top_line, baselinepol)
            newpol_hori = self.fit_horizontal(hard_data, pol, topcoords, t, s, top_line, baselinepol)
            
            pospols = gpd.GeoSeries([newpol_vert, newpol_hori], index = ['vert', 'hori'])
            pospols = pospols[~pospols.is_empty]
            if pospols.size == 0:
                return False, False, False
            else:
                pol_cond = np.random.choice(pospols.index)
                newpol = pospols[pol_cond]
            # if newpol_hori is False:
            #     return False, False, False
            #newpol = newpol_hori
            pol, topcoords = self.get_refined_pol_topcoords(tbaseline, newpol)
                
        if checks.multiple_typs_at_top(tbaseline, topcoords, self.cores):
          # print('type at top error')
          return False, False, False
        
        corexs, hard_data = self.get_hard_data(pol)
        hard_data = hard_data[hard_data.geometry.length>1]
        if (hard_data.typ.nunique()>1):
            return False, False, False
        elif hard_data.typ.nunique()==1:
            typ = hard_data.typ.iloc[0]
        else:
            pass
        return pol, topcoords, typ
    
    def get_hard_data(self, pol):
        '''
        Function to find the ground truth data that intersect with the DETRIS polygon.

        Parameters
        ----------
        pol : shapely.geometry.Polygon
            A polygon of the initial DETRIS reinforcement layer.

        Returns
        -------
        corexs : numpy.array
            Array containing the unique x-coordinates of the intersecting ground truth data.
        hard_data : geopandas.GeoDataFrame
            GeoDataFrame containing the ground truth data that intersect with the DETRIS polygon.

        '''
        hard_data = gpd.overlay(self.cores, gpd.GeoDataFrame(geometry=[pol]), how='intersection')
        hard_data  = hard_data[hard_data.length>1]
        hard_data['surface'] = [gmt.LineString([[x, -1E9], [x, 1E9]]).intersection(pol).bounds[3] for x in hard_data.geometry.bounds.minx]
        #hard_data['surface'] = [hard_data[hard_data.x==x].unary_union.bounds[3] for x in hard_data.x]    
        hard_data['y_over'] = hard_data.change - hard_data.surface
        corexs = hard_data.bounds.minx.unique()
        return corexs, hard_data
    
    def make_temp_baseline(self, baseline, topcoords):
        '''
        Create a temporarily updated baseline, without changing the original data.

        Parameters
        ----------
        baseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.
        topcoords : numpy.array
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer.

        Returns
        -------
        tbaseline : numpy.array
            Temporarily saved array of [[x1, z1], [x2, z2]] coordinates for the updated dike surface geometry.

        '''
        if baseline[:,1].max() == 0:
            tbaseline = topcoords[topcoords[:,1]==0]
            tbaseline = np.r_[[[tbaseline[0,0], tbaseline[0,1]-1]], tbaseline]
        else:
            tbaseline = baseline.copy()
        return tbaseline
    
    def fit_vertical(self, hard_data, pol, topcoords, t, s, top_line, baselinepol):
        '''
        Function to fit the preconstructed DETRIS reinforcement polygon to the ground truth data by a vertical movement.

        Parameters
        ----------
        hard_data : geopandas.GeoDataFrame
            GeoDataFrame containing the ground truth data that intersect with the DETRIS polygon.
        pol : shapely.geometry.Polygon
            A polygon of the initial DETRIS reinforcement layer.
        topcoords : numpy.array
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer.
        t : str
            Type of next reinforcement block, either "left", "right", or "top".
        s : float
            Layer slope for the next DETRIS layer.
        top_line : shapely.geometry.LineString
            Linestring of the preconstructed initial DETRIS layer "topcoords".
        baselinepol : shapely.geometry.Polygon
            A polygon of the baseline coordinates, representing the current DETRIS dike shape.

        Returns
        -------
        newpol: shapely.geometry.Polygon
            New DETRIS reinforcement layer polygon, fitted to the available ground truth data.

        '''
        errorpol = gmt.box(0,0,0,0).intersection(gmt.box(1,1,1,1))
        hard_data['miny'] = hard_data.bounds.miny
        lowest_change = hard_data.groupby('realx').miny.min().to_frame()
        # if t != 'top':
        minx_line = gpd.GeoSeries([gmt.LineString([[x, -1E9],[x, 1E9]]) for x in lowest_change.index])
        intps = minx_line.intersection(top_line)
        if not (intps.type == 'Point').all():
            return errorpol
        pol_y_at_change = intps.y.values
        
        lowest_change['y_dist'] = abs(lowest_change.miny-pol_y_at_change)
        furthest_away = hard_data[hard_data.realx==lowest_change.y_dist.idxmax()]
        furthest_away = furthest_away.loc[furthest_away.change.idxmin()]

        othercores = hard_data[hard_data.typ!=furthest_away.typ]
        

        core_grp = othercores.groupby('x')
        first_other_typ = core_grp.apply(lambda x: x.loc[x[x.typ!=furthest_away.typ].bounds.miny.idxmin()])

        if t == 'top':
            gety = furthest_away.change
            cury = np.interp(hard_data.bounds.minx.min(), topcoords[:,0], topcoords[:,1])
            sright = np.divide(*np.diff(topcoords[-2:].T[::-1], 1))[0]
            sleft = np.divide(*np.diff(topcoords[:2].T[::-1], 1))[0]
            if (hard_data.bounds.maxy.max() != pol.bounds[3]):# & (firstdata.change == firstdata.geometry.bounds[3]):
                cury = pol.bounds[3]
            topcoords[topcoords[:,1]==topcoords[:,1].max(),1]-=cury-gety
            topcoords[np.where(topcoords[:,1]==topcoords[:,1].max())[0][0],0]+=(gety-cury)/sleft
            topcoords[np.where(topcoords[:,1]==topcoords[:,1].max())[0][1],0]+=(gety-cury)/sright
            newpol = gmt.Polygon(topcoords)
        else:     
            b = first_other_typ.surface-first_other_typ.miny
            constrain_core = first_other_typ.loc[b.idxmin()]
            
            minx_line = gpd.GeoSeries([gmt.LineString([[x, -1E9],[x, 1E9]]) for x in [constrain_core.realx]])
            intps = minx_line.intersection(top_line)
            pol_y_at_change = intps.y.values
            top_change = constrain_core.geometry.bounds[1]-pol_y_at_change

            temp_newpol = affinity.translate(pol, yoff = top_change).difference(baselinepol)        
            temp_newpol = gpd.GeoSeries(temp_newpol).explode()
            newpol = temp_newpol[temp_newpol.intersects(furthest_away.geometry)]

            if newpol.size == 0:
                newpol = gmt.box(0,0,0,0).intersection(gmt.box(1,1,1,1))
            else:
                newpol = newpol.iloc[0]
        return newpol

    def fit_horizontal(self, hard_data, pol, topcoords, t, s, top_line, baselinepol):
        '''
        Function to fit the preconstructed DETRIS reinforcement polygon to the ground truth data by a horizontal movement.

        Parameters
        ----------
        hard_data : geopandas.GeoDataFrame
            GeoDataFrame containing the ground truth data that intersect with the DETRIS polygon.
        pol : shapely.geometry.Polygon
            A polygon of the initial DETRIS reinforcement layer.
        topcoords : numpy.array
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer.
        t : str
            Type of next reinforcement block, either "left", "right", or "top".
        s : float
            Layer slope for the next DETRIS layer.
        top_line : shapely.geometry.LineString
            Linestring of the preconstructed initial DETRIS layer "topcoords".
        baselinepol : shapely.geometry.Polygon
            A polygon of the baseline coordinates, representing the current DETRIS dike shape.

        Returns
        -------
        newpol: shapely.geometry.Polygon
            New DETRIS reinforcement layer polygon, fitted to the available ground truth data.

        '''

        errorpol = gmt.box(0,0,0,0).intersection(gmt.box(1,1,1,1))

        baseline = np.array(baselinepol.boundary.coords[1:-2])
        hard_data['miny'] = hard_data.bounds.miny
        lowest_change = hard_data.groupby('realx').miny.min().to_frame()
        if t != 'top':
            miny_line = gpd.GeoSeries([gmt.LineString([[baseline[0, 0], y],[baseline[-1, 0], y]]) for y in lowest_change.miny])
            intps = miny_line.intersection(top_line)
            if not (intps.type == 'Point').all():
                return errorpol
            pol_x_at_change = intps.x.values
            
            lowest_change['x_dist'] = abs(lowest_change.index-pol_x_at_change)
            furthest_away = hard_data[hard_data.realx==lowest_change.x_dist.idxmax()]
            furthest_away = furthest_away.loc[furthest_away.change.idxmin()]
        else:
            furthest_away = hard_data.loc[hard_data.change.idxmin()]
        
        othercores = hard_data[hard_data.typ!=furthest_away.typ]
        
        core_grp = othercores.groupby('x')
        first_other_typ = core_grp.apply(lambda x: x.loc[x[x.typ!=furthest_away.typ].bounds.miny.idxmin()])

        if t == 'top':
            gety = furthest_away.change
            cury = np.interp(hard_data.bounds.minx.min(), topcoords[:,0], topcoords[:,1])
            sright = np.divide(*np.diff(topcoords[-2:].T[::-1], 1))[0]
            sleft = np.divide(*np.diff(topcoords[:2].T[::-1], 1))[0]
            if (hard_data.bounds.maxy.max() != pol.bounds[3]):# & (firstdata.change == firstdata.geometry.bounds[3]):
                cury = pol.bounds[3]
            topcoords[topcoords[:,1]==topcoords[:,1].max(),1]-=cury-gety
            topcoords[np.where(topcoords[:,1]==topcoords[:,1].max())[0][0],0]+=(gety-cury)/sleft
            topcoords[np.where(topcoords[:,1]==topcoords[:,1].max())[0][1],0]+=(gety-cury)/sright
            newpol = gmt.Polygon(topcoords)
        else:
            if t == 'left':
                temp_s = s
            else:
                temp_s = -s
            
            b = first_other_typ.bounds.miny-temp_s*first_other_typ.realx
            constrain_core = first_other_typ.loc[b.idxmin()]
                      
            miny_line = gpd.GeoSeries([gmt.LineString([[baseline[0, 0], y],[baseline[-1, 0], y]]) for y in [constrain_core.miny]])
            pol_x_at_change = miny_line.intersection(top_line).x.values            
            side_change = constrain_core.geometry.bounds[0]-pol_x_at_change
            
            temp_newpol = affinity.translate(pol, xoff = side_change).difference(baselinepol)
            temp_newpol = gpd.GeoSeries(temp_newpol).explode()
            newpol = temp_newpol[temp_newpol.intersects(furthest_away.geometry)].iloc[0]
        return newpol
            
    def get_refined_pol_topcoords(self, tbaseline, newpol):
        '''
        Function to recalculate the coordinates of the top surface of the new DETRIS layer. Function is instated to disable floating point errors.

        Parameters
        ----------
        tbaseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.
        newpol :  shapely.geometry.Polygon
            DETRIS reinforcement layer polygon.

        Returns
        -------
        pol :  shapely.geometry.Polygon
            New updated DETRIS reinforcement layer polygon.
            "False" if no suitable Polygon can be determined.
        topcoords : numpy.array or "False"
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer.
            "False" if no suitable Polygon can be determined.

        '''
        gmtbase = gmt.LineString(tbaseline)
        topcoords = np.array([p for p in newpol.exterior.coords[:] if (gmt.Point(p).distance(gmtbase)>1E-5) | (p[0] in newpol.bounds[::2])])
        topcoords = np.unique(topcoords, axis = 0)
        basecoords = tbaseline[(tbaseline[:,0]>topcoords[:,0].min()) & (tbaseline[:,0] < topcoords[:,0].max())]
        polcoords = np.r_[topcoords, basecoords[::-1]]
        if polcoords.shape[0]<3:
            pol, topcoords = False
        else:
            pol = gmt.Polygon(polcoords).buffer(0)
        return pol, topcoords
            
class plotting:
    def __init__(self, parent, colordict = None):
        '''
        Inital parameters for plotting related functions in DETRIS. 
        Sets a random color for a material type if a possible material type is not specified in colordict.
        
        Parameters
        ----------
        parent : class
            The DETRIS class, to inherit parameters from.
        colordict : dictionary, optional
            Dictionary containing the plotting colors for each simulation material typ. H
            as the material type as key and the #hex-colorcode as item. The default is None. The default is None.

        Returns
        -------
        None.

        '''
       
        for i, v in vars(parent).items():
            setattr(self, i, v)
        
        if colordict is None:
            colordict = {}
        
        possible_typs = self.all_pdfs['typ_pdf'].columns.unique()
        not_present = possible_typs[~np.isin(possible_typs, list(colordict.keys()))]
        
        use_cmap = 'nipy_spectral'
        colormap = plt.get_cmap(use_cmap)

        for i, c in zip(np.linspace(0, 1, not_present.size), not_present):
            colordict[c] = to_hex(colormap(i))
        
        self.colordf = pd.Series(colordict)
        
    def fill_plot(self, parent):
        '''
        Function to create a plot of the DETRIS dike and its ground truth data (if available).
        
        Parameters
        ----------
        parent : Detris class
            Detris class, which includes all parameters needed for plotting.

        Returns
        -------
        None.

        '''
        intpol = gmt.Polygon(parent.simgeo)
        tfintyps, tgeopols = self.clip_geos(parent.typs, parent.geos, intpol)

        fig, ax= plt.subplots(figsize = (20,7))
        for c, typ in zip(tgeopols, tfintyps):
            if not c.is_empty and type(c) == gmt.Polygon:
                ax.fill(*np.array(c.boundary.coords[:]).T/100, self.colordf[typ], alpha = 0.6)
                temp_finsimgeo = parent.finsimgeo.copy()/100
                ax.set_ylim(temp_finsimgeo[:,1].min(), temp_finsimgeo[:,1].max()+1)
                ax.set_xlim(temp_finsimgeo[0,0]-1, temp_finsimgeo[-1,0]+1)
        ax.plot(*parent.simgeo.T/100, 'k--')
        if parent.ground_truth.cores is not None:
            parent.ground_truth.cores.scale(0.01, 0.01, origin = (0,0)).plot(ax = ax, color = [self.colordf[t] for t in parent.ground_truth.cores.typ], linewidth = 8)
        #fig.savefig(f'../Figures/t{q}.jpg')
        plt.show()
        
    def clip_geos(self, typs, geos, intpol):
        '''
        Function to clip the detris polygon geometries to the outline of the dike.

        Parameters
        ----------
        typs : list
            List of material types for the DETRIS layers.
        geos : list
            List of shapely.geometry.Polygons for the DETRIS layers.
        intpol : shapely.geometry.Polygon
            Polygon of the preconstructed DETRIS dike surface geometry.

        Returns
        -------
        typs : list
            List of material types for the DETRIS layers that are inside the intpol.
        geos : list
            List of shapely.geometry.Polygons for the DETRIS layers that are inside the intpol.

        '''
        time_dikedf = gpd.GeoDataFrame(typs, geometry = geos, columns = ['texture'])
        time_dikedf['geometry'] = time_dikedf.intersection(intpol)
        time_dikedf = time_dikedf.explode()
        time_dikedf = time_dikedf[~time_dikedf.is_empty].reset_index(drop=True)
        geos = time_dikedf.geometry.to_list()
        typs = time_dikedf.texture.to_list()
        return typs, geos

class checks:
    def multiple_typs_at_top(baseline, topcoords, cores):
        '''
        The DETRIS model gets stuck if the horizontal crest of the dike contains more than one material type, or if it intersects with cores of different material types.
        This function checks for this matter.

        Parameters
        ----------
        baseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.
        topcoords : numpy.array or False
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer. 
        cores : geopandas.GeoDataFrame
            GeoDataFrame containing the information on local ground trugh data data.

        Returns
        -------
        bool
            Boolean indicating whether or not the horizontal crest of the dike contains more than one material type.

        '''
        posnewbase = checks.update_test_baseline(baseline, topcoords)
        topnewbase = posnewbase[posnewbase[:,1] == posnewbase[:,1].max()]
        higherbase = topnewbase.copy()
        higherbase[:,1]+=0.1
        # print(np.r_[topnewbase, higherbase[::-1]])
        if np.r_[topnewbase, higherbase[::-1]].shape[0] >= 3:
            posbaseline = gmt.Polygon(np.r_[topnewbase, higherbase[::-1]])
            basedata = gpd.overlay(cores, gpd.GeoDataFrame(geometry=[posbaseline]), how='intersection')
            firsttyps = [basedata[basedata.bounds.minx == x].iloc[0].typ for x in np.unique(basedata.bounds.minx)]
            #print(firsttyps)
            if len(firsttyps) > 1:
                #plot_temppol(gmt.Polygon(topcoords), gmt.Polygon(topcoords), cores, baseline)
                return True
            else:
                return False
        return False
    
    def polygons_not_valid(baseline, topcoords):
        '''
        Function to check the validity of the created DETRIS layer polygon. 
        Validity is seen as nicely being placed on top of the previously known geometry outline.

        Parameters
        ----------
        baseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.
        topcoords : numpy.array or False
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer. 

        Returns
        -------
        pol_not_valid_if : bool
            Boolean inversely indicating the validity of the created DETRIS layer polygon. "True" if polygon not valid.

        '''
        if topcoords is False:
            pol_not_valid_if = True
        else:
            base_linestring = gmt.LineString(baseline)
            dists = []
            for tc in topcoords:
                dists.append(gmt.Point(tc).distance(base_linestring))
            dists = np.array(dists)
            pol_not_valid_if = dists[dists<1].size <= 1
        return pol_not_valid_if

    def finished(simgeo, baseline):  
        '''
        Function to check if the simulation space is entirely filled, and DETRIS is finished for the corresponding time period.
        
        Parameters
        ----------
        simgeo : numpy.array
            Array of [[x1, z1], [x2, z2]] containing the final geometry of the current construction time period.
        baseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.

        Returns
        -------
        finished : bool
            Boolean to indicate if DETRIS is finished for the corresponding time period. "True" indicates finished.

        '''
        xs = np.arange(simgeo[0,0], simgeo[-1,0],1)
        topline = np.interp(xs, simgeo[:,0], simgeo[:,1]).round(3)
        testbase = np.interp(xs, baseline[:,0], baseline[:,1]).round(3)
        finished = (testbase >= topline).all()
        return finished
    
    def update_test_baseline(baseline, topcoords):

        '''
        Function to update the baseline array (containing the current DETRIS dike surface geometry) with the coordinates of the new DETRIS layer.


        Parameters
        ----------
        baseline : numpy.array
            Array of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.
        topcoords : numpy.array
            Array of the top (surface) coordinates of the new DETRIS reinforcement layer.

        Returns
        -------
        baseline : numpy.array
            Updated rray of [[x1, z1], [x2, z2]] coordinates for the current dike surface geometry.

        '''
        baseline = baseline[(baseline[:,0]<topcoords[:,0].min()) | (baseline[:,0] > topcoords[:,0].max())]
        baseline = np.r_[baseline,topcoords]
        baseline = np.unique(baseline, axis = 0)
        baseline = baseline.round(6)
        return baseline
    
    def z_coords_below_zero(time_dikesize):
        '''
        Check if any of the dike surface profile z-coordinates < 0, which is not allowed.

        Parameters
        ----------
        time_dikesize : dictionary
            Dictionary containing the time (as key) and surface profile (as item) of the dike for various time periods.

        Returns
        -------
        below_zero : bool
            True if any ike surface profile z-coordinates < 0.

        '''
        below_zero = False
        for key, item in time_dikesize.items():
            if item[:,1].min() < 0:
                below_zero = True
        return below_zero
    
    def positive_trybreaks(maxtries, maxtotal):
        '''
        The number of tries before erroring cannot be smaller than 1. 
        This function sets the trynumber to 1 if the user-defined value < 1.
        
        Parameters
        ----------
        maxtries : int
            Maximum number of inner loops before error. 
        maxtotal : int
            Maximum number of outer loops before error. 

        Returns
        -------
        maxtries : int
            Maximum number of inner loops before error. 
        maxtotal : int
            Maximum number of outer loops before error. 

        '''
        if maxtries < 1:
            print(f'maxtries needs to be at least 1, current value of {maxtries} is set to 1')
            maxtries = 1
        if maxtotal < 1:
            print(f'maxtotal needs to be at least 1, current value of {maxtotal} is set to 1')
            maxtotal = 1
        return maxtries, maxtotal
