# math
import numpy as np
import pandas as pd
from scipy.integrate import odeint, simps
from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline
from scipy.misc import derivative
from scipy.special import lambertw

# plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

import os
from numba import njit # just-in-time compiling
from tqdm.auto import trange, tqdm # nice progress bars
from latticeeasy import run

class Model:

    """ Prototype for different model classes """

    def calculate_turn(self, expansion=True, tfin=2.):
        maxpoint, self.turn_data = turning_point(self.data, self.g**2, self.lamb, self.f**2, expansion=expansion, tfin=tfin)
        self.Phi1, self.loga1 = maxpoint
        
        #  calculate analytic estimate

        phichi2data = self.data["phi"]*self.data["chi2"]
        phichi2 = phichi2data[-50:].mean()

        t0, phi0, v0 = self.data.iloc[-1,1:4]
        x=v0**2/(self.g**2*phichi2)
        self.Phi1_analytic = phi0 + np.sqrt(2./3.)*np.log(1+np.sqrt(3./8.)*x)
        self.t1_analytic = t0+x/v0
        self.tau1_analytic = np.sqrt(self.g*self.v)*(self.t1_analytic - self.tzero)
        if expansion:
            pass #self.phi_analytic = phi0 + np.sqrt(2./3.)*np.log(1.+np.sqrt(3./2.)*(v0*self.turn_data["t"] - .5*self.g**2*phichi2*self.turn_data["t"]**2))
        else:
            pass

        return [maxpoint, self.turn_data]

    def save(self, DIR="."):
        """ Save the zero-crossing and turning data to disk """

        check = {True: "", False: "no"}
        if DIR[-1]=="/":
            DIR = DIR[:-1]

        r = 8 # round to get rid of machine imprecisions
        name = f"log10Omega={round(self.log10Omega, r)}_log10g={round(self.log10g, r)}_lambda={self.lamb}_f={self.f}_" + check[self.expansion] + "exp_" + check[self.vacuum_subtraction] + "vac_" + check[self.self_interaction] + "self"
        self.data.to_csv(f"{DIR}/zerocrossing_{name}.dat", index=False, sep="\t")
        if not self.turn_data.empty:
            self.turn_data.to_csv(f"{DIR}/turn_{name}.dat", index=False, sep="\t")

    def __str__(self):

        return  "hartree model: " + ', '.join([f"{key}={self.parameters[key]}" for key in self.parameters.keys()])

    def plot(self, x="tau", y="auto", yscale='auto', format=None):
        
        compare_models([self], format=format)

    def plot_spectra(self, field='chi', tau_min='zerocrossing', tau_max='zerocrossing', time_skip=1, colorscale='balance', xscale='log', yscale='auto'):
        
        # unless specified, plot only spectra around the tachyonic phase
        kappa2 = self.lamb*self.f**2/(self.g*self.v)
        tau_min = -2.*np.sqrt(kappa2) if tau_min=='zerocrossing' else tau_min
        tau_max = 2.*np.sqrt(kappa2) if tau_max=='zerocrossing' else tau_max
        
        plot_spectra(self.nk, tau_min=tau_min, tau_max=tau_max, time_skip=time_skip, colorscale=colorscale, xscale=xscale, yscale=yscale)


class Hartree(Model):
    """ Calculates the Hartree model for a given set of parameters."""
    
    def __init__(self, params, tbins='auto', kbins=50, self_interaction=True, vacuum_subtraction=True, expansion=True, progress=True, printout=True, phi_ini=0):
        
        # parse parameters
        log10Omega, log10g, lamb, f = parse_parameters(self, params)
        self.self_interaction = self_interaction
        self.vacuum_subtraction = vacuum_subtraction
        self.expansion = expansion
        
        # set the number of tbins depending on location in the parameter region unless explicitly given
        if tbins=='auto':
            base = 2000
            extra = 500.*np.sqrt(lamb*10**log10Omega)*10**(-log10g)
            tbins = base + int((extra//base)*base)
        self.tbins = tbins


        # calculate the evolution for quantities of interest
        self.data, self.chi_k_data, self.kvals = calculate_zerocrossing(log10Omega, log10g, lamb, f, tbins=tbins, kbins=kbins, self=float(self.self_interaction), vacuum_subtraction=float(vacuum_subtraction), expansion=float(expansion), phi_ini=phi_ini, progress=progress)
        self.tzero =  self.data["t"][len(self.data)//2]

        # calculate rescaled momenta k/sqrt(lambda)f
        self.k = self.kvals/np.sqrt(lamb*f**2)

        self.turn_data = pd.DataFrame([])

        # calculate omega_k and n_k TODO: add H term to dot_X
        self.omega_k = []
        self.nk = []
        for kind, kval in enumerate(self.kvals):
            omega_k = np.sqrt(np.abs(kval**2 * np.exp(-2.*self.data["loga"]) + self.g**2*self.data["phi"]**2 - self.lamb*self.f**2 + 3.*self.self_interaction*self.lamb*self.data["chi2"]))
            self.omega_k.append(omega_k)
            
            Xk2 = np.exp(3.*self.data["loga"])*(self.chi_k_data[kind]["re_chi_k"]**2 + self.chi_k_data[kind]["im_chi_k"]**2)
            dot_Xk2 = np.exp(3.*self.data["loga"])*(self.chi_k_data[kind]["dot_re_chi_k"]**2 + self.chi_k_data[kind]["dot_im_chi_k"]**2)
            nk = .5*(dot_Xk2/omega_k + omega_k*Xk2 - 1.)
            self.nk.append(nk)
        self.nk = pd.DataFrame(np.stack(self.nk), index=self.k, columns=self.data["tau"])
        self.nk.axes[0].name = "k"
        self.nk = PowerSpectrum(self.nk) # make callable

        # calculate n_chi
        self.n_chi = np.array(list(map(lambda tind: simps(self.kvals**2*self.nk.iloc[:,tind],self.kvals)/(2.*np.pi**2)*np.exp(-3.*self.data["loga"][tind]), range(len(self.data["tau"])))))
        self.n_chi = pd.DataFrame(np.stack([np.array(self.data["tau"]), self.n_chi]).T, columns=["tau", "n_chi"])

    def info(self):
        """ Bacic info about the run"""

        w = {True: 'yes', False: 'no'}

        print("Model: " + ', '.join([f"{key}={self.parameters[key]}" for key in self.parameters.keys()]) + '\n')        
        
        print("Expansion: " + w[self.expansion])
        print("Self-interaction: " + w[self.self_interaction])
        print("Vacuum subtraction: " + w[self.vacuum_subtraction])

    def report(self, kskip=1, format=None):
        
        """ Reports basic info about the run and plots quantities of interest."""

        self.info()

        field_fig = make_subplots(rows=2, cols=2)
        field_fig.add_trace(go.Scatter(x=self.data["tau"], y=np.abs(self.data["chi2"]), name="chi2"), row=1, col=1)
        field_fig.add_trace(go.Scatter(x=self.data["tau"], y=self.data["phi"], name="phi"), row=1, col=2)
        field_fig.add_hline(y=self.f**2/3., row=1,col=1)
        field_fig.add_trace(go.Scatter(x=self.data["tau"], y=self.data["phi_dot"], name="phi_dot"), row=2, col=1)
        field_fig.add_trace(go.Scatter(x=self.data["tau"], y=self.data["loga"], name="loga"), row=2, col=2)
        field_fig.update_yaxes(exponentformat="power", row=1,col=1, type='log')
        field_fig.update_yaxes(exponentformat="power")
        field_fig.update_xaxes(title="tau")
        if format:
            field_fig.update_layout(width=1000, height=650)
        field_fig.show(format)
        
        # plot n_chi
        particle_fig=make_subplots(rows=1, cols=2)
        particle_fig.add_trace(go.Scatter(x=self.n_chi["tau"], y=self.n_chi["n_chi"], name = "n_chi"), row=1, col=1)
        particle_fig.add_trace(go.Scatter(x=self.data["tau"], y=self.g*np.abs(self.data["phi"])*self.data["chi2"], name="g|phi|<chi2>"), row=1, col=1)
        particle_fig.update_xaxes(title="tau", row=1, col=1)
        
        # plot spectrum
        particle_fig.add_trace(go.Scatter(x=self.k, y=self.nk.iloc[:,-1], name="n_k"), row=1, col=2)
        particle_fig.update_xaxes(exponentformat="power")
        particle_fig.update_xaxes(title="k/sqrt(lambda) f", row=1, col=2)
        if format:
            particle_fig.update_layout(width=1000, height=400)
        particle_fig.show(format)
        
        # make colorbar and colors
        
        colorscale = 'viridis'

        colorbar_trace=go.Scatter(x=[None],
             y=[None],
             mode='markers',
             marker=dict(
                 colorscale=colorscale, 
                 showscale=True,
                 cmin=0,cmax=self.k[-1],
                 colorbar=dict(thickness=20, tickvals=self.k[::4], exponentformat="power", title="k/sqrt(lambda)f")
             )
            )
        
        kcolors=sample_colorscale(colorscale,[kval/self.kvals[-1] for kval in self.kvals])
        

        # plot n_k evolition
        nk_fig = go.Figure()
        for kind in range(len(self.kvals)):
            nk_fig.add_trace(go.Scatter(x=self.data["tau"], y=self.nk.iloc[kind, :], line=dict(color=kcolors[kind])))
        
        nk_fig.add_trace(colorbar_trace)
        nk_fig.update_layout(showlegend=False)
        nk_fig.update_yaxes(exponentformat="power", range=[0.,1.5*self.nk.iloc[0,-1]], title="n_k")
        nk_fig.update_xaxes(title="tau")
        if format:
            nk_fig.update_layout(width=1000, height=500)
        nk_fig.show(format)
        

        # plot X_k evolution
        Xk_fig = make_subplots(rows=1, cols=2)
        
        for kind, kval in enumerate(self.kvals):
            Xk_fig.add_trace(go.Scatter(x=self.data["tau"],y=self.chi_k_data[kind]["re_chi_k"], name=kval, line={'color': kcolors[kind]}),row=1, col=1)
            Xk_fig.add_trace(go.Scatter(x=self.data["tau"],y=self.chi_k_data[kind]["im_chi_k"], name=kval, line={'color': kcolors[kind]}), row=1, col=2)
        
        

        Xk_fig.add_trace(colorbar_trace)
        Xk_fig.update_layout(showlegend=False)
        
        Xk_fig.update_yaxes(exponentformat="power", title="Re(X_k)", row=1,col=1)
        Xk_fig.update_yaxes(exponentformat="power", title="Im(X_k)", row=1,col=2)
        Xk_fig.update_xaxes(title="tau")
        if format:
            Xk_fig.update_layout(width=1000, height=400)
        Xk_fig.show(format)

    
class ImportModel(Model):

    """ Imports model from data file """

    def __init__(self, file):
        
        slashsplit = file.split('/')
        
        self.directory = '/'.join(slashsplit[:-1]) # save the file directory
        if self.directory:
            self.directory += '/'

        self.filename = slashsplit[-1]
        metadata = self.filename[:-4].split('_')

        self.parameters = {entry.split("=")[0]:float(entry.split("=")[1]) for entry in metadata[1:5]}
        
        self.log10Omega = self.parameters["log10Omega"]
        self.log10g = self.parameters["log10g"]
        self.lamb = self.parameters["lambda"]
        self.f = self.parameters["f"]
        self.v = np.sqrt(.5*self.lamb * self.f**4 *10.**(-self.log10Omega))
        self.g = 10**self.log10g 
        
        if metadata[0]=="turn":
            self.turn_data = pd.read_csv(file, sep='\t')
            
            # try to find the zero-crossing data as well
            try:
                file2 = self.directory + "zerocrossing_" + '_'.join(metadata[1:]) + ".dat"
                self.data = pd.read_csv(file2, sep='\t')
            except:
                print(f"Zero-crossing data not found. Will import turn data only.")
                
        if metadata[0]=="zerocrossing":
            self.data = pd.read_csv(file, sep='\t')
            
            # try to find the turn data as well
            file2 = self.directory + "turn_" + '_'.join(metadata[1:]) + ".dat"
            try:
                self.turn_data = pd.read_csv(file2, sep='\t')
            except:
                maxpoint, self.turn_data = turning_point(self.data, self.g**2, self.lamb, self.f**2, expansion=True)
                self.Phi1, self.loga1 = maxpoint
                self.turn_data.to_csv(file2, index=False, sep="\t")
                
        if not self.turn_data.empty:
            self.Phi1 = self.turn_data["phi"].max()
            phi_end = self.turn_data["phi"].iloc[-1]
            
            # if maximum is found then keep it
            if self.Phi1 > phi_end:
                ind = self.turn_data["phi"].argmax()
                self.t1 = self.turn_data["t"].iloc[ind]
                self.tau1 = self.turn_data["tau"].iloc[ind]
                self.loga1 = self.turn_data["loga"].iloc[ind]
            else:
                print(f"Warning: no maximum found. ({self.log10Omega}, {self.log10g})")
                self.Phi1 = np.nan
                self.t1 = np.nan
                self.tau1 = np.nan
                self.loga1 = np.nan

class PowerSpectrum(pd.DataFrame):
    
    """ Spectrum object. Extends DataFrame by making it callable. Normally returns a pivot table of nk(tau, k).
        If called with no arguments returns the final spectrum. If called with a tau value, returns the spectrum n_k(k) for the nearest tau. If called with k value,
        returns the evolution of the occupation number n_k(tau) for the nearest k. If called with both,
        returns the closest value in both dimensions."""
    
    def __call__(self, tau="end", k="full"):
        
        # by default return the final spectrum
        if tau=="end" and k=="full":
            return self.iloc[:,-1]
        
        # if called with a tau value return nearest
        elif tau != "end" and k=="full":
            
            for tauvar in self.columns:
                
                if tau <= tauvar:
                    return self[tauvar]
        
        # if called with k value 
        elif tau=="end" and k != "full":
            
            for kvar in self.index:
                
                if kvar >= k:
                    return self.loc[kvar]

def tzero_f(phi_ini, v_ini, expansion=True):
    """ (cosmic) time of zero crossing """
    return np.sqrt(2./3.)*(np.exp(-np.sqrt(3./2.)*phi_ini)-1.)/v_ini if expansion else -phi_ini/v_ini

def tau_tach(v, g, lamb, f):
    """ tachyonic region in tau """
    return np.sqrt(lamb*f**2/g/v)

def initialize(kvals, v, g2, lamb, f2, phi_ini=0, v_ini=0, expansion=True, errors=False):
    """ Calculate intial conditions for background and fluctuations"""
    
    back = 10. # the intial ratio for tachyonicity and adiabaticity to require

    # initialize background

    no_tach = np.sqrt(lamb*f2/g2) # condition that no modes start tachyonic ; should have g^2 phi^2>>lambda f^2
    adi = np.sqrt(v/np.sqrt(g2)) # condition ensuring adiabaticity; should have |phi_ini| > adi
    
    # if v_ini is set to True then use v as the initial velocity rather than velocity at zero-crossing so remember the value
    if v_ini and type(v_ini) is bool:
        v_ini = v 

    # set initial conditions automatically if not given
    if not phi_ini:    
        if not expansion:
            phi_ini = -back*max(no_tach, adi) # pick initial value such that both conditions are satisfied
            #print("adi" if adi>no_tach else "tach")
        elif v/np.sqrt(g2) > .35/back**2:
            phi_ini=-max(1., back*no_tach) # if no adiabatic initial condition is possible just set to -1 or from tachyonicity
            print(f"Warning: no adiabatic (adi<{1/back}) initial condition is possible. Setting phi_ini={phi_ini}")
        else:
            alpha = v/np.sqrt(g2)*back**2
            phi_ini = np.sqrt(8./3)*lambertw(-np.sqrt(3.*alpha/8.)) # solve initial amplitude from kination dynamics
            if phi_ini.imag == 0:
                phi_ini = min(phi_ini.real, -back*no_tach)
                v = v * np.exp(-np.sqrt(3./2.)*phi_ini)
                #print(f"tach={g2*phi_ini**2/lamb/f2}")
            else:
                raise ValueError("Cannot set inflaton initial condition.")

    # overwrite the intial velocity if specifically given
    if v_ini:
        v = v_ini

    # check that the necessary conditions are satisfied
    if abs(phi_ini) < no_tach:
        if errors:
            return "tachyonic_modes_at_initial_time"
        print('Warning: tachyonic modes at initial time.')
    if abs(phi_ini) < np.sqrt(v/np.sqrt(g2)):
        if errors:
            return "non-adiabatic initial condition"
        print('Warning: initial condition is not adiabatic.')
    if v>1:
        if errors:
            return "quantum gravity regime"
        print("Warning: quantum gravity regime. Initial condition v>M_pl.")

    background_initial_state = [phi_ini, v, 0]

    # initialize fluctuations

    # frequency
    H_dot = -.5*v**2*float(expansion)
    H = np.sqrt((v**2 + .5*lamb*f2**2)/6.)*float(expansion)
    Delta = -1.5*(H_dot + 1.5*H**2) # term coming from expansion
    omega_k_0 = np.sqrt(kvals**2 + g2*phi_ini**2 - lamb*f2 + Delta)
    dot_omega_omega = g2*phi_ini*v - H*(kvals**2 + 3./8.*v**2) # time derivative of omega²/2

    # set fluctuation intial conditions
    Re_chi_k_ini = 1./np.sqrt(2.*omega_k_0)
    Im_chi_k_ini = kvals*0
    Re_dchi_k_ini = -1.5*H*Re_chi_k_ini - dot_omega_omega/omega_k_0**2/2.**1.5
    Im_dchi_k_ini = -np.sqrt(omega_k_0/2.)

    fluctuation_initial_state = [Re_chi_k_ini, Im_chi_k_ini, Re_dchi_k_ini, Im_dchi_k_ini]

    return (background_initial_state, fluctuation_initial_state)

@njit #accelerate with Numba
def derivatives(state, t, k, g2, lamb, f2, chi2, chi_dot2, chi_grad2, self, expansion):
    """ Derivative function to plug into odeint """

    phi, v, loga, chi_k, dchi_k = state
    
    # background EOM
    H = np.sqrt(v**2 + g2*phi**2 * chi2 + chi_dot2 + np.exp(-2.*loga)*chi_grad2 + .5*lamb*(3.*chi2**2 - 2.*chi2*f2 + f2**2))
    H = expansion*H/np.sqrt(6.)
    dv = -3*H*v - g2*phi*chi2
    
    # fluctiation EOM
    ddchi_k = -3.*H*dchi_k - (k**2 * np.exp(-2.*loga) + g2*phi**2 - lamb*f2 + 3.*self*lamb*chi2)*chi_k
    
    return [v, dv, H, dchi_k, ddchi_k]

def evolve_step(kvals, t, background_ini, fluct_ini, g2, lamb, f2, chi2, chi_dot2, chi_grad2, self, expansion=1.):
    
    """ Evolve the system a time step during which chi2 etc. constant"""
    
    # set initial conditions
    phi_ini, v_ini, loga_ini = background_ini
    Re_chi_k_ini, Im_chi_k_ini, Re_dchi_k_ini, Im_dchi_k_ini = fluct_ini
    
    # evolve the real and imaginary parts of the modes for each k value
    fluct_sols= []
    for i in range(len(kvals)):
        
        k = kvals[i]
        Re_chi_k = odeint(derivatives, [phi_ini, v_ini, loga_ini, Re_chi_k_ini[i], Re_dchi_k_ini[i]], t, args=(k, g2, lamb, f2, chi2, chi_dot2, chi_grad2, self, expansion))
        Im_chi_k = odeint(derivatives, [phi_ini, v_ini, loga_ini, Im_chi_k_ini[i], Im_dchi_k_ini[i]], t, args=(k, g2, lamb, f2, chi2, chi_dot2, chi_grad2, self, expansion))
        
        fluct_sols.append([Re_chi_k,Im_chi_k])
        
    return fluct_sols

def calculate_zerocrossing(
    log10Omega, log10g, lamb, f,
    nkratio = .01, # n_k threshold for momentum cutoff
    tbins = 2000, # number of time subdivisions 
    subtbins = 50, # number of datapoints for each time subdivision
    kbins = 20, # number of momentum bins
    self = 1., # self-interaction 1/0 to include/exclude
    expansion = 1., # expansion of the universe 1/0 to include/exclude
    vacuum_subtraction = 1., # 1/0 to subtract/not subtract the vacuum
    phi_ini = 0, # set initial amplitude of the inflaton; if 0 the initial value is set automatically
    v_ini = False,  # set initial value of the velocity to v; falsy value means the intial velocity will be calculated automatically by exptrapolating from v at zero-crossing
    progress = True, # whether to display the progress bar
    errors = False  # return an error signal if in a bad part of parameter space
                         ):

    """ Calculate the zero-crossing evolution. """

    v = np.sqrt(.5*lamb * f**4 *10.**(-log10Omega))
    g = 10**log10g
    
    g2 = g**2
    f2 = f**2
    
    # create momentum array
    kcutratio = np.sqrt(1. - np.log(nkratio)*g*v/(np.pi*lamb*f**2)) # cutoff in units of sqrt(lambda) f
    kcut = kcutratio*np.sqrt(lamb)*f # cutoff in units of M_pl
    kvals = np.linspace(0, kcut, kbins+1) # array of momentum values

    # initialize
    initialization = initialize(kvals, v, g2, lamb, f2, phi_ini=phi_ini, v_ini=v_ini, expansion=expansion, errors=errors)

    # if initialize returned an error signal return it
    if len(initialization) != 2:
        return initialization

    # if all is fine set initial conditions
    background_ini, fluct_ini = initialization
    phiinival = background_ini[0]
    tzero = tzero_f(phiinival,v,expansion)
    chi2 = 0
    chi_dot2 = 0
    chi_grad2 = 0

    # create time array
    tvals = np.linspace(0,2.*tzero, tbins + 1)
    dt = np.linspace(0, tvals[1], subtbins + 1)

    # record intial data
    data = [np.append(background_ini, chi2)]
    chi_k_data = [np.array(fluct_ini)]
    
    # calculate the evolution by solving successive time bins (two steps forward, one step back)
    range_func = trange if progress else range
    for i in range_func(tbins):

        # solve field evolution
        t = tvals[i] + 2.*dt
        solution = evolve_step(kvals, t, background_ini, fluct_ini, g2, lamb, f2, chi2, chi_dot2, chi_grad2, self, expansion)

        # subtract vacuum and calculate expectation values
        chi2_integrand = np.zeros(kbins + 1)
        for knum in range(kbins + 1):

            vac = .5/np.sqrt(np.abs(kvals[knum]**2 * np.exp(-2.*solution[0][0][:,2]) + g2*solution[0][0][:,0]**2 - lamb*f2 + 3.*self*lamb*chi2))
            chik2_normalized = solution[knum][0][:,3]**2 + solution[knum][1][:,3]**2 - vac*vacuum_subtraction
            chi2_integrand[knum] = chik2_normalized.mean()

        # reset the initial conditions for the next time step
        background_ini = solution[0][0][:,:3][subtbins//2]
        fluct_ini = [[solution[knum][0][subtbins//2,3] for knum in range(kbins+1)],
                        [solution[knum][1][subtbins//2,3] for knum in range(kbins+1)],
                        [solution[knum][0][subtbins//2,4] for knum in range(kbins+1)],
                        [solution[knum][1][subtbins//2,4] for knum in range(kbins+1)]]
        
        # record data
        chi2 = simps(kvals**2 * chi2_integrand, kvals)/(2.*np.pi**2)
        data.append(np.append(background_ini, chi2))
        chi_k_data.append(np.array(fluct_ini))
    
    # nicefy data
    tauvals = np.sqrt(g*v)*(tvals-tzero)
    result = np.zeros((tbins+1, 6))
    result[:,0] = tauvals
    result[:,1] = tvals
    result[:,2:] = np.stack(data)
    
    chi_k_data = np.array(chi_k_data)
    chi_k_data = [pd.DataFrame(chi_k_data[:,:,kind], columns=["re_chi_k","im_chi_k","dot_re_chi_k","dot_im_chi_k"]) for kind in range(len(kvals))]
    
    return [pd.DataFrame(result, columns=["tau","t","phi","phi_dot","loga","chi2"]), chi_k_data, kvals]

def background_derivatives_long(state, t, g2, lamb, f2, phichi2, chi_dot2, chi_grad2, expansion):
    """ Derivatives for background evolution assuming <chi2> scales as const/phi*a^-3"""
    
    phi, v, loga = state
    
    H = np.sqrt(v**2 + g2*np.abs(phi)*phichi2*np.exp(-3.*loga) + chi_dot2 + np.exp(-2.*loga)*chi_grad2 + .5*lamb*(3.*phichi2**2*np.exp(-6.*loga)/phi**2 - 2.*phichi2*np.exp(-3.*loga)*f2/np.abs(phi) + f2**2))
    H = expansion*H/np.sqrt(6.)
    dv = -3*H*v - g2*np.sign(phi)*phichi2*np.exp(-3.*loga)
    
    return [v, dv, H]

def turning_point(data, g2, lamb, f2, expansion=True, tfin=2.):
    """ Finds the turning point given zero crossing data"""
    
    g = np.sqrt(g2)
    phi_ini = data["phi"][0]
    v_ini = data["phi_dot"][0]
    if "t" in data.columns: # stanislav
        tzero = data["t"][len(data)//2]
    
    # check if turn already happened
    Phi1 = data["phi"][0]
    Phi1_found = False
    for ind, val in enumerate(data["phi"]):
        if val<Phi1:
            loga1 = data["loga"][ind-1]
            Phi1_found = True
            turn_data = data.loc[:,["tau","t","phi","phi_dot","loga"]]
            break
        else:
            Phi1 = val

    # otherwise extrapolate intil the turn
    if not Phi1_found:
        phichi2data = data["phi"]*data["chi2"]
        phichi2 = phichi2data[-50:].mean()

        t0, phi0, v0, log0 = data.iloc[-1,1:5]
        x=2.*v0**2/(g2*phichi2)
        if expansion:
            tanalytic = v0/(g2*phichi2)#(np.sqrt(x/lambertw(x).real) - 1)/v0
        else:
            tanalytic = v0/(g2*phichi2)
        t = np.linspace(t0, t0 + tfin*tanalytic, 1001)

        chi_dot2 = 0
        chi_grad2 = 0
        sol = odeint(background_derivatives_long, [phi0,v0,log0], t, args=(g2, lamb, f2, phichi2, chi_dot2, chi_grad2, float(expansion)))
        Phi1 = sol[:,0].max()
        ind = sol[:,0].argmax()
        loga1= sol[ind,2]
        turn_data = pd.DataFrame(sol, columns=["phi","phi_dot","loga"])
        turn_data.insert(loc=0, column="t", value=t)
        tau = np.sqrt(g*v_ini)*(t-tzero)
        turn_data.insert(loc=0, column="tau", value=tau)
        
    return [(Phi1,loga1), turn_data]

def parse_parameters(self,params):

    paramnames = ['log10Omega', 'log10g', 'lamb', 'f']

    # parse parameters
    if type(params) == dict:
        self.parameters = params
        log10Omega, log10g, lamb, f = [params[name] for name in paramnames]
    else:
        log10Omega, log10g, lamb, f = params
        self.parameters = {key:value for key,value in zip(paramnames, params)}
    
    # save info about the run
    self.log10Omega = log10Omega
    self.log10g = log10g
    self.v = np.sqrt(.5*lamb * f**4 *10.**(-log10Omega))
    self.g = 10**log10g 
    self.lamb = lamb
    self.f = f

    return (log10Omega, log10g, lamb, f)

def compare_models(models, legend='', legend_title='', title='', phi=True, tach_region=True, returns=False, format=None):
    
    if legend == '':
        legend = [f"model {i}" for i in range(len(models))]
    
    fig1 = go.Figure()
    if phi:
        fig2 = go.Figure()
    
    for i, model in enumerate(models):
    
        fig1.add_trace(go.Scatter(x=model.data['tau'][1:], y=np.abs(model.data['chi2'][1:]), name=legend[i]))
        if phi:
            fig2.add_trace(go.Scatter(x=model.data['tau'], y=model.data['phi'], name=legend[i]))

    fig1.add_hline(y=model.f**2/3., line_dash='dash')
    if tach_region:
        tach = tau_tach(models[0].v, models[0].g, models[0].lamb, models[0].f)
        fig1.add_vrect(x0=-tach, x1=tach, fillcolor="LightSalmon", opacity=0.2, layer="below", line_width=0)

    fig1.update_yaxes(exponentformat="power", type='log')
    fig1.update_layout(legend_title_text=legend_title, title=title, xaxis_title="tau", yaxis_title="chi2")
    if format:
        fig1.update_layout(width=800,height=450)
    fig1.show(format)
    if phi:
        if tach_region:
            fig2.add_vrect(x0=-tach, x1=tach, fillcolor="LightSalmon", opacity=0.2, layer="below", line_width=0)
        fig2.update_layout(legend_title_text=legend_title, title=title, xaxis_title="tau", yaxis_title="phi")
        fig2.update_yaxes(exponentformat="power")
        if format:
            fig2.update_layout(width=800,height=450)
        fig2.show(format) 

    if returns:
        return fig1 if not phi else [fig1, fig2]

def compare_spectra(models, title='', legend='', legend_title='', format=None):

    model_num = len(models)
    if legend=='':
        legend = [f"model {i}" for i in range(model_num)]

    # find the shortest final time
    taumax = models[0].nk.columns[-1]
    for model in models:
        taumax=min(model.nk.columns[-1], taumax)

    # sample a few times
    tauvals = np.linspace(-taumax, taumax, int(taumax//2))

    # extract spectra at those times
    spectra = []
    for model in models:
        spectra.append([])

    for tauval in tauvals:

        for i, model in enumerate(models):    
            for tau in model.nk.columns:
                if tau >= tauval:
                    spectra[i].append(model.nk.loc[:,tau])
                    break
    
    # make figure

    fig = go.Figure()

    for i, tau in enumerate(tauvals):
        
        vis=bool(i//(len(tauvals)-1)) # make only the last one visible
        for j, spectrum in enumerate(spectra):
            fig.add_trace(go.Scatter(x=spectrum[i].index, y=spectrum[i], name=legend[j], visible=vis))

    fig.update_yaxes(exponentformat="power", type="log")
        
    fig.update_layout(updatemenus=[dict(buttons=[dict(label = f'tau={round(tau)}', method = 'update', args = [{'visible': [False]*model_num*i + [True]*model_num + [False]*model_num*(len(tauvals)-1-i)}]) for i, tau in enumerate(tauvals)],
                                        active=len(tauvals)-1
                                        ),
                                    dict(buttons=list([dict(label="Log", method="relayout", args=[{"yaxis.type": "log"}]),
                                        dict(label="Linear",  method="relayout", args=[{"yaxis.type": "linear"}])
                                        ]),
                                        y=.9
                                        )])
    fig.update_layout(legend_title_text=legend_title, title=title, xaxis_title="k/sqrt(lambda)f", yaxis_title="n_k")
                                
    fig.show(format)

def plot_energies(data, x="tau", format=None):

    
    legend = ["phi kinetic", "chi kinetic", "phi gradient", "chi gradient", "potential cross-term", "potential self-interaction"]
    
    fig = go.Figure()
    
    for i, col in enumerate(data.columns[2:]):
    
        fig.add_trace(go.Scatter(x=data[x], y=data[col], name=legend[i]))
    fig.update_yaxes(exponentformat="power", type="log", title="energy density")
    fig.update_xaxes(exponentformat="power", title=x)
    
    fig.show(format)

def plot_spectra(nk_data, tau_min='all', tau_max='all', time_skip=1, colorscale='balance', xscale='log', yscale='auto', format=None):
    """ Plot spectra given a povot table of data"""
    
    # choose the initial and final times for the spectra
    tau_min = nk_data.columns[0] if tau_min=='all' else tau_min
    tau_max = nk_data.columns[-1] if tau_max=='all' else tau_max

    # create a curresponding subset of the spectra
    nk_slice = nk_data[nk_data.columns[tau_min<=nk_data.columns]]
    nk_slice = nk_slice[nk_slice.columns[tau_max>=nk_slice.columns]]
    if time_skip>1:
        nk_slice = nk_slice[nk_slice.columns[::time_skip]]
        
    # determine range and type of plot
    vmax = 1.5*nk_slice.max().max()
    vmin = 0
    if vmax > 100:
        yscale = 'log' if yscale == 'auto' else yscale
    else:
        yscale = 'linear' if yscale == 'auto' else yscale
    if yscale == 'log':
        vmax = np.log10(vmax)
        vmin = -.5


    if type(colorscale)==int:
            colorscale=["balance", "icefire", "jet"][colorscale]
            
    colors = sample_colorscale(colorscale,nk_slice.shape[1])
    fig = go.Figure()

    for i, tau in enumerate(nk_slice.columns):
        spectrum = nk_slice[tau]
        fig.add_trace(go.Scatter(x=spectrum.index, y=spectrum, line=dict(color=colors[i])))

    fig.update_yaxes(exponentformat="power", title="n_k", type=yscale, range=(vmin,vmax))
    fig.update_xaxes(type=xscale, title="k/sqrt(lambda)f")
    fig.update_layout(showlegend=False)
    fig.show(format)