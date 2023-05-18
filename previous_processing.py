from collections import defaultdict
from coffea import hist, processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
import numba as nb
from fast_histogram import histogram1d, histogram2d
import matplotlib.pyplot as plt
import os
def match_manual(fatjets, fatjetpfcands, idx):
    pfcs = []
    for j in range(len(fatjetpfcands[0])):
        if idx == fatjetpfcands[0][j]["jetIdx"]:
            pfcs.append(fatjetpfcands[0][j]["pFCandsIdx"])
    return pfcs, fatjets.eta, fatjets.phi, fatjets.pt

def histo_pfcand(x, y, cc,  hist_range, bins):
    # event loop
    return histogram2d(y, x, range=hist_range, bins=bins, weights=cc)


def plot(x, y, cc, ids):
    eta_min, eta_max = -0.8, 0.8
    phi_min, phi_max = -0.8, 0.8
    incr = 0.1

    hist_range = [[eta_min, eta_max], [phi_min, phi_max]]
    eta_bins = np.arange(eta_min, eta_max, incr)
    phi_bins = np.arange(phi_min, phi_max, incr)
    image_shape = (eta_bins.shape[0], phi_bins.shape[0])

    test = histo_pfcand(x, y, cc, hist_range, image_shape)
    '''
    fig, ax = plt.subplots()
    plt.imshow(test, origin='lower', vmin=0, vmax=0.85, extent=[eta_bins[0], eta_bins[-1], phi_bins[0], phi_bins[-1]], cmap='viridis')
    #plt.colorbar()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    #ax.set_xlabel('eta_prime')
    #ax.set_ylabel('phi_prime')
    fig.savefig('PFCands_{}.jpg'.format(ids), bbox_inches='tight')
    plt.cla()
    plt.close(fig)
    '''
    return test

def scatt(x, y, cc,ids):
    plt.clf()
    plt.scatter(x, y, c=cc, cmap='viridis', vmin = 0)
    plt.colorbar()
    plt.savefig('scatter_{}.png'.format(ids))
    plt.clf()

def each_event(events, ids):
    fatjets = events.FatJet
    store_fj = []

    for i in range(len(fatjets[0])):
        store_fj.append(fatjets[0][i])

    # remove overlapping jet and leptons

    electrons_veto = events.Electron
    electrons_veto = electrons_veto[electrons_veto.pt > 20.0]
    electrons_veto = fatjets.nearest(electrons_veto)
    # accept jet that doesn't have an electron nearby
    electrons_veto_selection = ak.fill_none(fatjets.delta_r(electrons_veto) > 0.4, True)
    fatjets = fatjets[electrons_veto_selection]

    muons_veto = events.Muon
    muons_veto = muons_veto[muons_veto.pt > 20.0]
    muons_veto = fatjets.nearest(muons_veto)
    # accept jet that doesn't have a muon nearby
    muons_veto_selection = ak.fill_none(fatjets.delta_r(muons_veto) > 0.4, True)
    fatjets = fatjets[muons_veto_selection]

    # gen-match
    fatjets = fatjets[~ak.is_none(fatjets.matched_gen, axis=1)]
    fatjets = fatjets[fatjets.delta_r(fatjets.matched_gen) < 0.4]

    # pre-selection
    selections = {}
    selections['pt'] = fatjets.pt > 200.0
    selections['eta'] = abs(fatjets.eta) < 2.0
    #qcd sample is 0 but for others not
    #selections['hadronFlavour'] = fatjets.matched_gen.hadronFlavour == 0 # jet not originated from b or c
    selections['all'] = selections['pt'] & selections['eta'] #& selections['hadronFlavour']

    fatjets = fatjets[selections['all']]
    fatjets = fatjets[ak.num(fatjets) > 0]
    fatjets = fatjets[ak.argsort(fatjets.pt, axis=1)]
    fatjets = ak.firsts(fatjets)


    #print('length of fj: ', len(fatjets))
    if len(fatjets) != 1:
        return -1

    for j in range(len(store_fj)):
        fji = store_fj[j]
        if abs(fji.eta - fatjets.eta[0]) < 0.1 and abs(fji.phi-fatjets.phi[0]) < 0.1 and abs(fji.pt-fatjets.pt[0]) < 1:
            pfcs, feta, fphi, fpt = match_manual(fatjets, events.FatJetPFCands, j)
    if len(pfcs) == 0:
        return -1

    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    fatjetpfcands = events.PFCands
    fatjetpfcands['delta_phi'] = fatjetpfcands.delta_phi(fatjets)
    fatjetpfcands['delta_eta'] = fatjetpfcands['eta'] - fatjets['eta']
    fatjetpfcands['delta_r'] = fatjetpfcands.delta_r(fatjets)

    eta = ak.to_numpy(fatjetpfcands['delta_eta'])
    phi = ak.to_numpy(fatjetpfcands['delta_phi'])

    eta = eta.flatten()
    phi = phi.flatten()

    fatjetpfcands['pt_ratio'] = fatjetpfcands['pt']/fatjets['pt']
    manual_pt = []
    manual_eta = []
    manual_phi = []
    manual_sum = []

    pt = ak.to_numpy(fatjetpfcands['pt_ratio'])
    pt = pt.flatten()

    pt_sum = ak.to_numpy(fatjetpfcands['pt'])
    pt_sum = pt_sum.flatten()

    for i in pfcs:
        manual_pt.append(pt[i])
        manual_eta.append(eta[i])
        manual_phi.append(phi[i])
        manual_sum.append(pt_sum[i])

    return plot(manual_eta, manual_phi, manual_pt, ids)

if __name__ == "__main__":
    #fname = "/isilon/data/users/dkhanal/anamoly/JetAutoencoder_ntuplizer/data.root"
    fname = "/isilon/data/users/dkhanal/anamoly/JetAutoencoder_ntuplizer/nano_mc2016post_4.root"
    events = NanoEventsFactory.from_root(fname, schemaclass=PFNanoAODSchema).events()
    pts = []
    #min_pt = float('inf')
    for k in range(0, 20000, 5000):
        for i in range(k, k+5000):
            try:
                #print('Event number: ', i)
                pts.append(each_event(events[i:i+1], i))
                #if pt != -1:
                #pts = pts + pt
                #min_pt = min(min(pt), min_pt)
            except:
                #print('error in event number: ', i)
                a = 0
        vector_filename = str(int(k/5000))+'_wjet.npy'
        if not os.path.exists("vector_results"):
            os.makedirs("vector_results")
        np.save(os.path.join("vector_results",vector_filename), pts)
        print(vector_filename)

    #plt.clf()
    #plt.hist(pts, bins='auto')
    #plt.title('pT Ratio in Light-Jet 10000 FJs')
    #plt.xlabel('pT ratio')
    #plt.ylabel('num of FatJets')
    #plt.savefig('pT ratio 10000 FJs in Light-jet.png')
