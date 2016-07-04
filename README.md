# data_mc_plotter

Short explanation of how to write a config
```
[General]
Components: SumMC,IC86II
Title: Level 3 Data -> MC
IDKeys: [I3EventHeader.Run, I3EventHeader.Event, I3EventHeader.SubEvent]
Outpath: None
Observables: [SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino.energy.log,
              SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino.x,
              SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino.zenith.cos]
Uncertainties: SumMC
Alphas: 0.682689492, 0.9, 0.99
```




[Blacklist]
Columns: [SubEventStream, fit_status, type, time, pdg_encoding, exists]
Tables: [I3EventHeader, SRTHVInIcePulses]
Observales: None

[11374_baseline]
Type: MC
Label: Muon Neutrino NuGen
Weight: NeutrinoWeights.honda2006_gaisserH3a_elbert_v2_numu-conv-nuflux
Directory: /home/mathis/Documents/icecube/data_mc_plotter/test_data/11374/
Livetime: 400

[IC86II]
Type: Data
Label: Data Burnsample
Livetime: 57600
Directory: /home/mathis/Documents/icecube/data_mc_plotter/test_data/data/
Color: w

[Corsika]
Type: MC
Label: Corsika
Aggregation: 11058+11057
KeepComponents: False
Color: #D62728

[11058]
Type: MC
Label: Low Energetic CORSIKA
Weight: CorsikaWeights.GaisserH3aWeight
Directory: /home/mathis/Documents/icecube/data_mc_plotter/test_data/11057/
Livetime: 16000


[11057]
Type: MC
Label: High Energetic CORSIKA
Weight: CorsikaWeights.GaisserH3aWeight
Directory:/home/mathis/Documents/icecube/data_mc_plotter/test_data/11058/
Livetime: 10500


[SumMC]
Type: MC
Label: Sum Simulation
Aggregation: 11058+11057+11374_baseline
KeepComponents: False
Color: w
