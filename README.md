# data_mc_plotter

Short explanation of how to write a config.

## Definiton of the General Options
```
[General]
Components: SumMC,IC86II
Title: Level 3 Data -$ MC
IDKeys: [I3EventHeader.Run, I3EventHeader.Event, I3EventHeader.SubEvent]
Outpath: None
Observables: [SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino.energy.log,
              SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino.x,
              SplineMPETruncatedEnergy_SPICEMie_DOMS_Neutrino.zenith.cos]
Uncertainties: SumMC
Alphas: 0.682689492, 0.9, 0.99
```
* Components: Components that should be plotted names refer to the definitions of the components in the ini
* Title: Title shown at the top of the plot
* IDKeys: Keys in to match different tables
* Outpath: Default is 'None' and the results we be saved in the folder output
* Observables: List of observables $TableName$.$ColumnName$.$Transformation$. Possible Transformations .log/.sin/.sindeg/.cos/.cosdeg. * can be used when all components should be plotted
* Uncertainties: Component for which Uncertainties is shown
* Alphas: Confidence-Levels shown

```
[Blacklist]
Columns: [SubEventStream, fit_status, type, time, pdg_encoding, exists]
Tables: [I3EventHeader, SRTHVInIcePulses]
Observales: None
```
When * is Used für Observables. Columns/Tables/Observables can be blacklisted.

## Definition of componets

```
[11374_baseline]
Type: MC
Label: Muon Neutrino NuGen
Weight: NeutrinoWeights.honda2006_gaisserH3a_elbert_v2_numu-conv-nuflux
FileList: [/home/mathis/Documents/icecube/data_mc_plotter/test_data/11374/file1.hd5, /home/mathis/Documents/icecube/data_mc_plotter/test_data/11374/file2.hd5]
Livetime: 400

[IC86II]
Type: Data
Label: Data Burnsample
Livetime: 57600
Directory: /home/mathis/Documents/icecube/data_mc_plotter/test_data/data/
MaxFiles: 1
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
```

* [$ComponentName$]: Name used to internal reference of the different components
* Type: Possible types are Data/MC
* Label: Name shown in the legend
* Directory: Directory/Filepattern used to find files (runs glob for the path)
* FileList: [File1, File2] specific files
* MaxFiles: Maximum number of files when used directory
* Livetime: Livetime of the component
* Color: Matplotlibname or hexcode to specify the color of the component
* Aggregation: $Comp1$+$Comp2$-$Comp3$ possible operators (+, -, *, /)
* KeepComponents: Whether the components used for the aggregation should be shown in the plot
