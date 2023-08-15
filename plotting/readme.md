# Inference & Plotting

## Introduction
Find here the code to use a trained model to infer on unseen data in the same annotations.json or elsewhere. Jet metrics and plotting requires FJ installed.

## Code Flow
Organise running to cache results at different steps. 

- make_inference.py > make_metrics.py > plot_metrics.py

make_inference.py: 
Inputs -- SSDRealDataset, saved model
Outputs -- np structured array dtype=([('event_no', 'i4'), ('h5file', 'i4'), ('extent','f8', (4)),('t_boxes', 'f4', (30,4)), ('p_boxes', 'f4', (40, 4)), ('p_scores', 'f4', (40))])

make(_phys/jet)_metrics.py:
Inputs -- np structured array 
Outputs -- pkl lists (and lists of lists)

plot(_phys/jet)_metrics.py:
Inputs -- pkl lists
Outputs -- plt plots




<h1>Step A &larr; Step B &larr; Step C</h1>