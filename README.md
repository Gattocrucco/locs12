# locs12

Code to study the localization of S1 and S2 signals in DarkSide20k.

## Scripts

  * `temps1plots20210131.py`: save performance and diagnostic plots of s1
    temporal localization filters with the version of `temps1.py` from January
    31, 2021.

  * `temps1plots20210203.py`: save performance and diagnostic plots of s1
    temporal localization filters with the version of `temps1.py` from February
    3, 2021.

  * `temps1series0203.py`: save efficiency vs. rate curves of s1 localization
    with the version of `temps1.py` from February 3, 2021.
    
  * `temps1series0203plot.py`: plot the results from the above script (can be
    executed while the other script is still running to show partial results).

## Modules

  * `aligntwin.py`: code to align the ticks of multiple plot scales.

  * `ccdelta.py`: compute the cross-correlation of dicrete points with a
    continuous function.

  * `clusterargsort.py`: filter away values which are close to an higher value
    in a signal.

  * `dcr.py`: generate uniform hits.
  
  * `downcast.py`: downcast numpy data types recursively.
  
  * `filters.py`: filters to be applied to a temporal sequence of hits.
  
  * `named_cartesian_product.py`: cartesian product of arrays.
  
  * `npzload.py`: class to serialize objects to numpy archives.
    
  * `numba_scipy_special/`: module to add support for `scipy.special` functions 
    in numba.

  * `pS1.py`: compute and sample the temporal distribution of S1 photons.
  
  * `qsigma.py`: equivalent of standard deviation with quantiles.
  
  * `runsliced.py`: do something in batches with a progressbar.
  
  * `sampling_bounds.py`: bounds for random ramples.
  
  * `symloglocator.py`: class to place minor ticks on symlog scales.
  
  * `temps1.py`: simulate the temporal localization of S1 signals.
  
  * `textbox.py`: draw a box with text on a plot.
