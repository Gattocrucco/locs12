def textbox(ax, text, loc='lower left', **kw):
    """
    Draw a box with text on a matplotlib plot.
    
    Parameters
    ----------
    ax : matplotlib axis
        The plot where the text box is drawn.
    text : str
        The text.
    loc : {'lower left', 'upper left', 'lower right'}
        The location of the box.
    
    Keyword arguments
    -----------------
    Additional keyword arguments are passed to ax.annotate.
    
    Return
    ------
    The return value is that from ax.annotate.
    """
    
    # TODO update dictionaries in kwargs with dictionaries in kw? for bbox
    # (only if both are dictionaries, to allow deletion)
    
    M = 8
    locparams = {
        'lower left' : dict(xy=(0, 0), xytext=( M,  M), va='bottom', ha='left' ),
        'upper left' : dict(xy=(0, 1), xytext=( M, -M), va='top'   , ha='left' ),
        'lower right': dict(xy=(1, 0), xytext=(-M,  M), va='bottom', ha='right'),
    }
    
    kwargs = dict(
        fontsize='x-small',
        xycoords='axes fraction',
        textcoords='offset points',
        bbox=dict(
            facecolor='white',
            alpha=0.5,
            edgecolor='gray',
            boxstyle='round'
        ),
    )
    kwargs.update(locparams[loc])
    kwargs.update(kw)
    
    return ax.annotate(text, **kwargs)
