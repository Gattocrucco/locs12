def textbox(ax, text, loc='lower left', **kw):
    """
    Draw a box with text on a matplotlib plot.
    
    Parameters
    ----------
    ax : matplotlib axis
        The plot where the text box is drawn.
    text : str
        The text.
    loc : {'lower left', 'upper left'}
        The location of the box.
    
    Keyword arguments
    -----------------
    Additional keyword arguments are passed to ax.annotate.
    
    Return
    ------
    The return value is that from ax.annotate.
    """
    M = 8
    locparams = {
        'lower left': dict(xy=(0, 0), xytext=(M,  M), va='bottom'),
        'upper left': dict(xy=(0, 1), xytext=(M, -M), va='top')
    }
    
    kwargs = dict(
        ha='left',
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
