def textbox(ax, text, loc='lower left'):
    M = 8
    locparams = {
        'lower left': dict(xy=(0, 0), xytext=(M,  M), va='bottom'),
        'upper left': dict(xy=(0, 1), xytext=(M, -M), va='top')
    }
    
    kw = dict(
        ha='left',
        fontsize='small',
        xycoords='axes fraction',
        textcoords='offset points',
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='gray',
            boxstyle='round'
        ),
    )
    kw.update(locparams[loc])
    
    return ax.annotate(text, **kw)
