def textbox(ax, text):
    kw = dict(
        xytext=(8, 8),
        va='bottom',
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
    
    return ax.annotate(text, (0, 0), **kw)
