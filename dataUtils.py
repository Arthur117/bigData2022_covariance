import numpy as np

def get_cat_as_nb(ds):
    """For a given ds, returns the category as a number (0 for storm or dep, 1, ..., 5 for other categories)"""
    cat = np.array(ds['cyclone_category'])
    if cat == 'storm' or cat == 'dep':
        cat = 0
    else: # then it's 'cat-0', 1, ..., or 5
        cat = int(str(cat)[-1])
    return cat

