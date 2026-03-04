import numpy as np
from satpy.composites import GenericCompositor

class HighlightOptimisedTrueColor(GenericCompositor):
    def compute(self, scene, **kwargs):
        data = {}
        for band in ['Oa08', 'Oa06', 'Oa03']:
            band_data = scene[band].data
            modified_data = np.sqrt(0.9 * band_data - 0.055)
            data[band] = modified_data
        return data