import numpy as np

class VDat:
    def __init__(self, roi):
        self.roi = roi
        self.block_spacing = []
        self.pixel_size = []
    