# package containing all the functions for the multi-fidelity optimization and modeling
# author: Wei Xing
# date: 2023-12-12
# version: 1.0
# history:
# 1.0    2023-12-12    Initial version

# TODO: we need a multi-fidelity data class to store the data and easily access the data
# requirements:
# 1. the data should be stored in a dictionary
# 2. the data should be able to be accessed by the fidelity index
# 3. the data should be able to be accessed by the absolute index
# 4. the data should be able to be accessed by the relative index
# 
# required functions:
# 1. (x,y1,y2) = get_overlapInput_data(fidelity_index1, fidelity_index2)
# 2. (x, y1, x2, y2) = get_uniqueInput_data(fidelity_index1, fidelity_index2)
# 3. add_data(fidelity_index, x, y) and update index system


import torch
import torch
import torch

class MultiFidelityData:
    def __init__(self, fidelity_data):
        self.fidelity_data = fidelity_data
        self.unique_x, self.absolute_index_mapping = self._create_absolute_index_system()
        self.relative_index_mapping = self._create_relative_index_system()

    def _create_absolute_index_system(self):
        combined_x = torch.cat([data['X'] for data in self.fidelity_data])
        unique_x, inverse_indices = torch.unique(combined_x, sorted=True, return_inverse=True)

        index_mapping = {}
        start = 0
        for data in self.fidelity_data:
            end = start + len(data['X'])
            indices = (inverse_indices[start:end] == torch.arange(len(unique_x)).unsqueeze(1)).nonzero(as_tuple=True)[1]
            index_mapping[data['fidelity_index']] = indices.tolist()
            start = end

        return unique_x, index_mapping

    def _create_relative_index_system(self):
        index_mapping = {}
        for i in range(1, len(self.fidelity_data)):
            current_fidelity = self.fidelity_data[i]['X']
            previous_fidelity = self.fidelity_data[i-1]['X']

            # Find overlapping and non-overlapping indices
            overlap = set(current_fidelity.tolist()).intersection(previous_fidelity.tolist())
            overlap_indices = torch.tensor([idx for idx, x in enumerate(current_fidelity) if x in overlap])
            non_overlap_indices = torch.tensor([idx for idx, x in enumerate(current_fidelity) if x not in overlap])

            index_mapping[self.fidelity_data[i]['fidelity_index']] = {'overlap': overlap_indices, 'non_overlap': non_overlap_indices}

        return index_mapping

    def get_absolute_indices(self):
        return self.unique_x, self.absolute_index_mapping

    def get_relative_indices(self):
        return self.relative_index_mapping
    
    def get_overlap_indices(self, fidelity_index1, fidelity_index2):
        overlap_indices = self.relative_index_mapping[fidelity_index1]['overlap']
        non_overlap_indices = self.relative_index_mapping[fidelity_index2]['non_overlap']
        
# Example usage
# fidelity_1 = {'fidelity_index': 'low', 'X': torch.tensor([1, 2, 3, 4, 5]), 'Y': torch.tensor([5, 4, 3, 2, 1])}
# fidelity_2 = {'fidelity_index': 'medium', 'X': torch.tensor([3, 4, 5, 6, 7]), 'Y': torch.tensor([7, 6, 5, 4, 3])}
# fidelity_3 = {'fidelity_index': 'high', 'X': torch.tensor([2, 4, 6, 8]), 'Y': torch.tensor([8, 6, 4, 2])}

fidelity_1 = {'fidelity_index': '0', 'X': torch.tensor([1, 2, 3, 4, 5]), 'Y': torch.tensor([5, 4, 3, 2, 1])}
fidelity_2 = {'fidelity_index': '1', 'X': torch.tensor([3, 4, 5, 6, 7]), 'Y': torch.tensor([7, 6, 5, 4, 3])}
fidelity_3 = {'fidelity_index': '2', 'X': torch.tensor([2, 4, 6, 8]), 'Y': torch.tensor([8, 6, 4, 2])}

mf_data = MultiFidelityData([fidelity_1, fidelity_2, fidelity_3])
unique_x, absolute_indices = mf_data.get_absolute_indices()
relative_indices = mf_data.get_relative_indices()

print("Unique x values (Absolute System):", unique_x)
print("Absolute Index Mapping:", absolute_indices)
print("Relative Index Mapping (Overlap and Non-Overlap):", relative_indices)
