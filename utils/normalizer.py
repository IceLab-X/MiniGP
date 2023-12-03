
class Normalizer:
    def __init__(self, inputs, dim=0) -> None:
        self.mean = inputs.mean(dim=dim, keepdim=True)
        self.std = inputs.std(dim=dim, keepdim=True)
        self.dim = dim

    def normalize(self, inputs):
        return (inputs - self.mean) / (self.std + 1e-8)

    def denormalize(self, inputs):
        return inputs * self.std + self.mean


class Dateset_normalize_manager:
    in_signal = ['input', 'in', 'inputs']
    out_signal = ['output', 'out', 'outputs']

    def __init__(self, inputs, outputs, dim=0) -> None:
        self.inp_n = []
        for _inp in inputs:
            self.inp_n.append(Normalizer(_inp, dim=dim))

        self.out_n = []
        for _out in outputs:
            self.out_n.append(Normalizer(_out, dim=dim))

    def _normalize(self, tensor, type, index):
        if type in self.in_signal:
            return self.inp_n[index].normalize(tensor)
        elif type in self.out_signal:
            return self.out_n[index].normalize(tensor)
        else:
            raise ValueError('type should be input or output')

    def normalize_input(self, tensor, index):
        return self._normalize(tensor, 'input', index)
    
    def normalize_output(self, tensor, index):
        return self._normalize(tensor, 'output', index)
    
    def normalize_inputs(self, tensors):
        return [self.normalize_input(t, i) for i,t in enumerate(tensors)]
    
    def normalize_outputs(self, tensors):
        return [self.normalize_output(t, i) for i,t in enumerate(tensors)]
    
    def _denormalize(self, tensor, type, index):
        if type in self.in_signal:
            return self.inp_n[index].denormalize(tensor)
        elif type in self.out_signal:
            return self.out_n[index].denormalize(tensor)
        else:
            raise ValueError('type should be input or output')
        
        
    def denormalize_input(self, tensor, index):
        return self._denormalize(tensor, 'input', index)
    
    def denormalize_output(self, tensor, index):
        return self._denormalize(tensor, 'output', index)
    
    def denormalize_inputs(self, tensors):
        return [self.denormalize_input(t, i) for i,t in enumerate(tensors)]
    
    def denormalize_outputs(self, tensors):
        return [self.denormalize_output(t, i) for i,t in enumerate(tensors)]
    
    def normalize_all(self, inputs, outputs):
        return self.normalize_inputs(inputs), self.normalize_outputs(outputs)
