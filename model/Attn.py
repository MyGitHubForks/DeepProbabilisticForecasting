import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self._device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size).cuda()

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()
            self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size)).cuda() # source code has no device argument

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        # Create variable to store attention energies
        attn_energies = torch.autograd.Variable(torch.zeros(this_batch_size, max_len, device=self._device)).cuda() # B x S
        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                # changing order of :,b in hidden[:,b]
                attn_energies[b, i] = self.score(hidden[:,b], encoder_outputs[i, b].unsqueeze(0)).cuda()

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        if self.method == "general":
            energy = self.attn(encoder_output).cuda()
            energy = energy.squeeze()
            hidden = hidden.squeeze().cuda()
            energy = hidden.dot(energy)

            return energy
        else:
            assert False, "sorry I didn't implement that method yet"
            """
            if self.method == 'dot':
                energy = hidden.dot(encoder_output)
                return energy

            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = energy.squeeze()
                hidden = hidden.squeeze()
                energy = hidden.dot(energy)

                return energy

            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = self.v.dot(energy)
                return energy
            """                          
