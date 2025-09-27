import numpy as np
import torch

# trace operator Tr(A) sum of diagonal elements of matrix
A = np.array([[25, 2], [5, 4]])

traceA = 25 + 4

traceA_np = np.trace(A)

# trace operator properties for rearranging linear algebra equations
# Tr(A) = Tr(Aᵀ)
# assuming matrix shapes line up Tr(ABC) = Tr(CAB) = Tr(BCA)
# convenient way to calculate matrix's frobenius norm ||A||ꜰ = √Tr(AAᵀ)

# calculate trace of A_p
A_p = torch.tensor([[-1, 2], [3, -2], [5, 7.]])

# must have same dimensions
# sqA_p = torch.concatenate((A_p, torch.tensor([[0, 0, 0]]).T), axis=1)

traceA_p = torch.trace(A_p)

# demonstrate ||A||ꜰ = √Tr(AAᵀ)
frobA_p = torch.norm(A_p) # using frobenius norm

frob_traceA_p = torch.sqrt(torch.trace(torch.matmul(A_p, A_p.T)))

print(frobA_p)
print(frob_traceA_p)