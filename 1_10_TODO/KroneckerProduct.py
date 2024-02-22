import torch 
import torch.nn as nn

class KroneckerProduct(nn.Module):
    """
    Class to perform the Kronecker Product inside the Autograd framework.

    See: https://en.wikipedia.org/wiki/Kronecker_product

    Computes Kronecker Product for matrices A and B, where 
    A has shape (batch_size, Ar, Ac) and B has shape (batch_size, Br, Bc)

    Usage: 
      * Initialise an instance of this class by specifying the shapes of A and B. 
      * Call the class on A and B, which calls the forward function.
    """
    def __init__(self, A_shape, B_shape):
        """
        Inputs: 
            A_shape         A tuple of length 2 specifying the shape of A---(Ar, Ac)
            B_shape         A tuple of length 2 specifying the shape of B---(Br, Bc)
        """

        super(KroneckerProduct, self).__init__()

        # Extract rows and columns. 
        Ar, Ac              = A_shape
        Br, Bc              = B_shape

        # Output size. 
        Fr, Fc              = Ar * Br, Ac * Bc
   
        # Shape for the left-multiplication matrix
        left_mat_shape      = (Fr, Ar)
        # Shape for the right-multiplication matrix. 
        right_mat_shape     = (Ac, Fc)
 
        # Identity matrices for left and right matrices.
        left_eye            = torch.eye(Ar)
        right_eye           = torch.eye(Ac)

        # Create left and right multiplication matrices. 
        self.register_buffer('left_mat', torch.cat([x.view(1, -1).repeat(Br, 1) for x in left_eye], dim=0))
        self.register_buffer('right_mat', torch.cat([x.view(-1, 1).repeat(1, Bc) for x in right_eye], dim=1))
   
        # Unsqueeze the batch dimension.
        self.left_mat       = self.left_mat.unsqueeze(0)
        self.right_mat      = self.right_mat.unsqueeze(0)

        # Function to expand A as required by the Kronecker Product. 
        self.A_expander     = lambda A: torch.bmm(self.left_mat.expand(A.size(0),Fr,Ar), torch.bmm(A, self.right_mat.expand(A.size(0),Ac,Fc)))

        # Function to tile B as required by the kronecker product. 
        self.B_tiler        = lambda B: B.repeat(1, Ar, Ac)

    def forward(self, A, B):
        """
        Compute the Kronecker product for A and B. 
        """
        # This operation is a simple elementwise-multiplication of the expanded and tiled matrices. 
        return self.A_expander(A) * self.B_tiler(B)



