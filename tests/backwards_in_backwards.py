import torch                                                    
                                                                
class exampleFct(torch.autograd.Function):                      
    @staticmethod                                               
    def forward(self, x):
        output = x**2
        self.save_for_backward(x,output)
        return output
                                                                
    @staticmethod                                               
    def backward(self, dy):                                     
        x,output = self.saved_variables
        with torch.enable_grad():                               
            return torch.autograd.grad(output, x, dy)
                                                                
                                                                
x = torch.tensor([[3., 4.]], requires_grad=True)                  
m = exampleFct.apply(x).sum().backward()                        
print(x.grad.data) 
