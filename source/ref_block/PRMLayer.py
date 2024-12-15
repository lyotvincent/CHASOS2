
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class PRMLayer(nn.Module):
    '''
    Position-Aware Recalibration Module: Learning From Feature Semantics and Feature Position
    https://github.com/13952522076/PRM
    https://www.ijcai.org/proceedings/2020/0111.pdf
    '''
    def __init__(self,groups=64,mode='dotproduct'):  
        super(PRMLayer, self).__init__()
        self.mode = mode
        self.groups = groups
        self.max_pool = nn.AdaptiveMaxPool2d(1,return_indices=True)
        self.weight = nn.Parameter(torch.zeros(1,self.groups,1,1)) # type: ignore
        self.bias = nn.Parameter(torch.ones(1,self.groups,1,1)) # type: ignore
        self.sig = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.one = nn.Parameter(torch.ones(1,self.groups,1)) # type: ignore
        self.zero = nn.Parameter(torch.zeros(1, self.groups, 1)) # type: ignore
        self.theta = nn.Parameter(torch.rand(1,2,1,1)) # type: ignore
        self.scale =  nn.Parameter(torch.ones(1)) # type: ignore

    def forward(self, x):

        b,c,h,w = x.size()
        position_mask = self.get_position_mask(x, b, h, w, self.groups) # shape [b*number,2,h,w]
        # Similarity function
        query_value, query_position = self.get_query_position(x, self.groups)  # shape [b,c,1,1] [b*num,2,1,1]
        # print(query_position.float()/h)
        query_value = query_value.view(b*self.groups,-1,1)
        x_value = x.view(b*self.groups,-1,h*w)
        # print(x_value.shape)
        # print(query_value.shape)
        similarity_max = self.get_similarity(x_value, query_value, mode=self.mode)
        # print(similarity_max.shape)
        similarity_gap = self.get_similarity(x_value, self.gap(x).view(b*self.groups,-1,1), mode=self.mode)
        # print(similarity_gap.shape)

        similarity_max = similarity_max.view(b,self.groups,h*w)
        # print(similarity_max.shape)

        Distance = abs(position_mask - query_position)
        # print(Distance.shape)
        Distance = Distance.type(query_value.type())
        # Distance = torch.exp(-Distance * self.theta)
        distribution = Normal(0, self.scale)
        Distance = distribution.log_prob(Distance * self.theta).exp().clone()
        # print(Distance.shape)
        Distance = (Distance.mean(dim=1)).view(b, self.groups, h * w)
        # print(Distance.shape)
        # print_Dis = Distance.mean(dim=0).mean(dim=0).view(h,w)
        # np.savetxt(time.perf_counter().__str__()+'.txt',print_Dis.detach().cpu().numpy())
        # # add e^(-x), means closer more important
        # Distance = torch.exp(-Distance * self.theta)
        # Distance = (self.distance_embedding(Distance)).reshape(b, self.groups, h*w)
        similarity_max = similarity_max*Distance


        similarity_gap = similarity_gap.view(b, self.groups, h*w)
        similarity = similarity_max*self.zero+similarity_gap*self.one



        context = similarity - similarity.mean(dim=2, keepdim=True)
        std = context.std(dim=2, keepdim=True) + 1e-5
        context = (context/std).view(b,self.groups,h,w)
        # affine function
        context = context * self.weight + self.bias
        context = context.view(b*self.groups,1,h,w)\
            .expand(b*self.groups, torch.div(c, self.groups, rounding_mode='floor'), h, w).reshape(b,c,h,w)
        value = x*self.sig(context)

        return value

    def get_position_mask(self,x,b,h,w,number):
        # print(x[0, 0, :, :].shape)
        # print(x[0, 0, :, :] != 2020)
        # print((x[0, 0, :, :] != 2020).shape)
        mask = (x[0, 0, :, :] != 19950203).nonzero()
        # print(mask.shape)
        mask = (mask.reshape(h,w, 2)).permute(2,0,1).expand(b*number,2,h,w)
        return mask


    def get_query_position(self, query,groups):
        b,c,h,w = query.size()
        value = query.view(b*groups,torch.div(c, groups, rounding_mode='floor'),h,w)
        sumvalue = value.sum(dim=1,keepdim=True)
        maxvalue, maxposition = self.max_pool(sumvalue)
        t_position = torch.cat((torch.div(maxposition, w, rounding_mode='floor'), maxposition % w),dim=1)

        t_value = value[torch.arange(b*groups),:,t_position[:,0,0,0],t_position[:,1,0,0]]
        t_value = t_value.view(b, c, 1, 1)
        # print(t_value.shape)
        # print(t_position.shape)
        return t_value, t_position

    def get_similarity(self, query, key_value, mode='dotproduct'):
        if mode == 'dotproduct':
            similarity = torch.matmul(key_value.permute(0, 2, 1), query).squeeze(dim=1)
        elif mode == 'l1norm':
            similarity = -(abs(query - key_value)).sum(dim=1)
        elif mode == 'cosine':
            similarity = torch.cosine_similarity(query,key_value,dim=1)
        else:
            similarity = torch.matmul(key_value.permute(0, 2, 1), query)
        return similarity

def simulate_get_position_mask():
    b = a.reshape(4,4,2)
    print(b.shape)
    print(b)
    c = b.permute(2,0,1)
    print(c.shape)
    print(c)
    d = c.expand(2,2,4,4)
    print(d.shape)
    print(d)

if __name__ == "__main__":
    model = PRMLayer(4)
    x = torch.randn(2, 64, 32, 16)
    y = model(x)

    a = torch.tensor([[1,1],[1,2],[1,3],[1,4],
                      [2,1],[2,2],[2,3],[2,4],
                      [3,1],[3,2],[3,3],[3,4],
                      [4,1],[4,2],[4,3],[4,4]])
