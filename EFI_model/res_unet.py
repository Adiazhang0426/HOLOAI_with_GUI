import torch.nn as nn
import torch
def Downsample(dim):
    return nn.Conv2d(dim, dim, (4,4), (2,2), (1,1))

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, (4,4), (2,2), (1,1))

class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, (3,3), padding = 1)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x
class ResnetBlock(nn.Module):
    """Deep Residual Learning for Image Recognition"""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, (1,1)) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)
class Unet(nn.Module):
    def __init__(self,input_dim, output_dim, group,max_dim):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim = output_dim
        self.group=group
        self.max_dim=max_dim


        self.downdimlist=sorted([int(self.max_dim/2**i) for i in range(group)])
        self.updiminlist = [self.max_dim*2,self.max_dim]
        self.updimoutlist = [int(self.max_dim / 2 ** i) for i in range(1,group)]
        self.init_conv = nn.Conv2d(self.input_dim, self.downdimlist[0], 7, padding=3)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for j in range(len(self.downdimlist)-1):
            if j==len(self.downdimlist)-2:
                self.is_last=True
            else:
                self.is_last=False
            self.downs.append(
                nn.ModuleList(
                    [
                       ResnetBlock(self.downdimlist[j],self.downdimlist[j+1]),
                       ResnetBlock(self.downdimlist[j+1], self.downdimlist[j + 1]),
                        Downsample(self.downdimlist[j+1]) if not self.is_last else nn.Identity()
                    ]))

        # self.mid=ResnetBlock(self.downdimlist[-1], self.downdimlist[-1]*2)
        # self.mid2 = ResnetBlock(self.downdimlist[-1]*2, self.downdimlist[-1])
        for upin,upout in zip(self.updiminlist,self.updimoutlist):
            if upout==self.updimoutlist[-1]:
                self.is_last=True
            else:
                self.is_last=False
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(upin, upout),
                        ResnetBlock(upout,upout),
                        Upsample(upout) if not self.is_last else nn.Identity()
                    ]))

        self.final_conv = nn.Conv2d(self.updimoutlist[-1], self.output_dim,1)
    def forward(self, x):
        x = self.init_conv(x)
        h1 = []
        downnum=0
        # downsample
        for block1, block2,downsample in self.downs:
            if downnum<=1:
                x = block1(x)
                # x = block2(x)
                h1.append(x)
                x = downsample(x)
            else:
                x = block1(x)
                h1.append(x)
                x = downsample(x)
            downnum+=1

        # bottleneck
        # x = self.mid(x)
        # x = self.mid2(x)

        # upsample

        upnum=0
        for block1, block2,upsample in self.ups:
            if upnum<=1:
                x = torch.cat((x, h1.pop()), dim=1)
                x = block1(x)
                # x = block2(x)
                x = upsample(x)
            else:
                x = torch.cat((x, h1.pop()), dim=1)
                x = block1(x)
                x = upsample(x)
            upnum+=1
        return self.final_conv(x)
if __name__=='__main__':
    model = Unet(1,2,4,64
    )
    print(model)