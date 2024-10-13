from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from transformers import SegformerForSemanticSegmentation
from torchvision.models import resnet34, resnet50, resnet18, ResNet18_Weights, ResNet50_Weights, ResNet34_Weights
import copy
import math
try:
    from efficient_atttention import Efficient_Cross_Attention
except ModuleNotFoundError:
    from .efficient_atttention import Efficient_Cross_Attention

# 定义一个clones函数，来更方便的将某个结构复制若干份
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    #首先取query的最后一维的大小，对应词嵌入维度
    d_k = query.size(-1)
    #按照注意力公式，将query与key的转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    #接着判断是否使用掩码张量
    if mask is not None:
        #使用tensor的masked_fill方法，将掩码张量和scores张量每个位置一一比较，如果掩码张量则对应的scores张量用-1e9这个置来替换
        scores = scores.masked_fill(mask == 0, -1e9)
        
    #对scores的最后一维进行softmax操作，使用F.softmax方法，这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim = -1)
    
    #之后判断是否使用dropout进行随机置0
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    #最后，根据公式将p_attn与value张量相乘获得最终的query注意力表示，同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, feature_size, eps=1e-6):
        #初始化函数有两个参数，一个是features,表示词嵌入的维度,另一个是eps它是一个足够小的数，在规范化公式的分母中出现,防止分母为0，默认是1e-6。
        super(LayerNorm, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #根据features的形状初始化两个参数张量a2，和b2，第一初始化为1张量，也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数。因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，使其即能满足规范化要求，又能不改变针对目标的表征，最后使用nn.parameter封装，代表他们是模型的参数
        self.a_2 = nn.Parameter(torch.ones(feature_size, device=device))
        self.b_2 = nn.Parameter(torch.zeros(feature_size, device=device))
        #把eps传到类中
        self.eps = eps

    def forward(self, x):
    #输入参数x代表来自上一层的输出，在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致，接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果。
    #最后对结果乘以我们的缩放参数，即a2,*号代表同型点乘，即对应位置进行乘法操作，加上位移参b2，返回即可
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
    # def __init__(self, h, d_model, dropout=0.1):
        #在类的初始化时，会传入三个参数，h代表头数，d_model代表词嵌入的维度，dropout代表进行dropout操作时置0比率，默认是0.1
        super(MultiHeadedAttention, self).__init__()
        #在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，这是因为我们之后要给每个头分配等量的词特征，也就是embedding_dim/head个
        assert d_model % h == 0
        #得到每个头获得的分割词向量维度d_k
        self.d_k = d_model // h
        #传入头数h
        self.h = h
        
        #创建linear层，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用，为什么是四个呢，这是因为在多头注意力中，Q,K,V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个
        self.linears = clones(nn.Linear(d_model, d_model), 4).cuda()
        #self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        #前向逻辑函数，它输入参数有四个，前三个就是注意力机制需要的Q,K,V，最后一个是注意力机制中可能需要的mask掩码张量，默认是None
        if mask is not None:
            # Same mask applied to all h heads.
            #使用unsqueeze扩展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)
        #接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # 首先利用zip将输入QKV与三个线性层组到一起，然后利用for循环，将输入QKV分别传到线性层中，做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结构进行维度重塑，多加了一个维度h代表头，这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，计算机会根据这种变换自动计算这里的值，然后对第二维和第三维进行转置操作，为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，从attention函数中可以看到，利用的是原始输入的倒数第一和第二维，这样我们就得到了每个头的输入
        # self.linears = self.linears.cuda()
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        # 得到每个头的输入后，接下来就是将他们传入到attention中，这里直接调用我们之前实现的attention函数，同时也将mask和dropout传入其中
        # x, self.attn = attention(query, key, value, mask=mask, dropout=None)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法。这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，所以，下一步就是使用view重塑形状，变成和输入形状相同。  
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        #最后使用线性层列表中的最后一个线性变换得到最终的多头注意力结构的输出
        return self.linears[-1](x)

# # batch_size 为 64，有 12 个词，每个词的 Key 向量是 256 维
# query = torch.rand(64, 12, 256)
# key = torch.rand(64, 12, 256)
# value = torch.rand(64, 12, 256)
# attention_fn = MultiHeadedAttention(h=8, d_model=256, dropout=0.1)

class Fusion_Block(nn.Module):
    def __init__(self, emb_dim, heads, mlp_dim, dropout = 0., height=16):
    # def __init__(self, emb_dim, heads, mlp_dim, dropout = 0., height=16):
        super().__init__()
        self.norm1 = LayerNorm(emb_dim)
        self.cross_attention_1 = MultiHeadedAttention(heads, emb_dim)
        self.cross_attention_2 = MultiHeadedAttention(heads, emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        ).cuda()
        self.dropout2 = nn.Dropout(dropout)
        self.height = height
    
    def forward(self, infrared_features, rgb_features):
        if infrared_features.shape != rgb_features.shape:
            return infrared_features, rgb_features
        else:
            h = self.height
            infrared_features = rearrange(infrared_features, 'b c h w -> b (h w) c', h=h)
            # import pdb; pdb.set_trace()
            rgb_features = rearrange(rgb_features, 'b c h w -> b (h w) c', h=h)
            infrared_memory = infrared_features
            rgb_memory = rgb_features
            infrared_features = self.norm1(infrared_features)
            rgb_features = self.norm1(rgb_features)
            # cross attention for infrared features and rgb features
            infrared_features = self.cross_attention_1(infrared_features, rgb_features, rgb_features)
            rgb_features = self.cross_attention_2(rgb_features, infrared_features, infrared_features)
            infrared_features = self.norm2(self.dropout1(infrared_features) + infrared_memory)
            rgb_features = self.norm2(self.dropout1(rgb_features) + rgb_memory)
            # ffn
            # self.mlp = self.mlp.cuda()
            # infrared_features += self.mlp(infrared_features)
            # rgb_features += self.mlp(rgb_features)
            infrared_features = infrared_features + self.mlp(infrared_features)
            rgb_features = rgb_features + self.mlp(rgb_features)
            infrared_features = rearrange(infrared_features, 'b (h w) c -> b c h w', h=h)
            rgb_features = rearrange(rgb_features, 'b (h w) c -> b c h w', h=h)
            return infrared_features, rgb_features


class Fusion_feature_encoder(nn.Module):
    def __init__(self, infrared_first_conv, infrared_resnet_blocks, segformer_encoder, fusion_block_lists):
        super().__init__()
        '''
            Args:
                infrared_first_conv: {nn.Conv2d, BatchNorm2d, ReLU, MaxPool2d}, the first downsample layer of resnet34 or resnet50, size --> bs, 64, H/4, W/4
                infrared_resnet_blocks: list of [layer1, layer2, layer3, layer4], the resnet layers of resnet34 or resnet50
                    before layer1: size --> bs, 64, H/4, W/4
                    after layer1: size --> bs, 64, H/4, W/4
                    after layer2: size --> bs, 128, H/8, W/8
                    after layer3: size --> bs, 256, H/16, W/16
                    after layer4: size --> bs, 512, H/32, W/32
                segformer_encoder: SegformerForSemanticSegmentation, the encoder of segformer
                fusion_block_lists: list of Fusion_Block, the fusion block lists of each stage of segformer encoder, channels == [64, 128, 320, 512]
        '''
        self.infrared_first_conv = infrared_first_conv
        self.infrared_resnet_blocks = infrared_resnet_blocks
        self.segformer_encoder = segformer_encoder
        self.segformer_encoder = self.segformer_encoder
        self.fusion_block_lists = fusion_block_lists
        self.config = self.segformer_encoder.config
        self.config.reshape_last_stage = True
    
    def forward(self, infrared_images, rgb_images):
        assert infrared_images.shape[0] == rgb_images.shape[0]
        assert infrared_images.shape[2:] == rgb_images.shape[2:]
        batch_size = infrared_images.shape[0]
        ori_H, ori_W = infrared_images.shape[2:]
        infrared_features = self.infrared_first_conv(infrared_images)
        infrared_feature_list = []
        rgb_feature_list = []
        rgb_features = rgb_images
        for idx, x in enumerate(zip(self.segformer_encoder.encoder.patch_embeddings, self.segformer_encoder.encoder.block, self.segformer_encoder.encoder.layer_norm, self.infrared_resnet_blocks, self.fusion_block_lists)):
            embedding_layer, block_layer, norm_layer, res_cnn_block, fusion_block = x
            # first, obtain patch embeddings
            # import pdb; pdb.set_trace()
            rgb_features, height, width = embedding_layer(rgb_features)
            infrared_features = res_cnn_block(infrared_features)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(rgb_features, height, width)
                rgb_features = layer_outputs[0]
                # third, apply layer norm
                rgb_features = norm_layer(rgb_features)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.segformer_encoder.encoder.patch_embeddings) - 1 or (
                idx == len(self.segformer_encoder.encoder.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                rgb_features = rgb_features.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            # third, apply fusion block to fuse rgb features and infrared features
            # infrared_features, rgb_features = fusion_block(rgb_features = rgb_features, infrared_features = infrared_features)
            # efficient cross attention
            # import pdb; pdb.set_trace()
            rgb_features, infrared_features = fusion_block(rgb_features, infrared_features)

            infrared_feature_list.append(infrared_features)
            rgb_feature_list.append(rgb_features)
        
        return infrared_feature_list, rgb_feature_list


class FusionFormer(nn.Module):
    def __init__(self, num_classes, infrared_channels, image_size, heads = 8, mlp_dim = 512,
                 infrared_encoder = 'Resnet34', auxiliary_head = False, **kwargs):
        super().__init__()
        '''
            Args:
                num_classes: int, the number of output classes
                infrared_channels: int, the number of channels of infrared images
                image_size: int, the size of input images
                infrared_encoder: str, the encoder of infrared images, 'Resnet34' or 'Resnet50'
                auxiliary_head: bool, whether to use auxiliary head
        '''
        self.num_classes = num_classes
        self.image_size = image_size
        self.infrared_channels = infrared_channels
        self.auxiliary_head = auxiliary_head
        self.kwargs = kwargs

        # init segformer-b3-finetuned-ade-512-512
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("./pretrained/segformer-b3-finetuned-ade-512-512", local_files_only=True)
        
        rgb_encoder = self.segformer.segformer 
        self.decoder = self.segformer.decode_head

        # init fusion block lists
        emb_dim_list = [64, 128, 320, 512]
        emb_dim_list = [i.proj.weight.shape[0] for i in rgb_encoder.encoder.patch_embeddings]
        height = [self.image_size // 4, self.image_size // 8, self.image_size // 16, self.image_size // 32]
        self.fusion_block_lists = [Fusion_Block(emb_dim=i, heads=heads, mlp_dim=512, height=h) for i, h in zip(emb_dim_list, height)]

        self.fusion_block_lists = [Efficient_Cross_Attention(in_channels=i, value_channels=i, head_count=8, key_channels=i).cuda() for i in emb_dim_list]
        # self.fusion_block_lists = [Fusion_Block(emb_dim=i, heads=8, mlp_dim=512, height=h) for i, h in zip(emb_dim_list, height)]

        # init infrared encoder as resnet34 or resnet50
        self.infrared_encoder = {
            'Resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
            'Resnet34': resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
            'Resnet50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)}[infrared_encoder]
        # print('infrared_encoder:', self.infrared_encoder)
        self.infrared_first_conv = nn.Conv2d(infrared_channels, self.infrared_encoder.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.infrared_encoder.conv1 = self.infrared_first_conv
        infrared_encoder_first_conv = nn.Sequential(*list(self.infrared_encoder.children())[0:4])   
        resnet_blocks = [self.infrared_encoder.layer1, self.infrared_encoder.layer2, self.infrared_encoder.layer3, self.infrared_encoder.layer4]
        
        # init fusion encoder
        self.fusion_encoder = Fusion_feature_encoder(infrared_first_conv=infrared_encoder_first_conv, 
                                                     infrared_resnet_blocks=resnet_blocks, 
                                                     segformer_encoder=rgb_encoder, fusion_block_lists=self.fusion_block_lists)
        # get resnet last stage dim
        # import pdb; pdb.set_trace()
        try:
            last_stage_dim = resnet_blocks[-1][-1].conv3.out_channels
        except AttributeError:
            last_stage_dim = resnet_blocks[-1][-1].conv2.out_channels
        # print(last_stage_dim)
        self.out = nn.Conv2d(768, self.num_classes, kernel_size=1)
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(last_stage_dim, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(256, self.num_classes, kernel_size=1))
        
    def decode(self, encoder_hidden_states):
    # def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.decoder.linear_c):
            if self.decoder.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            # all_hidden_states += (encoder_hidden_state,)
            all_hidden_states = all_hidden_states + (encoder_hidden_state,)
        concat_hidden_states = torch.cat(all_hidden_states[::-1], dim=1)
        hidden_states = self.decoder.linear_fuse(concat_hidden_states)
        hidden_states = self.decoder.batch_norm(hidden_states)
        hidden_states = self.decoder.activation(hidden_states)
        hidden_states = self.decoder.dropout(hidden_states)
        logits = self.out(hidden_states)

        return logits 

    def forward(self, infrared_images, rgb_images):
        output_attentions = False
        middle_layer = []
        assert infrared_images.shape[0] == rgb_images.shape[0]
        bs, c, h, w = infrared_images.shape
        batch_size = infrared_images.shape[0]
        hidden_states = rgb_images
        # get the infrared features
        hidden_states_infrared = infrared_images
        infrared_features, rgb_features = self.fusion_encoder(infrared_images, rgb_images)
        # print(len(infrared_features), len(rgb_features))
        logits = self.decode(rgb_features)
        upsampled_logits = nn.functional.interpolate(
            logits, size=(h,w), mode="bilinear", align_corners=False
        )
        if self.training:
            # import pdb; pdb.set_trace()
            # upsample infrared features[-1] to he upsampled_logits size
            # calculate the loss as auxiliary loss
            infrared_logits = self.aux(infrared_features[-1])
            # import pdb; pdb.set_trace()
            infrared_logits = nn.functional.interpolate(
                infrared_logits, size=(h,w), mode="bilinear", align_corners=False
            )
            return [upsampled_logits, infrared_logits]
        return upsampled_logits
        



if __name__ == '__main__':
    # fusion = Fusion_Block(256, 8, 256, dropout = 0., height=56)
    # infrared_features = torch.rand(3, 256, 56, 56)
    # rgb_features = torch.rand(3, 256, 56, 56)
    # infrared_features, rgb_features = fusion(infrared_features, rgb_features)
    # print(infrared_features.shape, rgb_features.shape)
    image_size = 480
    batch_size = 1
    model = FusionFormer(num_classes=2, infrared_channels=1, image_size=image_size, heads=4, infrared_encoder='Resnet34', auxiliary_head=False)
    # print(model(rgb_images = torch.rand(3, 3, 224, 224), infrared_images = torch.rand(3, 1, 224, 224)).shape)
    # print(model(rgb_images = torch.rand(batch_size, 3, image_size, image_size), infrared_images = torch.rand(batch_size, 1, image_size, image_size))[0].shape)
    from torchinfo import summary
    summary(model, input_size=[(batch_size, 1, image_size, image_size), (batch_size, 3, image_size, image_size)], device='cuda')
    # print(model)
    # torch.Size([64, 12, 256]) torch.Size([64, 12, 256])
    # print(resnet34(pretrained=True))
    # resnet