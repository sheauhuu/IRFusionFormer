from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import copy
import math

class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class Efficient_Cross_Attention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels, inverse=False):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        print('using cross attention with efficient attention')
        self.keys_rgb = nn.Conv2d(in_channels, key_channels, 1)
        self.keys_ir = nn.Conv2d(in_channels, key_channels, 1)
        self.queries_rgb = nn.Conv2d(in_channels, key_channels, 1)
        self.queries_ir = nn.Conv2d(in_channels, key_channels, 1)
        self.values_rgb = nn.Conv2d(in_channels, value_channels, 1)
        self.values_ir = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection_rgb = nn.Conv2d(value_channels, in_channels, 1)
        self.reprojection_ir = nn.Conv2d(value_channels, in_channels, 1)
        self.inverse = inverse

    def forward(self, input_rgb, input_ir):
        if input_rgb.shape != input_ir.shape:
            return input_rgb, input_ir
        else:
            n, _, h, w = input_rgb.size()
            keys_rgb = self.keys_rgb(input_rgb).reshape((n, self.key_channels, h * w))
            queries_rgb = self.queries_rgb(input_rgb).reshape(n, self.key_channels, h * w)
            values_rgb = self.values_rgb(input_rgb).reshape((n, self.value_channels, h * w))

            keys_ir = self.keys_ir(input_ir).reshape((n, self.key_channels, h * w))
            queries_ir = self.queries_ir(input_ir).reshape(n, self.key_channels, h * w)
            values_ir = self.values_ir(input_ir).reshape((n, self.value_channels, h * w))

            head_key_channels = self.key_channels // self.head_count
            head_value_channels = self.value_channels // self.head_count

            attended_values_rgb = []
            attended_values_ir = []
            # use ir key and value with rgb query
            for i in range(self.head_count):
                key = F.softmax(keys_ir[
                    :,
                    i * head_key_channels: (i + 1) * head_key_channels,
                    :
                ], dim=2)
                query = F.softmax(queries_rgb[
                    :,
                    i * head_key_channels: (i + 1) * head_key_channels,
                    :
                ], dim=1)
                value = values_ir[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                ]
                context = key @ value.transpose(1, 2)
                attended_value = (
                    context.transpose(1, 2) @ query
                ).reshape(n, head_value_channels, h, w)
                if self.inverse:
                    attended_values_ir.append(attended_value)
                else:
                    attended_values_rgb.append(attended_value)
            
            # use rgb key and value with ir query
            for i in range(self.head_count):
                key = F.softmax(keys_rgb[
                    :,
                    i * head_key_channels: (i + 1) * head_key_channels,
                    :
                ], dim=2)
                query = F.softmax(queries_ir[
                    :,
                    i * head_key_channels: (i + 1) * head_key_channels,
                    :
                ], dim=1)
                value = values_rgb[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                ]
                context = key @ value.transpose(1, 2)
                attended_value = (
                    context.transpose(1, 2) @ query
                ).reshape(n, head_value_channels, h, w)
                if self.inverse:
                    attended_values_rgb.append(attended_value)
                else:
                    attended_values_ir.append(attended_value)
            
            aggregated_values_rgb = torch.cat(attended_values_rgb, dim=1)
            aggregated_values_ir = torch.cat(attended_values_ir, dim=1)
            reprojected_value_rgb = self.reprojection_rgb(aggregated_values_rgb)
            reprojected_value_ir = self.reprojection_ir(aggregated_values_ir)
            attention_rgb = reprojected_value_rgb + input_rgb
            attention_ir = reprojected_value_ir + input_ir

            return attention_rgb, attention_ir

if __name__ == '__main__':
    input_ = torch.randn(1, 3, 64, 64)
    model = EfficientAttention(3, 64, 8, 64)
    output = model(input_)
    print(output.shape)
    rgb_input = torch.randn(8, 1024, 128, 128)
    ir_input = torch.randn(8, 1024, 128, 128)
    model = Efficient_Cross_Attention(1024, 1024 // 4, 8, 1024)
    output = model(rgb_input, ir_input)
    print(output[0].shape, output[1].shape)
