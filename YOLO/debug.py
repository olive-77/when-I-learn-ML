import torch

class Rotator:
    """根据hidden_dim, 和position_ids生成对应的旋转位置编码,
    和论文中定义略有不同, 一个个二维的子空间被
    分割到了前后两部分, 分别进行旋转, 然后拼接起来
    """

    def __init__(self, D, position_ids):
        """
        position_ids: [seq_len], D 和单个头的hidden_dim对应
        """
        base = 10000
        d = D / 2
        B = base ** (1 / d)
        theta_base = 1.0 / (B ** (torch.arange(0, d)))  # 等比数列, $\Theta$
        thetas = position_ids.outer(theta_base)  # [seq_len, D/2]
        full_thetas = torch.cat((thetas, thetas), dim=1)  # [seq_len, D]
        self.cos = full_thetas.cos()
        self.sin = full_thetas.sin()

    def rotate(self, x):
        """
        x: [bs, num_attention_heads, seq_len, D]
        q: [bs, num_attention_heads, seq_len, D]
        cos: [seq_len, D]
        [x,y] @ [[cos, sin], [-sin, cos]] = [x*cos - y*sin, y*cos + x*sin]
        = [x,y] * cos + [-y, x] * sin
        """
        return x * self.cos + Rotator.reverse_half(x) * self.sin

    @staticmethod
    def reverse_half(q):
        """
        q: [bs, num_attention_heads, seq_len, D] trick2
        """
        u = q[..., : q.shape[-1] // 2]  # 认为是各个二维子空间的第一维的向量集结
        v = q[..., q.shape[-1] // 2 :]  # 认为是各个二维子空间的第二维的向量集结
        return torch.cat((-v, u), dim=-1)