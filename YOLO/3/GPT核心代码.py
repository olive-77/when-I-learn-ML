


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 一个模块，可以使用键索引到子模块（像字典一样）
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embedding的权重
            wpe=nn.Embedding(config.block_size, config.n_embd),  # position embedding的权重
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 包含多个 Block 的模型列表
            ln_f=nn.LayerNorm(config.n_embd),  # 输出层归一化， gpt-2 需要一个额外的最终层规范
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言模型头的线性层，投射到词汇表大小（分类）

    def forward(self, idx):
        # idx 是输入的 token 索引，形状为 (B, T)
        B, T = idx.size()  # 获取 batch size【输入的样本数量】 和序列长度 【序列长度（T）指的是输入的 token 序列的长度。】
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings 【嵌入层】
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # 将位置索引转换为 long 类型【shape（T）】
        pos_emb = self.transformer.wpe(pos)  # 将位置索引 映射到 位置嵌入【shape(T, n_embd) 】
        tok_emb = self.transformer.wte(idx)  # 将 token 索引 映射到  token 嵌入 【shape(B, T, n_embd) 】
        x = tok_emb + pos_emb  # 将 token 嵌入 和 位置嵌入 相加【shape(B, T, n_embd) 】

        # forward the blocks of the transformer
        for block in self.transformer.h:  # 遍历每个block
            x = block(x)  # 传递输入到block，得到输出

        # 应用最终的层规范 【forward the final layernorm and classifier】
        x = self.transformer.ln_f(x)  # 应用最终的层规范 #传递了这么多个block只进行一次Norm吗
        logits = self.lm_head(x)  # 应用语言模型头，得到 logits
        return logits