import torch
import torch.nn as nn

from transformers.activations import gelu_new

class CustomGELU(nn.Module):
    #GeLU implementation from huggingface transformers

    def forward(self, x):
        return gelu_new(x)


class Block(nn.Module):
    '''
    Decoder Block

    Params
    ------
    n_embd : int
        embd dimensions

    n_head : int
        number of attention heads
    
    n_positions : int
        max number of tokens
    
    attn_pdrop : float
        probability of dropout on attention weights
    
    resid_pdrop : float
        probability of dropout after multi-layer-perceptron
    
    layer_norm_epsilon : float
        hyperparameter for layer norlamization

    attributes
    ----------

    ln_1, ln_2 : nn.Layernorm
        Layer normalization

    attention : nn.MultiHeadAttention
        attention module

    mlp : nn.Sequential
        Multilayer Perceptron

    '''

    def __init__(
        self, 
        *,
        n_embd, 
        n_head, 
        n_positions,
        attn_pdrop,
        resid_pdrop,
        layer_norm_epsilon
    ):

        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

        self.attention = nn.MultiHeadAttention(
            embd_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            bias=True,
            batch_first=True
        )
        
        self.register_buffer(
            "mask",
            (1-torch.tril(torch.ones(n_positions, n_positions))).to(dtype=torch.bool) #lower triangular matrix
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            CustomGELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        def forward(self, x):
            """
            Forward pass

            parameters
            ----------
            x : torch.Tensor
                input tensor of shape (batch_size, n_tokens, n_embd)

            returns
            -------
            torch.Tensor
                output tensor of shape (batch_size, n_tokens, n_embd)

            """

            batch_size, n_tokens, n_embd = x.shape
            x_ = self.ln_1(x) #layer norm
            mask = self.mask[:n_tokens, :n_tokens]

            attn_out, _ = self.attention(
                x_, x_, x_, attn_mask = mask, need_weights = False
            )

            x = x + attn_out
            x = self.mlp(self.ln_2(x))

            return(x)


class GPT(nn.Module):
    """

    parameters
    ----------

    vocab_size : int
        number of tokens on vocabulary

    n_layer : int
        number of decoder blocks to include

    n_embd : int
        dimensionality of embeddings

    n_head : int
        number of attention heads

    n_positions : int
        maximum number of tokens

    attn_pdrop : float
        probability of dropout on attention weights

    embd_pdrop : float
        Probability of dropout on the sum of embeddings

    resid_pdrop : float
        Probability of dropout after applying the MLP

    layer_norm_epsilon : float
        Hyperparameter of layer normalization
    Attributes
    ----------
    token_emb : nn.Embedding
        Token embeddings

    pos_emb : nn.Embedding
        Positional embedding

    drop : nn.Dropout
        Dropout module to be applied on the sum of embeddings

    blocks : nn.Sequential
        List of decoder blocks

    ln : nn.LayerNorm
        Layer norm applied before applying head

    head : nn.Linear
        Final linear layer.
    """

    def __init__(
        self,
        *,
        vocab_size,
        n_layer,
        n_embd,
        n_head,
        n_positions,
        attn_pdrop,
        embd_pdrop,
        resid_pdrop,
        layer_norm_epsilon,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(n_positions, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    n_positions=n_positions,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        """forward pass

        parameters
        ----------
        idx : torch.Tensor
            Integer tensor of shape (batch_size, n_tokens) where each
            element is in the range [0, vocab_size)

        returns
        -------
        logits : torch.Tensor
            Tensor of shape (batch_size, n_tokens, vocab_size)
        """

        batch_size, n_tokens = idx.shape
        device = idx.device

        if n_tokens > self.n_positions:
            raise ValueError("There are too many tokens in the input")

        positions = torch.arange(n_tokens, device=device)  # (n_tokens,)

        token_emb = self.token_emb(idx)  # (batch_size, n_tokens, n_embd)
        pos_emb = self.pos_emb(positions)[None, ...]  # (1, n_tokens, n_embd)
        x = self.drop(token_emb + pos_emb)  # (batch_size, n_tokens, n_embd)
        x = self.blocks(x)  # (batch_size, n_tokens, n_embd)
        x = self.ln(x)  # (batch_size, n_tokens, n_embd)
        logits = self.head(x)  # (batch_size, n_tokens, vocab_size)

        return logits
