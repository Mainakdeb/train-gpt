## utils
import torch 

def copy_parameter(param_official, param_ours):
    """
    copy values from tensor

    Parameters
    ----------
    param_official : torch.Tensor
        the tensor which would be copied

    param_ours : torch.Tensor
        this tensor will be overwritten in-place with the values from 
        param official

    """
    if param_official.shape != param_ours.shape:
        raise ValueError("the shapes of tensors are different")
    
    with torch.no_grad():
        param_ours.copy_(param_official)

def copy_block(block_official, block_ours):
    """Copy all parameters within a transformer block.
    Parameters
    ----------
    block_official : transformers.models.gpt2.modeling_gpt2.GPT2Block
        Block coming from the huggingface code.
    block_ours : Block
        Our block.
    """
    b_a = block_official
    b_b = block_ours

    # LN 1
    copy_parameter(b_a.ln_1.weight, b_b.ln_1.weight)
    copy_parameter(b_a.ln_1.bias, b_b.ln_1.bias)

    # Attention
    copy_parameter(b_a.attn.c_attn.weight.T, b_b.attention.in_proj_weight)
    copy_parameter(b_a.attn.c_attn.bias, b_b.attention.in_proj_bias)

    copy_parameter(b_a.attn.c_proj.weight.T, b_b.attention.out_proj.weight)
    copy_parameter(b_a.attn.c_proj.bias, b_b.attention.out_proj.bias)

    # LN 2
    copy_parameter(b_a.ln_2.weight, b_b.ln_2.weight)
    copy_parameter(b_a.ln_2.bias, b_b.ln_2.bias)

    # MLP
    copy_parameter(b_a.mlp.c_fc.weight.T, b_b.mlp[0].weight)
    copy_parameter(b_a.mlp.c_fc.bias, b_b.mlp[0].bias)

    copy_parameter(b_a.mlp.c_proj.weight.T, b_b.mlp[2].weight)
    copy_parameter(b_a.mlp.c_proj.bias, b_b.mlp[2].bias)


def copy_model(model_official, model_ours):
    """Copy all trainable weights.
    Parameters
    ----------
    model_official : transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
        Huggingface model.
    model_ours : GPT
        Our model.
    """
    m_a = model_official
    m_b = model_ours

    # Token and positional embeddings
    copy_parameter(m_a.transformer.wpe.weight, m_b.pos_emb.weight)
    copy_parameter(m_a.transformer.wte.weight, m_b.token_emb.weight)

    # Blocks
    for block_official, block_ours in zip(m_a.transformer.h, m_b.blocks):
        copy_block(block_official, block_ours)

    # Head
    copy_parameter(m_a.transformer.ln_f.weight, m_b.ln.weight)
    copy_parameter(m_a.transformer.ln_f.bias, m_b.ln.bias)
    copy_parameter(m_a.lm_head.weight, m_b.head.weight)

@torch.no_grad()
def generate_token(model, token_ixs, sample, temperature=1.0, top_k=None):
    """
    Generate a single token, given previous tokens

    Parameters
    ----------
    model : GPT
        our GPT model

    Token_ixs : list
        list of conditional input token ids
    
    temperature : float
        higher temperature -> higher variability
    
    sample : bool
        if True -> sample randomly from distribution, 
        else take argmax
    
    top_k : int or None
        if not None -> modify distribution to have top k most 
        probable outcomes

    Returns
    -------
    new_token_ix : int
        index of new token

    """
    context_token_ixs = token_ixs[-model.n_positions : ]
    ixs = torch.tensor(context_token_ixs).to(dtype=torch.long)[
        None, :
    ] #(1, n_tokens) add dummy patch_dim 
    
    logits_all = model(ixs)
    logits = logits_all[0, -1, :] #take only last token embedding of first batch sample
    logits = logits/temperature
    
    if top_k is not None:
        top_values, _ = torch.topk(logits, top_k)
        logits[logits < top_values.min()] = -torch.inf

    probs = torch.nn.functional.softmax(logits, dim=0)

    if sample:
        new_token_ix = torch.multinomial(probs, num_samples=1)
    else:
        new_token_ix = probs.argmax()

    return(new_token_ix.item())