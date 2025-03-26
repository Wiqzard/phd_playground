import torch

from titans.titans_pytorch.ttt_custom import TTTConfig, Block, TTTModel, TTTMoELinear


def quick_check_ttt_mlp():
    """
    1) Constructs a single Block that uses TTTMLP internally.
    2) Passes a small random input through it (no caching) just to sanity-check the shape and functionality.
    """
    # Minimal config that uses the TTT-MLP layer
    config = TTTConfig(
        ttt_layer_type="mlp",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        use_cache=False,  # omit cache for a small shape check
    )
    # Single-layer block
    test_block = Block(config, layer_idx=0)

    # Create a random hidden_states tensor of shape [batch_size, seq_len, hidden_size]
    hidden_states = torch.randn(2, 5, config.hidden_size)

    # Forward pass; no position_ids / no caching
    position_ids = torch.arange(
        0, hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
    ).unsqueeze(0)
    out = test_block(
        hidden_states, attention_mask=None, position_ids=position_ids, cache_params=None
    )
    print("[Block with TTTMLP] Output shape:", out.shape)
    # Should be [batch_size, seq_len, hidden_size]


def quick_check_ttt_model():
    """
    1) Constructs a small TTTModel end-to-end that uses TTTMLP blocks.
    2) Passes random input_ids through the model and prints out shapes.
    """
    torch.autograd.set_detect_anomaly(True)
    config = TTTConfig(
        ttt_layer_type="mlp",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        output_ttt_stats=True,
        use_cache=True,  # enabling caching is optional
    )
    model = TTTModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # Random input_ids
    input_ids = torch.randint(
        0, config.vocab_size, (2, 128)
    )  # [batch_size=2, seq_len=5]
    outputs = model(input_ids, use_cache=True)

    loss = outputs.last_hidden_state.mean()
    loss.backward()

    # Optimizer step
    optimizer.step()

    print("[TTTModel] Last hidden state shape:", outputs.last_hidden_state.shape)
    # If caching is on, you can also see that outputs.cache_params exists:
    if outputs.cache_params is not None:
        print(
            "[TTTModel] Cache params object exists:",
            isinstance(outputs.cache_params, TTTCache),
        )


if __name__ == "__main__":
    # quick_check_ttt_mlp()
    quick_check_ttt_model()
